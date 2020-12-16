"""Trainining script for seq2seq text-to-speech synthesis model.

usage: train.py [options]

options:
    --data-root=<dir>            Directory contains preprocessed features.
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --waveglow_path=<path>       waveglow_path.
    --restore-parts=<path>       Restore part of the model.
    --log-event-path=<name>      Log event path.
    --reset-optimizer            Reset optimizer.
    --load-embedding=<path>      Load embedding from checkpoint.
    --speaker-id=<N>             Use specific speaker of data in case for multi-speaker datasets.
    -h, --help                   Show this help message and exit
"""
from docopt import docopt

import sys
import gc
import platform
from os.path import dirname, join
from tqdm import tqdm, trange
from datetime import datetime

# The deepvoice3 model
from deepvoice3_pytorch import frontend, builder
import audio
import lrschedule

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler
import numpy as np

from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from os.path import join, expanduser
import random

import librosa.display
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
import os
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from matplotlib import cm
from warnings import warn
from hparams import hparams, hparams_debug_string
import training_module as tm
from training_module import TextDataSource, MelSpecDataSource, F0DataSource

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False

_frontend = None  # to be set later

class PyTorchDataset(object):
    def __init__(self, X, Mel, F0):
        self.X = X
        self.Mel = Mel
        self.F0 = F0
        # alias
        self.multi_speaker = X.file_data_source.multi_speaker

    def __getitem__(self, idx):
        if self.multi_speaker:
            text, speaker_id = self.X[idx]
            return text, self.Mel[idx], self.F0[idx], speaker_id
        else:
            return self.X[idx], self.Mel[idx], self.F0[idx]


    def __len__(self):
        return len(self.X)


def collate_fn(batch):
    """Create batch"""
    r = hparams.outputs_per_step
    multi_speaker = len(batch[0]) == 4

    # Lengths
    # Find the maximum length (text)
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)

    # Find the maximum length (mel)
    target_lengths = [len(x[1]) for x in batch]
    max_target_len = max(target_lengths)

    # Find the length (F0)
    world_lengths = [len(x[2]) for x in batch]

    # Convert to a multiple of r (text)
    if max_input_len % r != 0:
        max_input_len += r - max_input_len % r
        assert max_input_len % r == 0

    # Convert to a multiple of r (mel)
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    # Set 0 for zero beginning padding
    # imitates initial decoder states
    b_pad = r
    max_target_len += b_pad # TODO: b_padを足し合わせる意義
    max_world_len = int(max_target_len * hparams.world_upsample)

    # padding (text)
    a = np.array([tm._pad(x[0], max_input_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)
    target_lengths = torch.LongTensor(target_lengths)
    world_lengths = torch.LongTensor(world_lengths)

    # padding (mel)
    b = np.array([tm._pad_2d(x[1], max_target_len, b_pad=b_pad) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    # padding (F0)
    rw = int(r * hparams.world_upsample)
    d = np.array([np.pad(x[2], (rw, max_world_len - len(x[2]) - rw), mode="constant") for x in batch], dtype=np.float32)
    f0_batch = torch.FloatTensor(d)

    # text positions
    text_positions = np.array([tm._pad(np.arange(1, len(x[0]) + 1), max_input_len)
                               for x in batch], dtype=np.int)
    text_positions = torch.LongTensor(text_positions)

    max_decoder_target_len = max_target_len // r

    # frame positions
    s, e = 1, max_decoder_target_len + 1
    # if b_pad > 0:
    #    s, e = s - 1, e - 1
    frame_positions = torch.arange(s, e).long().unsqueeze(0).expand(
        len(batch), max_decoder_target_len).clone()

    # done flags
    done = np.array([tm._pad(np.zeros(len(x[1]) // r - 1),
                          max_decoder_target_len, constant_values=1)
                     for x in batch])
    done = torch.FloatTensor(done).unsqueeze(-1)

    if multi_speaker:
        speaker_ids = torch.LongTensor([x[3] for x in batch])
    else:
        speaker_ids = None

    return x_batch, input_lengths, mel_batch, f0_batch, \
        (text_positions, frame_positions), done, target_lengths, speaker_ids

def collate_fn_norm_lj(batch):
    """Create batch"""
    r = hparams.outputs_per_step
    multi_speaker = len(batch[0]) == 4

    # Lengths
    # Find the maximum length (text)
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)

    # Find the maximum length (mel)
    target_lengths = [len(x[1]) for x in batch]
    max_target_len = max(target_lengths)

    # Find the maximum length (F0)
    world_lengths = [len(x[2]) for x in batch]
    max_world_len = max(world_lengths)

    # Convert to a multiple of r (text)
    if max_input_len % r != 0:
        max_input_len += r - max_input_len % r
        assert max_input_len % r == 0

    # Convert to a multiple of r (mel)
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    if max_world_len % r != 0:
        max_world_len += r - max_world_len % r
        assert max_world_len % r == 0

    # Set 0 for zero beginning padding
    # imitates initial decoder states
    b_pad = r
    c_pad = r
    max_target_len += b_pad # TODO: b_padを足し合わせる意義
    max_world_len += c_pad
    # max_world_len = int(max_target_len * hparams.world_upsample)

    # padding (text)
    a = np.array([tm._pad(x[0], max_input_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)
    target_lengths = torch.LongTensor(target_lengths)
    world_lengths = torch.LongTensor(world_lengths)

    # padding (mel)
    b = np.array([tm._pad_2d(x[1], max_target_len, b_pad=b_pad) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    # padding (F0)
    # rw = int(r * hparams.world_upsample)
    d = np.array([np.pad(x[2], (c_pad, max_world_len - len(x[2]) - c_pad), mode="constant") for x in batch], dtype=np.float32)
    f0_batch = torch.FloatTensor(d)

    # text positions
    text_positions = np.array([tm._pad(np.arange(1, len(x[0]) + 1), max_input_len)
                               for x in batch], dtype=np.int)
    text_positions = torch.LongTensor(text_positions)

    max_decoder_target_len = max_target_len // r

    # frame positions
    s, e = 1, max_decoder_target_len + 1
    # if b_pad > 0:
    #    s, e = s - 1, e - 1
    frame_positions = torch.arange(s, e).long().unsqueeze(0).expand(
        len(batch), max_decoder_target_len).clone()

    # done flags
    done = np.array([tm._pad(np.zeros(len(x[1]) // r - 1),
                          max_decoder_target_len, constant_values=1)
                     for x in batch])
    done = torch.FloatTensor(done).unsqueeze(-1)

    if multi_speaker:
        speaker_ids = torch.LongTensor([x[3] for x in batch])
    else:
        speaker_ids = None

    return x_batch, input_lengths, mel_batch, f0_batch, \
        (text_positions, frame_positions), done, target_lengths, speaker_ids


#TODO: Neural Vocoderでテストできるようにする
def eval_model(global_step, writer, device, model, checkpoint_dir, ismultispeaker):
    # harded coded
    texts = [
        "And debtors might practically have as much as they liked%if they could only pay for it.",
        "There's a way to measure the acute emotional intelligence that has never gone out of style.",
        "President trump met with other leaders at the group of 20 conference.",
        "Generative adversarial network or variational auto encoder.",
        "Please call stella.",
        "Some have accepted this as a miracle without any physical explanation.",
    ]
    import synthesis
    synthesis._frontend = _frontend

    eval_output_dir = join(checkpoint_dir, "eval")
    os.makedirs(eval_output_dir, exist_ok=True)

    # Prepare model for evaluation
    model_eval = tm.build_model().to(device)
    model_eval.load_state_dict(model.state_dict())

    # hard coded
    speaker_ids = [0, 1, 10] if ismultispeaker else [None]
    for speaker_id in speaker_ids:
        speaker_str = "multispeaker{}".format(speaker_id) if speaker_id is not None else "single"

        for idx, text in enumerate(texts, 1):
            model_eval.eval()
            model_eval.make_generation_fast_()

            sequence = np.array(_frontend.text_to_sequence(text, p=0.5))
            #import pdb; pdb.set_trace()
            sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
            text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(device)
            speaker_ids = None if speaker_id is None else torch.LongTensor([speaker_id]).to(device)

            # Greedy decoding
            with torch.no_grad():
                mel, alignments, done = model_eval(
                    sequence, text_positions=text_positions, speaker_ids=speaker_ids)
            alignments = alignments[0].cpu().data.numpy()
            mel = mel[0].cpu().data.numpy()
            #mel = audio._denormalize(mel)

            # Alignment
            for i, alignment in enumerate(alignments, 1):
                alignment_dir = join(eval_output_dir, "alignment_layer{}".format(i))
                os.makedirs(alignment_dir, exist_ok=True)
                path = join(alignment_dir, "step{:09d}_text{}_{}_layer{}_alignment.png".format(
                    global_step, idx, speaker_str, i))
                tm.save_alignment(path, alignment, global_step)
                tag = "eval_text_{}_alignment_layer{}_{}".format(idx, i, speaker_str)
                writer.add_image(tag, np.uint8(cm.viridis(np.flip(alignment, 1)) * 255).T, global_step)

            # Mel
            writer.add_image("(Eval) Predicted mel spectrogram text{}_{}".format(idx, speaker_str),
                             tm.prepare_spec_image(mel).transpose(2,0,1), global_step)




def train(device, model, data_loader, optimizer, writer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          max_clip=100,
          clip_thresh=1.0,
          waveglow_path=None):
    r = hparams.outputs_per_step
    current_lr = init_lr

    match_frame_size = True

    # loss function
    binary_criterion = nn.BCELoss()
    l1 = nn.L1Loss()

    #save用にNoneデータを準備
    converter_outputs, y = None, None


    global global_step, global_epoch
    while global_epoch < nepochs:
        running_f0_loss = 0.
        running_pre_mel_loss = 0.
        running_post_mel_loss = 0
        running_loss = 0.
        print("{}epoch:".format(global_epoch))

        for step, (x, input_lengths, mel, f0, positions, done, target_lengths,
                   speaker_ids) \
                in tqdm(enumerate(data_loader)):
            model.train()
            ismultispeaker = speaker_ids is not None
            # Learning rate schedule
            if hparams.lr_schedule is not None:
                lr_schedule_f = getattr(lrschedule, hparams.lr_schedule)
                current_lr = lr_schedule_f(
                    init_lr, global_step, **hparams.lr_schedule_kwargs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            optimizer.zero_grad()

            # Used for Position encoding
            text_positions, frame_positions = positions

            # Lengths
            input_lengths = input_lengths.long().numpy()
            decoder_lengths = target_lengths.long().numpy() // r

            max_seq_len = max(input_lengths.max(), decoder_lengths.max())
            if max_seq_len >= hparams.max_positions:
                raise RuntimeError(
                    """max_seq_len ({}) >= max_posision ({})
Input text or decoder targget length exceeded the maximum length.
Please set a larger value for ``max_position`` in hyper parameters.""".format(
                        max_seq_len, hparams.max_positions))

            # Transform data to CUDA device
            x = x.to(device)
            text_positions = text_positions.to(device)
            frame_positions = frame_positions.to(device)
            f0 = f0.to(device)
            mel, done = mel.to(device), done.to(device)
            target_lengths = target_lengths.to(device)
            speaker_ids = speaker_ids.to(device) if ismultispeaker else None

            # f0_size = f0.size()
            # np.savetxt("np_f0_input.txt", f0.to('cpu').detach().numpy().copy())
            # f0 = f0.to(device)
            # mel_size = mel.size()

            # model output (postnet version)
            mel_outputs, mel_outputs_postnet, f0_outputs, attn, done_hat = model(
                x, mel, f0, speaker_ids=speaker_ids,
                text_positions=text_positions, frame_positions=frame_positions,
                input_lengths=input_lengths)
            # reshape
            mel_outputs = mel_outputs.view(len(mel), -1, mel.size(-1))
            mel_outputs_postnet = mel_outputs_postnet.view(len(mel), -1, mel.size(-1))

            # Losses
            # mel:
            mel_loss = l1(mel_outputs[:, :-r, :], mel[:, r:, :])
            mel_postnet_loss = l1(mel_outputs_postnet[:, :-r, :], mel[:, r:, :])
            # done:
            done_loss = binary_criterion(done_hat, done)
            # f0:
            if not match_frame_size:
                rw = int(r * hparams.world_upsample)
                f0_loss = l1(f0_outputs[:, :-rw], f0[:, rw:])
            else:
                f0_loss = l1(f0_outputs[:, :-r], f0[:, r:])

            # combine Losses
            loss = mel_loss + mel_postnet_loss + done_loss + f0_loss

            if global_epoch == 0 and global_step == 0:
                tm.save_states(
                    global_step, writer, mel_outputs, mel_outputs_postnet, f0_outputs, converter_outputs, attn,
                    mel, f0, y, input_lengths, checkpoint_dir)
                tm.save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)


            # Update
            loss.backward()
            if clip_thresh > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.get_trainable_parameters(), max_clip)
                grad_value = torch.nn.utils.clip_grad_value_(
                    model.get_trainable_parameters(),clip_thresh)
            optimizer.step()

            # Logs
            # writer.add_scalar("loss", float(loss.item()), global_step)
            writer.add_scalar("done_loss", float(done_loss.item()), global_step)
            # writer.add_scalar("mel_l1_loss (pre)", float(mel_loss.item()), global_step)
            # writer.add_scalar("mel_l1_loss (post)", float(mel_postnet_loss.item()), global_step)
            # writer.add_scalar("f0_l1_loss", float(f0_loss.item()), global_step)
            if clip_thresh > 0:
                writer.add_scalar("gradient norm", grad_norm, global_step)
            writer.add_scalar("learning rate", current_lr, global_step)

            global_step += 1
            running_f0_loss += f0_loss.item()
            running_pre_mel_loss += mel_loss.item()
            running_post_mel_loss += mel_postnet_loss.item()
            running_loss += loss.item()

            if global_step > 0 and global_step % checkpoint_interval == 0:
                tm.save_states(
                    global_step, writer, mel_outputs, mel_outputs_postnet, f0_outputs, converter_outputs, attn,
                    mel, f0, y, input_lengths, checkpoint_dir, waveglow_path=waveglow_path, device=device)
                tm.save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step > 1e5 and global_step % hparams.eval_interval == 0 :
                if waveglow_path is not None:
                    tm.eval_model(global_step, writer, device, model, checkpoint_dir, ismultispeaker, waveglow_path=waveglow_path)
                else:
                    eval_model(global_step, writer, device, model, checkpoint_dir, ismultispeaker)

        averaged_f0_loss = running_f0_loss / (len(data_loader))
        averaged_pre_mel_loss = running_pre_mel_loss / (len(data_loader))
        averaged_post_mel_loss = running_post_mel_loss / (len(data_loader))
        averaged_loss = running_loss / (len(data_loader))
        writer.add_scalar("f0_loss (per epoch)", averaged_f0_loss, global_epoch)
        writer.add_scalar("mel_loss (per epoch)", averaged_pre_mel_loss, global_epoch)
        writer.add_scalar("mel_loss_post (per epoch)", averaged_post_mel_loss, global_epoch)
        writer.add_scalar("loss (per epoch)", averaged_loss, global_epoch)
        print("Loss: {}".format(running_loss / (len(data_loader))))

        global_epoch += 1

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    waveglow_path = args['--waveglow_path']
    load_embedding = args["--load-embedding"]
    checkpoint_restore_parts = args["--restore-parts"]
    speaker_id = args["--speaker-id"]
    speaker_id = int(speaker_id) if speaker_id is not None else None
    preset = args["--preset"]


    data_root = args["--data-root"]
    if data_root is None:
        data_root = join(dirname(__file__), "data", "ljspeech")

    log_event_path = args["--log-event-path"]
    reset_optimizer = args["--reset-optimizer"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())

    # Override hyper parameters
    hparams.parse(args["--hparams"])

    # Preventing Windows specific error such as MemoryError
    # Also reduces the occurrence of THAllocator.c 0x05 error in Widows build of PyTorch
    if platform.system() == "Windows":
        print(" [!] Windows Detected - IF THAllocator.c 0x05 error occurs SET num_workers to 1")

    assert hparams.name == "deepvoice3"
    print(hparams_debug_string())

    # Specify preprocessing function (Depends on language)
    _frontend = getattr(frontend, hparams.frontend)

    # Creating a checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Input dataset definitions
    # Todo: FilesourceDatasetで何が行われているのかを調査
    X = FileSourceDataset(TextDataSource(data_root, speaker_id))
    Mel = FileSourceDataset(MelSpecDataSource(data_root, speaker_id))
    F0 = FileSourceDataset(F0DataSource(data_root, speaker_id))

    # Prepare sampler
    frame_lengths = Mel.file_data_source.frame_lengths
    sampler = tm.PartialyRandomizedSimilarTimeLengthSampler(
        frame_lengths, batch_size=hparams.batch_size)

    # Dataset and Dataloader setup
    dataset = PyTorchDataset(X, Mel, F0)
    data_loader = data_utils.DataLoader(
        dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, sampler=sampler,
        collate_fn=collate_fn_norm_lj, pin_memory=hparams.pin_memory)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = tm.build_model().to(device)

    optimizer = optim.Adam(model.get_trainable_parameters(),
                           lr=hparams.initial_learning_rate, betas=(
        hparams.adam_beta1, hparams.adam_beta2),
        eps=hparams.adam_eps, weight_decay=hparams.weight_decay,
        amsgrad=hparams.amsgrad)

    if checkpoint_restore_parts is not None:
        tm.restore_parts(checkpoint_restore_parts, model)

    # Load checkpoints
    if checkpoint_path is not None:
        model, global_step, global_epoch = tm.load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer)

    # Load embedding
    if load_embedding is not None:
        print("Loading embedding from {}".format(load_embedding))
        tm._load_embedding(load_embedding, model)

    # Setup summary writer for tensorboard
    if log_event_path is None:
        if platform.system() == "Windows":
            log_event_path = "log/run-test" + \
                str(datetime.now()).replace(" ", "_").replace(":", "_")
        else:
            log_event_path = "log/run-test" + str(datetime.now()).replace(" ", "_")
    print("Log event path: {}".format(log_event_path))
    writer = SummaryWriter(log_event_path)

    # Train!
    try:
        train(device, model, data_loader, optimizer, writer,
              init_lr=hparams.initial_learning_rate,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs,
              max_clip=hparams.max_clip,
              clip_thresh=hparams.clip_thresh,
              waveglow_path=waveglow_path)
    except KeyboardInterrupt:
        tm.save_checkpoint(
            model, optimizer, global_step, checkpoint_dir, global_epoch)

    print("Finished")
    sys.exit(0)
