# Deepvoice3の再現実装

postnetを追加する上で行った変更点は
- `deepvoice3.py`の`Decoder`の`forward`内にGLU層が5層のpostnetを追加(`incremental_forward`にも同様に追加)
- `__init__.py`の`AttentionSeq2Seq`と`MultiSpeakerSeq2seq`内でpostnetの出力を受け取ることができるように変更
- `train_module.py`の`eval_model`と`save_states`でpostnetの出力を使用するように変更
- `train_seq2seq.py`の`train`でpostnetの出力の誤差を考慮して学習が進むように変更
- `synthesis.py`の`tts_use_waveglow`でpostnet後の音声を生成するように変更

1. [arXiv:1710.07654](https://arxiv.org/abs/1710.07654): Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning.

# use_waveglow Repository

[WaveGlow](https://github.com/NVIDIA/waveglow) を利用できるように修正．ただし，こちらのRepositoryをforkして，それをsubmoduleとして扱えるようにWaveGlowも修正している．
masterとの変更点として
- WaveGlowに合わせてメルスペクトログラムのfft_size等を調整
- メルスペクトログラムを正規化せずに学習
- このRepositoryでは，waveglowを利用する前提のため，Linear, WORLDの学習データは出力しない
- LJSpeechのみ対応


## Setup
### cloneする場合

```
git clone https://github.com/mitsu-h/deepvoice3
cd deepvoice3
git submodule init
git submodule update
```

### master Repositoryをforkしている場合

```
git remote add upstream git://github.com/mitsu-h/deepvoice3
git fetch upstream
git branch use_waveglow
git checkout use_waveglow
git merge upstream/use_waveglow
```

## 学習済みモデル
- [本家の学習済みデータ](https://drive.google.com/file/d/1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF/view)
- [deepvoice3のメルスペクトログラムでファインチューニング](https://drive.google.com/file/d/1voxcNRVwMhaOKUAk6MhLkU5lbhdyONSP/view?usp=sharing)

download後，適当なフォルダに配置する．使用する学習済みデータはお好みの方を選択


## Requirements
[torch13.yml](torch13.yml)参照

また，コンソールで`python -c "import nltk; nltk.download('cmudict')"`を実行して音素辞書のダウンロードをする．
## Getting started
### 概要
- データの前処理：[preprocess.py](preprocess.py)
- 学習：：[train_seq2seq.py](train_seq2seq.py)
- 推論：[synthesis.py](synthesis.py)

また，ハイパーパラメータは`hparams.py`で主に設定を行う．ただし，メルスペクトログラムのパラメータは`ljspeech.py`で調整を行う．
### データの準備
このリポジトリでは，英語話者のみ学習を行ったため，他言語に関する動作は保証しない
- LJSpeech：https://keithito.com/LJ-Speech-Dataset/
- VCTK：http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
### データの前処理
使い方はr9y9様の実装と同じように使用可能
>Usage:
>```
>python preprocess.py ${dataset_name} ${dataset_path} ${out_dir} --preset=<json>
>```
>Supported `${dataset_name} s are:
>- `ljspeech`(en,single speaker)
>- `vctk`(en,multi-speaker)
>- `jsut`(jp, single speaker)
>- `nikl_m`(ko, multi-speaker)
>- `nikl_s`(ko, single speaker)
  
変更点として，メルスペクトログラム，スペクトログラム，WORLD Vocoderのパラメータ全てを出力するようにしている．

また，`hparams.py`の`key_position_rate`及び`world_upsample`は`python comute_timestamp_ratio.py <data-root>`を実行することで求める事ができる．

### 学習
使い方：
```
python train_${training_type}.py --data-root=${data-root} --log-event-path=${log_dir} --checkpoint=${checkpoint_path} --waveglow_path=${waveglow_path}
```
`--checkpoint`は学習済みのデータを再学習する場合のみ指定．

### 推論
学習済みデータを用いて，自己回帰で推論を行う．waveglowを利用する場合，
```
python synthesis.py --type='seq2seq' --waveglow_path=${waveglow_path} ${checkpoint_path} ${test_list.txt} ${output_dir}
```

）

