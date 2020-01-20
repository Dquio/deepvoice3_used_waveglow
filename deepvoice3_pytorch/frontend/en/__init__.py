# coding: utf-8
from deepvoice3_pytorch.frontend.text.symbols import symbols
from deepvoice3_pytorch.frontend.text.numbers import normalize_numbers

import nltk
from random import random

n_vocab = len(symbols)

_arpabet = nltk.corpus.cmudict.dict()


def _maybe_get_arpabet(word, p):
    if word[-1] in '!,.:;?':
        word = word[:-1]
        punc = ' %'
    else:
        punc = None
    try:
        phonemes = _arpabet[word.lower()][0]
        phonemes = " ".join(phonemes)
        phonemes = '{%s}' %phonemes
        phonemes = phonemes + punc if punc is not None else phonemes
    except KeyError:
        return word + punc if punc is not None else word

    word = word + punc if punc is not None else word
    return phonemes if random() < p else word


def mix_pronunciation(text, p):
    #text = '%'.join(word for word in text.split(', '))
    text = ' '.join(_maybe_get_arpabet(word, p) for word in text.split(' '))
    return text


def text_to_sequence(text, p=0.0):
    text = normalize_numbers(text)
    if p >= 0:
        text = mix_pronunciation(text, p)
    from deepvoice3_pytorch.frontend.text import text_to_sequence
    text = text_to_sequence(text, ["english_cleaners"])
    return text


from deepvoice3_pytorch.frontend.text import sequence_to_text
