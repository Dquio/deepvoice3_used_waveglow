# coding: utf-8
from deepvoice3_pytorch.frontend.text.symbols import symbols
from deepvoice3_pytorch.frontend.text.numbers import normalize_numbers

import nltk
from random import random

n_vocab = len(symbols)

_arpabet = nltk.corpus.cmudict.dict()


# Todo: 具体的な処理を理解する
# Acquisition of phonemes using nltk
def _maybe_get_arpabet(word, p):
    #if word contains punctuation, it cannot change phonemes.
    #e.g. word='Printing,' cannot change, but word='Printing' can change phonemes
    if len(word) != 0 and word[-1] in '!,.:;?':
        punc = ' %' + word[-1] if word[-1] in '!.?' else ' %'
        word = word[:-1]
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


# Convert part of text to phonemes
def mix_pronunciation(text, p):
    #text = '%'.join(word for word in text.split(', '))
    text = ' '.join(_maybe_get_arpabet(word, p) for word in text.split(' '))
    return text


# Main function
# text processing
def text_to_sequence(text, p=0.0):
    # Replace specific characters
    text = normalize_numbers(text)
    text = text.replace('\r', '')
    # If the last character is not '!,.:;?', add '.'
    text = text + '.' if text[-1] not in '!,.:;?' else text

    # Convert part of text to phonemes
    if p >= 0:
        text = mix_pronunciation(text, p)

    # text to IDs
    from deepvoice3_pytorch.frontend.text import text_to_sequence
    text = text_to_sequence(text, ["english_cleaners"])
    return text


from deepvoice3_pytorch.frontend.text import sequence_to_text

#test
if __name__ == '__main__':
    print('input ratio:')
    p = float(input())
    print('input English sentence:')
    text = input()
    seq = text_to_sequence(text, p)
    print('sequence:{}'.format(seq))
    seq2text = sequence_to_text(seq)
    print('sequence to text:{}'.format(seq2text))