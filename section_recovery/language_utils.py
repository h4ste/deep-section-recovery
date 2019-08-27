from __future__ import division

from collections import Counter

import numpy as np
from six import iterkeys

from editdistance import eval as lev

import record_data as data


def format_outputs(outputs, end_id=data.PARAGRAPH_END_ID):
    outputs = np.asarray(outputs)
    if end_id in outputs:
        end_index = np.where(outputs == end_id)
        outputs = outputs[:end_index[0][0] + 1]
    return outputs


def logits_to_outputs(output_logits):
    outputs = [int(np.argmax(logits)) for logits in output_logits]
    return format_outputs(outputs)


def outputs_to_words(outputs, rev_vocab):
    words = []
    for output in outputs:
        try:
            words.append(rev_vocab[output])
        except IndexError:
            print('Failed to produce word for ouput', output, ' with vocabulary size ', len(rev_vocab))
    return words


def words_to_ngrams(words, n=2):
    ngrams = ["_".join(t) for t in zip(*(words[i:] for i in range(n)))]
    return ngrams


def get_word_error_rate(guess, gold, n=1):
    guess = words_to_ngrams(guess, n)
    gold = words_to_ngrams(gold, n)
    return lev(guess, gold) / len(gold)


def get_precision_bleu(guess, gold, n=2):
    guess = words_to_ngrams(guess, n)
    gold = words_to_ngrams(gold, n)
    f_guess = Counter(guess)
    f_gold = Counter(gold)
    bleu_sum = 0.0
    for ngram in iterkeys(f_guess):
        bleu_sum += min(f_guess[ngram], f_gold[ngram])
    if len(f_guess.keys()) == 0:
        return 0
    else:
        return np.float32(bleu_sum) / np.float32(len(guess))


def get_recall_rouge(guess, gold, n=2):
    guess = words_to_ngrams(guess, n)
    gold = words_to_ngrams(gold, n)
    f_guess = Counter(guess)
    f_gold = Counter(gold)
    rouge_sum = 0.0
    for ngram in iterkeys(f_gold):
        rouge_sum += min(f_guess[ngram], f_gold[ngram])
    if len(f_gold.keys()) == 0:
        return 0
    else:
        return np.float32(rouge_sum) / np.float32(len(gold))
