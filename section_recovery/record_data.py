# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Data batchers for data described in ..//data_prep/README.md."""
from __future__ import print_function
from __future__ import absolute_import

import os
import csv
import re
import sys
import logging

from .trace_logger import TraceLogger

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin

logging.setLoggerClass(TraceLogger)
logging.basicConfig()
log = logging.getLogger("data")  # type: TraceLogger

LEVEL = 'DEBUG'
log.setLevel(LEVEL)

csv.field_size_limit(sys.maxsize)

# Special tokens
PARAGRAPH_START = '<p>'
PARAGRAPH_END = '</p>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
DOCUMENT_START = '<d>'
DOCUMENT_END = '</d>'

PAD_ID = 5
DOCUMENT_START_ID = 6
DOCUMENT_END_ID = 7
PARAGRAPH_START_ID = 0
PARAGRAPH_END_ID = 1
UNKNOWN_TOKEN_ID = 4

_START_VOCAB = [PARAGRAPH_START, PARAGRAPH_END, SENTENCE_START, SENTENCE_END,
                UNKNOWN_TOKEN, PAD_TOKEN, DOCUMENT_START, DOCUMENT_END]


def _pad(ids, length, pad_id=PAD_TOKEN):
    """Pad or trim list to len length.

    Args:
    ids: list of ints to pad
    pad_id: what to pad with
    length: length to pad or trim to

    Returns:
    ids trimmed or padded with pad_id
    """
    assert pad_id is not None
    assert length is not None

    if len(ids) < length:
        a = [pad_id] * (length - len(ids))
        return ids + a
    if len(ids) > length:
        return ids[:length - 1] + ids[-1:]
    else:
        return ids


def sublist(small, big):
    for i in xrange(len(big) - len(small) + 1):
        for j in xrange(len(small)):
            if big[i + j] != small[j]:
                break
        else:
            return i, i + len(small)
    return False


def read_records_tsv(path, section_name,
                     min_record_len, min_section_len,
                     add_paragraph_symbols=False):
    record_ids = []
    records = []
    sections = []
    log.debug("Reading %s", path)
    with open(path) as itsv:
        tsvin = csv.reader(itsv, delimiter='\t', quoting=csv.QUOTE_NONE, lineterminator='\n')
        header = next(tsvin)
        log.trace('Header: %s', header)
        if section_name.upper() not in header:
            sys.exit('Invalid section, valid options are %s' % ','.join(header[1:]))
        secid = header.index(section_name.upper())
        for row in tsvin:
            record_id = row[0]
            record = [DOCUMENT_START]
            record_len = 0
            for i, x in enumerate(row[1:-1]):
                if i + 1 != secid:
                    section_words = x.split()
                    record_len += len(section_words)
                    if add_paragraph_symbols:
                        record.append(PARAGRAPH_START)
                        record.extend(section_words)
                        record.append(PARAGRAPH_END)
                    else:
                        record.extend(section_words)
            record.append(DOCUMENT_END)
            try:
                section_words = row[secid].split()
                section_len = len(section_words)
                if add_paragraph_symbols:
                    section = [PARAGRAPH_START]
                    section.extend(section_words)
                    section.append(PARAGRAPH_END)
                else:
                    section = section_words
            except IndexError:
                print("Failed to parse line:", row)
            if record_len >= min_record_len and section_len >= min_section_len:
                record_ids.append(record_id)
                records.append(record)
                sections.append(section)
            else:
                log.trace('Skipping %s', row)

    assert len(records) == len(sections) == len(record_ids)
    log.debug("Read %d records", len(records))
    return records, sections, record_ids

from tqdm import tqdm
class Batcher(object):
    def __init__(self, name, records, sections, record_ids, vocab, batch_size, max_record_len, max_section_len):
        self._log = logging.getLogger("data:" + name)  # type: TraceLogger
        self._log.setLevel(LEVEL)
        self.name = name
        batched_record_lens = []
        batched_records = []
        batched_section_lens = []
        batched_sections = []
        batched_record_ids = []
        for i in tqdm(xrange(0, len(records) - batch_size, batch_size)):
            batch_record_lens = []
            batch_records = []
            batch_section_lens = []
            batch_sections = []
            batch_record_ids = []
            for j in xrange(batch_size):
                # if len(records[i + j]) > min_record_len and len(sections[i + j]) > min_section_len:
                    record = seq_to_token_ids(_pad(records[i + j], max_record_len), vocab)
                    section = seq_to_token_ids(_pad(sections[i + j], max_section_len), vocab)
                    batch_record_lens.append(min(len(records[i + j]), len(record)))
                    batch_records.append(record)
                    batch_section_lens.append(min(len(sections[i + j]), len(section)))
                    batch_sections.append(section)
                    batch_record_ids.append(record_ids[i + j])
                # else:
                #     log.debug('Skipping record %s', record_ids[i + j])
                #     j -= 1
                #     i += 1
            assert len(batch_records) == batch_size, "expected " + str(batch_size) + " records but found " + \
                                                     str(len(batch_records))
            batched_record_lens.append(batch_record_lens)
            batched_records.append(batch_records)
            batched_section_lens.append(batch_section_lens)
            batched_sections.append(batch_sections)
            batched_record_ids.append(batch_record_ids)

        self._records = np.asarray(batched_records, dtype=np.int32)
        self._record_lens = np.asarray(batched_record_lens, dtype=np.int32)
        self._sections = np.asarray(batched_sections, dtype=np.int32)
        self._section_lens = np.asarray(batched_section_lens, dtype=np.int32)
        self._record_ids = np.asarray(batched_record_ids)
        self._num_batches = len(self._records)
        self._log.debug('Generated %d batches', self._num_batches)
        self._batch_queue = np.random.permutation(range(0, self._num_batches))
        self._queue_position = 0

    def num_batches(self):
        return self._records.shape[0]

    def get_batch(self):
        if self._queue_position == self._num_batches:
            self._batch_queue = np.random.permutation(self._batch_queue)
            self._queue_position = 0
        self._log.trace('Batch queue: %s; position: %d', self._batch_queue, self._queue_position)
        i = self._batch_queue[self._queue_position]
        self._log.trace('Batch index: %d', i)
        self._queue_position += 1

        # log.trace('- Records: %s', self._records[i])
        # self._log.trace('Record Lens: %s', self.name, self._record_lens[i])
        # log.trace('- Sections: %s', self._sections[i])
        # self._log.trace('Section Lens: %s', self.name, self._section_lens[i])
        self._log.trace('Batch queue: %s; position: %d, i: %d', self._batch_queue, self._queue_position, i)

        return self._records[i], self._record_lens[i], \
            self._sections[i], self._section_lens[i], self._record_ids[i]


_DIGIT_RE = re.compile(r"\d")


def create_vocabulary(vocabulary_path, words, max_vocabulary_size, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from data file.
    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.
    Args:
      :param vocabulary_path: path where the vocabulary will be created.
      :param words: words that will be used to create vocabulary.
      :param max_vocabulary_size: limit on the size of the created vocabulary.
      :param normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not os.path.isfile(vocabulary_path):
        print("Creating vocabulary %s with max size %d" % (vocabulary_path, max_vocabulary_size))
        vocab = {}
        counter = 0
        for w in words:
            if w in _START_VOCAB:
                continue
            counter += 1
            if counter % 10000 == 0:
                print("  processing word %d = %s" % (counter, w))
            word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with open(vocabulary_path, mode="w") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.
    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].
    Args:
      vocabulary_path: path to the file containing the vocabulary.
    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).
    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if os.path.isfile(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def seq_to_token_ids(words, vocabulary, normalize_digits=True):
    if not normalize_digits:
        return [vocabulary.get(w, UNKNOWN_TOKEN_ID) for w in words]
        # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNKNOWN_TOKEN_ID) for w in words]
