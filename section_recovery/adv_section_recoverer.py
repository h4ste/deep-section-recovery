"""Trains a section recovery model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from timeit import default_timer as timer
from tabulate import tabulate
import errno
import shutil
import sys
import os
import csv

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from . import record_data as data
from .adv_section_recovery_model import AdversarialSectionRecoveryModel, HParams
from .trace_logger import TraceLogger

import logging

from .language_utils import logits_to_outputs, outputs_to_words, format_outputs, \
    get_word_error_rate, get_recall_rouge, get_precision_bleu

logging.setLoggerClass(TraceLogger)
logging.basicConfig()
log = logging.getLogger("main")  # type: TraceLogger
log.setLevel(logging.DEBUG)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('record_path', '', 'Path to records.')
tf.app.flags.DEFINE_string('mode', 'train', 'whether to TRAIN or INFER.')
tf.app.flags.DEFINE_string('section', 'CORRELATION', 'Section to recover.')

tf.app.flags.DEFINE_integer('max_run_steps', 10000000, 'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Maximum vocabulary size.')
tf.app.flags.DEFINE_integer("num_dev_eval_samples", 5, "number of samples to evaluate on the dev set for each bucket")

tf.app.flags.DEFINE_bool('use_lstm', False, 'Use LSTM units.')
tf.app.flags.DEFINE_bool('use_adam', False, 'Use ADAM optimizer.')
tf.app.flags.DEFINE_bool('clear', False, 'Delete all previous checkpoints.')
tf.app.flags.DEFINE_bool('add_paragraph_symbols', False, 'Delete all previous checkpoints.')
tf.app.flags.DEFINE_bool('coupled_disc_loss', False, 'Delete all previous checkpoints.')

tf.app.flags.DEFINE_float('lr', 1e-4, 'Learning rate')

tf.app.flags.DEFINE_string('save_measures', '/dev/null', 'Where to save test-set measures.')
tf.app.flags.DEFINE_integer('random_seed', 1337, 'A seed value for randomness.')
tf.app.flags.DEFINE_integer('min_record_len', 10, 'Minimumm number of words in a record.')
tf.app.flags.DEFINE_integer('max_record_len', 300, 'Maximum number of words in a record.')
tf.app.flags.DEFINE_integer('min_section_len', 4, 'Minimum number of words in generated section.')
tf.app.flags.DEFINE_integer('max_section_len', 50, 'Maximum number of words in generated section.')
tf.app.flags.DEFINE_integer("batch_size", 20, "Number of records in each training batch.")
tf.app.flags.DEFINE_string("working_dir", "/tmp", "Working directory")
tf.app.flags.DEFINE_string("log_dir", FLAGS.working_dir + "/logs/", "Logging directory")
tf.app.flags.DEFINE_string("checkpoint_dir", FLAGS.working_dir + "/checkpoints/", "Checkpoint directory")
tf.app.flags.DEFINE_string("test_dir", FLAGS.working_dir + "/output/", "Output directory")
tf.app.flags.DEFINE_string("vocab_path", FLAGS.record_path + "/" + "vocab.dat", "Vocabulary")
tf.app.flags.DEFINE_string('save_encodings', FLAGS.record_path + ".encoded", 'Where to save test-set measures.')


def _running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.999):
    """Calculate the running average of losses."""
    if running_avg_loss == 0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)
    loss_sum = tf.Summary()
    loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    sys.stdout.write('running_avg_loss: %f\n' % running_avg_loss)
    return running_avg_loss


def _delete_dir_quiet(directory):
    try:
        shutil.rmtree(directory, ignore_errors=True)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise exc
        pass


def _make_dirs_quiet(directory):
    try:
        os.makedirs(directory)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise exc
        pass


def _simple_summary(tag, value):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    return summary


def _create_model(session, hps, vocab):
    model = AdversarialSectionRecoveryModel(hps,
                                            len(vocab),
                                            FLAGS.use_lstm,
                                            FLAGS.mode.lower() == 'train',
                                            FLAGS.coupled_disc_loss)

    logging.info("Building graph...")
    start_time = timer()
    model.build_graph()
    end_time = timer()
    log.debug("Graph constructed in %6.3f seconds!", end_time - start_time)

    if FLAGS.clear:
        log.debug("Clearing previous checkpoints...")
        _delete_dir_quiet(FLAGS.checkpoint_dir)
    _make_dirs_quiet(FLAGS.checkpoint_dir)

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path + '.index'):
        log.info("Reading model parameters from '%s'...", ckpt.model_checkpoint_path)
        start_time = timer()
        model.saver.restore(session, ckpt.model_checkpoint_path)
        end_time = timer()
        log.debug("Model restored in %6.3f seconds!", end_time - start_time)
    else:
        log.info("Creating model with fresh parameters...")
        start_time = timer()
        session.run(tf.global_variables_initializer())
        end_time = timer()
        log.debug("Model initialized in %6.3f seconds!", end_time - start_time)
    return model


# noinspection PyShadowingNames
def _evaluate_output_texts(writer, guess_batched_logits, gold_texts, step, rev_vocab, return_avg=True):
    bleus = [[], [], []]
    rouges = [[], [], []]
    wers = [[], [], []]
    log.trace('EOT | Guess batched logits: %s', guess_batched_logits)
    log.trace('EOT | Gold text outputs: %s', gold_texts)
    for j in range(FLAGS.batch_size):
        logits = guess_batched_logits[j]  # [logit[j] for logit in guess_texts]
        log.trace('EOT | Logits: %s', logits)
        outputs = logits_to_outputs(logits)
        log.trace('EOT | Outputs: %s', outputs)
        guess = outputs_to_words(format_outputs(outputs), rev_vocab)
        log.trace('EOT | Guess Words: %s', guess)
        gold = outputs_to_words(format_outputs(gold_texts[j]), rev_vocab)
        log.trace('EOT | Gold Words: %s', gold)
        bleus[0].append(get_precision_bleu(guess, gold, 1))
        bleus[1].append(get_precision_bleu(guess, gold, 2))
        bleus[2].append(get_precision_bleu(guess, gold, 3))
        rouges[0].append(get_recall_rouge(guess, gold, 1))
        rouges[1].append(get_recall_rouge(guess, gold, 2))
        rouges[2].append(get_recall_rouge(guess, gold, 3))
        wers[0].append(get_word_error_rate(guess, gold, 1))
        wers[1].append(get_word_error_rate(guess, gold, 2))
        wers[2].append(get_word_error_rate(guess, gold, 3))
    log.trace('EOT | Bleus raw: %s', bleus)
    log.trace('EOT | Rouges raw: %s', rouges)
    if return_avg:
        bleus = np.mean(bleus, axis=1)
        rouges = np.mean(rouges, axis=1)
        wers = np.mean(wers, axis=1)
        log.trace('EOT | Bleus mean: %s', bleus)
        log.trace('EOT | Rouges mean: %s', rouges)
        if writer:
            writer.add_summary(_simple_summary('BLEU-1', bleus[0].item()), step)
            writer.add_summary(_simple_summary('BLEU-2', bleus[1].item()), step)
            writer.add_summary(_simple_summary('BLEU-3', bleus[2].item()), step)
            writer.add_summary(_simple_summary('ROUGE-1', rouges[0].item()), step)
            writer.add_summary(_simple_summary('ROUGE-2', rouges[1].item()), step)
            writer.add_summary(_simple_summary('ROUGE-3', rouges[2].item()), step)
    return bleus, rouges, wers


# noinspection PyShadowingNames
def _print_evaluation_table(bleus, rouges, wer):
    log.trace('PET | Bleus raw: %s', bleus)
    log.trace('PET | Rouges raw: %s', rouges)
    bleus = np.mean(bleus, axis=1)
    rouges = np.mean(rouges, axis=1)
    wer = np.mean(wer, axis=1)
    log.trace('PET | Bleus mean: %s', bleus)
    log.trace('PET | Rouges mean: %s', rouges)
    return [["Measure", "Unigram", "Bigram", "Trigram"],
            ["BLEU", bleus[0], bleus[1], bleus[2]],
            ["ROUGE", rouges[0], rouges[1], rouges[2]],
            ["WER", wer[0], wer[1], wer[2]]]


def _logits_to_string(logits, rev_vocab):
    log.trace('L2S | Logits: %s', logits)
    outputs = logits_to_outputs(logits)
    log.trace('L2S | Outputs: %s', outputs)
    return _outputs_to_string(outputs, rev_vocab)


def _outputs_to_string(outputs, rev_vocab):
    formatted_outputs = format_outputs(outputs)
    log.trace('O2S | Formatted Outputs: %s', formatted_outputs)
    words = outputs_to_words(formatted_outputs, rev_vocab)
    log.trace('O2S | Words: %s', words)
    return ' '.join(words)


train_losses = []
train_step_times = []
train_output_logits = []
train_sections = []
train_record_ids = []
train_step = 0


def _train(train_set, devel_set, hps, vocab, rev_vocab):
    sess = tf.Session()
    model = _create_model(sess, hps, vocab)

    # Create summary
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    devel_writer = tf.summary.FileWriter(FLAGS.log_dir + '/devel', sess.graph)

    import itertools
    import threading
    condition = threading.Condition()

    def evaluate_training():
        global train_losses
        global train_step_times
        global train_output_logits
        global train_sections
        global train_record_ids
        global train_step

        train_bleus = [[], [], []]
        train_rouges = [[], [], []]
        train_wers = [[], [], []]
        record_id = []
        section = []
        output_logit = []
        condition.acquire()
        if not train_output_logits or not train_sections or not train_record_ids:
            condition.wait()
        for output_logit, section, record_id in itertools.izip(
                train_output_logits, train_sections, train_record_ids):
            bleus, rouges, wers = _evaluate_output_texts(train_writer, output_logit, section, train_step, rev_vocab)
            train_bleus = np.concatenate((train_bleus, np.expand_dims(bleus, axis=1)), axis=1)
            train_rouges = np.concatenate((train_rouges, np.expand_dims(rouges, axis=1)), axis=1)
            train_wers = np.concatenate((train_wers, np.expand_dims(wers, axis=1)), axis=1)
        print("Current Step: %d" % train_step)
        log.info("Average Step Time: %6.3fs" % np.mean(train_step_times))
        log.info("Average Loss: %6.3f" % np.mean(train_losses))
        perf_table = _print_evaluation_table(train_bleus, train_rouges, train_wers)
        print(tabulate(perf_table, headers="firstrow"))
        step_table = [["TRAIN | Guess:", _logits_to_string(output_logit[0], rev_vocab)],
                      ["TRAIN | Gold:", _outputs_to_string(section[0], rev_vocab)]]
        print("TRAIN | Record", record_id[0], ":")
        print(tabulate(step_table, tablefmt="plain"))
        sys.stdout.flush()
        train_losses = []
        train_step_times = []
        train_output_logits = []
        train_sections = []
        train_record_ids = []
        condition.notify()
        condition.release()

        # Save checkpoint
        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "srm")
        print('Saving model to', checkpoint_path, '...')
        model.saver.save(sess, checkpoint_path, global_step=model.generator_step)

        evaluate_thread = threading.Timer(60.0, evaluate_training)
        evaluate_thread.daemon = True
        evaluate_thread.start()

    def evaluate_devel():
        global train_step

        devel_bleus = [[], [], []]
        devel_rouges = [[], [], []]
        devel_wers = [[], [], []]
        condition.acquire()

        # Run evals on development set and print their perplexity.
        record_ids = []
        section_batch = []
        output_logits = []
        for j in range(FLAGS.num_dev_eval_samples):
            record_batch, record_lens_batch, section_batch, section_lens_batch, record_ids = \
                devel_set.get_batch()
            output_logits = model.run_eval_step(sess, record_batch, record_lens_batch)
            log.trace('Devel Output Logits: %s', output_logits)
            bleus, rouges, wers = _evaluate_output_texts(devel_writer, output_logits, section_batch, train_step,
                                                         rev_vocab)
            devel_bleus = np.concatenate((devel_bleus, np.expand_dims(bleus, axis=1)), axis=1)
            devel_rouges = np.concatenate((devel_rouges, np.expand_dims(rouges, axis=1)), axis=1)
            devel_wers = np.concatenate((devel_wers, np.expand_dims(wers, axis=1)), axis=1)
        condition.notify()
        condition.release()
        perf_table = _print_evaluation_table(devel_bleus, devel_rouges, devel_wers)
        print(tabulate(perf_table, headers="firstrow"))
        step_table = [["DEVEL | Guess:", _logits_to_string(output_logits[0], rev_vocab)],
                      ["DEVEL | Gold:", _outputs_to_string(section_batch[0], rev_vocab)]]
        print("DEVEL | Record", record_ids[0], ":")
        print(tabulate(step_table, tablefmt="plain"))
        sys.stdout.flush()

        evaluate_thread = threading.Timer(300.0, evaluate_devel)
        evaluate_thread.daemon = True
        evaluate_thread.start()

    t = threading.Timer(60.0, evaluate_training)
    t.daemon = True
    t.start()

    t = threading.Timer(90.0, evaluate_devel)
    t.daemon = True
    t.start()

    def train_loop():
        global train_losses
        global train_step_times
        global train_output_logits
        global train_sections
        global train_record_ids
        global train_step

        while True:
            # Get a batch and make a step.
            start_time = timer()
            record_batch, record_lens_batch, section_batch, section_lens_batch, record_ids = train_set.get_batch()
            log.trace('Train | Record Batch: %s', record_batch.shape)
            log.trace('Train | Record Length Batch: %s', record_lens_batch)
            log.trace('Train | Section Batch: %s', section_batch.shape)
            log.trace('Train | Section Length Batch: %s', section_lens_batch)

            loss, d1, d2, summaries, train_step = model.run_discriminator_train_step(sess,
                                                                                     record_batch,
                                                                                     record_lens_batch,
                                                                                     section_batch,
                                                                                     section_lens_batch)
            avg_d1 = np.mean(d1)
            avg_d2 = np.mean(d2)

            record_batch, record_lens_batch, section_batch, section_lens_batch, record_ids = train_set.get_batch()
            log.trace('Train | Record Batch: %s', record_batch.shape)
            log.trace('Train | Record Length Batch: %s', record_lens_batch)
            log.trace('Train | Section Batch: %s', section_batch.shape)
            log.trace('Train | Section Length Batch: %s', section_lens_batch)

            loss, output_logits, summaries, step = model.run_generator_train_step(sess,
                                                                                  record_batch,
                                                                                  record_lens_batch,
                                                                                  section_batch,
                                                                                  section_lens_batch)
            log.trace('Train | Loss: %s', loss)
            log.trace('Train | Output Logits: %s', output_logits)
            log.trace('Train | Output Logits Shape: %s', output_logits.shape)
            train_writer.add_summary(summaries, train_step)
            condition.acquire()
            train_losses.append(loss)
            train_output_logits.append(output_logits)
            train_sections.append(section_batch)
            train_record_ids.append(record_ids)
            end_time = timer()
            step_time = end_time - start_time
            train_step_times.append(step_time)
            condition.notify()
            condition.release()
            log.debug('Step %d | Loss: %9.7f; D1: %9.7f; D2: %9.7f; Time: %5.3f seconds', train_step, loss, avg_d1,
                      avg_d2, step_time)

    t = threading.Thread(target=train_loop, args=())
    t.daemon = True
    t.start()

    print("Training the model...")
    # This kills me inside
    import time
    while True:
        time.sleep(1)


def _test(test_set, hps, vocab, rev_vocab):
    sess = tf.Session()
    model = _create_model(sess, hps, vocab)

    test_bleus = [[], [], []]
    test_rouges = [[], [], []]
    test_wers = [[], [], []]

    # Run evals on development set and print their perplexity.
    record_ids = []
    section_batch = []
    output_logits = []
    log.info('Testing model...')
    from tqdm import tqdm
    for _ in tqdm(range(test_set.num_batches())):
        record_batch, record_lens_batch, section_batch, section_lens_batch, record_ids = \
            test_set.get_batch()
        output_logits = model.run_eval_step(sess, record_batch, record_lens_batch)
        log.trace('Test Output Logits: %s', output_logits)
        if FLAGS.save_measures == '/dev/null':
            bleus, rouges, wers = _evaluate_output_texts(None, output_logits, section_batch, train_step, rev_vocab)
            test_bleus = np.concatenate((test_bleus, np.expand_dims(bleus, axis=1)), axis=1)
            test_rouges = np.concatenate((test_rouges, np.expand_dims(rouges, axis=1)), axis=1)
            test_wers = np.concatenate((test_wers, np.expand_dims(wers, axis=1)), axis=1)
        else:
            bleus, rouges, wers = _evaluate_output_texts(None, output_logits, section_batch, train_step, rev_vocab,
                                                         return_avg=False)
            test_bleus = np.concatenate((test_bleus, bleus), axis=1)
            test_rouges = np.concatenate((test_rouges, rouges), axis=1)
            test_wers = np.concatenate((test_wers, wers), axis=1)

    # Save individual scores to file
    if FLAGS.save_measures != '/dev/null':
        log.info('Saving test sample evaluation scores to \'%s\'', FLAGS.save_measures)
        with open(FLAGS.save_measures, 'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')
            csv_writer.writerow(
                ['WER-1', 'WER-2', 'WER-3', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'ROUGE-1', 'ROUGE-2', 'ROUGE-3'])
            for i in tqdm(range(len(test_bleus[0]))):
                csv_writer.writerow([test_wers[0, i],
                                     test_wers[1, i],
                                     test_wers[2, i],
                                     test_bleus[0, i],
                                     test_bleus[1, i],
                                     test_bleus[2, i],
                                     test_rouges[0, i],
                                     test_rouges[1, i],
                                     test_rouges[2, i]])

    perf_table = _print_evaluation_table(test_bleus, test_rouges, test_wers)
    print(tabulate(perf_table, headers="firstrow"))
    step_table = []
    for j in range(30):
        step_table.append(["TEST | Guess:", _logits_to_string(output_logits[j], rev_vocab)])
        step_table.append(["TEST | Gold:", _outputs_to_string(section_batch[j], rev_vocab)])
        step_table.append(["TEST | Record", record_ids[j]])
    print(tabulate(step_table, tablefmt="plain"))
    sys.stdout.flush()


def _encode(test_set, hps, vocab):
    sess = tf.Session()
    model = _create_model(sess, hps, vocab)

    # Run evals on development set and print their perplexity.
    log.info('Encoding documents...')
    if FLAGS.save_encodings != '/dev/null':
        log.info('Saving encodings scores to \'%s\'', FLAGS.save_encodings)
        with open(FLAGS.save_encodings, 'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')
            csv_writer.writerow(
                ['PARAGRAPH_ID', 'ENCODING'])
            from tqdm import tqdm
            for _ in tqdm(range(test_set.num_batches())):
                record_batch, record_lens_batch, section_batch, section_lens_batch, record_ids = \
                    test_set.get_batch()
                batch_encodings = model.run_encode_step(sess, record_batch, record_lens_batch)
                encoding_array = np.char.mod('%f', batch_encodings)
                for record_id, encoding in zip(record_ids, encoding_array):
                    encoding_vector = " ".join(encoding)
                    csv_writer.writerow([record_id, encoding_vector])


# noinspection PyUnusedLocal
def main(unused_argv):
    batch_size = FLAGS.batch_size

    if FLAGS.mode.lower() == 'train':
        log.info("Preparing data from %s...", FLAGS.record_path)
        _train_records, _train_sections, _train_record_ids = data.read_records_tsv(
            FLAGS.record_path + '/train.tsv',
            FLAGS.section,
            min_record_len=FLAGS.min_record_len,
            min_section_len=FLAGS.min_section_len,
            add_paragraph_symbols=FLAGS.add_paragraph_symbols)
        log.trace('Training sample 1')
        log.trace('Record: | %s', _train_records[0])
        log.trace('Section:| %s', _train_sections[0])
        log.trace('ID:     | %s', _train_record_ids[0])
        train_words = [word for text in _train_records + _train_sections for word in text]
        data.create_vocabulary(FLAGS.vocab_path, train_words, FLAGS.vocab_size)
        vocab, rev_vocab = data.initialize_vocabulary(FLAGS.vocab_path)
        print("Vocabulary Size: %d == %d" % (len(vocab), len(rev_vocab)))

        hps = HParams(batch_size=batch_size,
                      enc_layers=2,  # 2,
                      dec_layers=1,
                      enc_timesteps=FLAGS.max_record_len,  # 200,
                      dec_timesteps=FLAGS.max_section_len,  # 100,
                      num_hidden=256,  # 256,
                      emb_dim=200,
                      min_lr=1e-12,
                      lr=0.001,
                      max_grad_norm=2,
                      attn_option='luong',
                      decay_steps=len(_train_records) // batch_size,
                      decay_rate=.95,
                      cnn_filters=100,
                      cnn_dropout_rate=0.25)

        train_set = data.Batcher('train', _train_records, _train_sections, _train_record_ids,
                                 vocab, batch_size, hps.enc_timesteps, hps.dec_timesteps)
        del _train_records
        del _train_sections
        del _train_record_ids
        devel_records, devel_sections, devel_record_ids = data.read_records_tsv(
            FLAGS.record_path + '/devel.tsv',
            FLAGS.section,
            min_record_len=FLAGS.min_record_len,
            min_section_len=FLAGS.min_section_len,
            add_paragraph_symbols=FLAGS.add_paragraph_symbols)
        devel_set = data.Batcher('devel', devel_records, devel_sections, devel_record_ids,
                                 vocab, batch_size, hps.enc_timesteps, hps.dec_timesteps)

        del devel_records
        del devel_sections
        del devel_record_ids

        _train(train_set, devel_set, hps=hps, vocab=vocab, rev_vocab=rev_vocab)

    elif FLAGS.mode.lower() == 'infer':
        log.info("Preparing data from %s...", FLAGS.record_path)
        test_records, test_sections, test_record_ids = data.read_records_tsv(
            FLAGS.record_path + '/test.tsv',
            FLAGS.section,
            min_record_len=FLAGS.min_record_len,
            min_section_len=FLAGS.min_section_len,
            add_paragraph_symbols=FLAGS.add_paragraph_symbols)

        log.info("Loading vocabulary from %s...", FLAGS.vocab_path)
        vocab, rev_vocab = data.initialize_vocabulary(FLAGS.vocab_path)

        hps = HParams(batch_size=batch_size,
                      enc_layers=2,  # 2,
                      dec_layers=1,
                      enc_timesteps=FLAGS.max_record_len,  # 200,
                      dec_timesteps=FLAGS.max_section_len,  # 100,
                      num_hidden=256,  # 256,
                      emb_dim=200,
                      min_lr=1e-12,
                      lr=FLAGS.learning_rate,
                      max_grad_norm=2,
                      attn_option='luong',
                      decay_steps=len(test_records) // batch_size,
                      decay_rate=.95,
                      cnn_filters=100,
                      cnn_dropout_rate=0.25)

        test_set = data.Batcher('test', test_records, test_sections, test_record_ids,
                                vocab, batch_size, hps.enc_timesteps, hps.dec_timesteps)
        del test_records
        del test_sections
        del test_record_ids

        _test(test_set, hps=hps, vocab=vocab, rev_vocab=rev_vocab)

    elif FLAGS.mode.lower() == 'encode':
        log.info("Preparing data from %s...", FLAGS.record_path)
        test_records, test_sections, test_record_ids = data.read_records_tsv(
            FLAGS.record_path,
            FLAGS.section,
            min_record_len=FLAGS.min_record_len,
            min_section_len=FLAGS.min_section_len,
            add_paragraph_symbols=FLAGS.add_paragraph_symbols)

        log.info("Loading vocabulary from %s...", FLAGS.vocab_path)
        vocab, rev_vocab = data.initialize_vocabulary(FLAGS.vocab_path)

        hps = HParams(batch_size=batch_size,
                      enc_layers=2,  # 2,
                      dec_layers=1,
                      enc_timesteps=FLAGS.max_record_len,  # 200,
                      dec_timesteps=FLAGS.max_section_len,  # 100,
                      num_hidden=256,  # 256,
                      emb_dim=200,
                      min_lr=1e-12,
                      lr=1e-4,
                      max_grad_norm=2,
                      attn_option='luong',
                      decay_steps=len(test_records) // batch_size,
                      decay_rate=.95,
                      cnn_filters=100,
                      cnn_dropout_rate=0.25)

        test_set = data.Batcher('input', test_records, test_sections, test_record_ids,
                                vocab, batch_size, hps.enc_timesteps, hps.dec_timesteps)

        del test_records
        del test_sections
        del test_record_ids

        _encode(test_set, hps=hps, vocab=vocab)


if __name__ == "__main__":
    tf.app.run()
