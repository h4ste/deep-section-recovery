"""Sequence-to-Sequence with attention model for text summarization.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import logging

import numpy as np
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf


from . import record_data

from .trace_logger import TraceLogger

logging.setLoggerClass(TraceLogger)
logging.basicConfig()
log = logging.getLogger("model")
log.setLevel(logging.DEBUG)

HParams = namedtuple('HParams',
                     'batch_size,'
                     'enc_layers, dec_layers, enc_timesteps, dec_timesteps,'
                     'num_hidden, emb_dim,'
                     'min_lr, lr, max_grad_norm,'
                     'attn_option,'
                     'decay_steps, decay_rate ')


class SectionRecoveryModel(object):
    """Wrapper for Tensorflow model graph for section recovery parameters."""

    def __init__(self,
                 hps,
                 vocab_size,
                 use_lstm=False,
                 use_adam=False,
                 is_training=True,
                 disable_extractor=False,
                 disable_word_vectors=False,
                 disable_report_vector=False,
                 disable_attention=False):
        self._hps = hps
        assert hps.enc_layers >= 1, "Must have at least 1 encoding layer"
        assert hps.dec_layers >= 1, "Must have at least 1 decoding layer"
        assert hps.enc_timesteps >= 1, "Records must have at least 1 word"
        assert hps.dec_timesteps >= 1, "Sections must have at least 1 word"
        assert hps.emb_dim >= 1, "Embedding size must be at least 1 dimension"

        self._vocab_size = vocab_size
        log.debug("Setting vocabulary size to %d", vocab_size)
        self._use_lstm = use_lstm
        if use_lstm: log.debug("Using LSTM units to build encoder/decoder RNNs")
        else: log.debug("Using GRUs to build encoder/decoder RNNs")
        self._use_adam = use_adam
        if use_adam: log.debug("Using ADAM optimization to train the model")
        else: log.debug("Using (Stochastic) Gradient Descent to train the model")
        self._is_training = is_training
        if is_training: log.debug("Adding training configuration to RNLM decoder")

        self.disable_extractor = disable_extractor
        self.disable_attention = disable_attention
        self.disable_word_vectors = disable_word_vectors
        self.disable_report_vector = disable_report_vector

    def _add_placeholders(self):
        log.debug('Adding placeholders to computation graph...')
        """Inputs to be fed to the graph."""
        hps = self._hps
        self._records = tf.placeholder(tf.int32,
                                       [hps.batch_size, hps.enc_timesteps],
                                       name='records')
        self._record_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                           name='record_lens')
        self._sections = tf.placeholder(tf.int32,
                                        [hps.batch_size, hps.dec_timesteps - 1],
                                        name='sections')
        self._section_lens = tf.placeholder(tf.int32,
                                        [hps.batch_size],
                                        name='section_lens')
        self._targets = tf.placeholder(tf.int32,
                                       [hps.batch_size, hps.dec_timesteps - 1],
                                       name='targets')
        self._loss_weights = tf.placeholder(tf.float32,
                                            [hps.batch_size, hps.dec_timesteps - 1],
                                            name='loss_weights')

    def _add_embedding_ops(self):
        log.debug('Adding embedding layers to computation graph...')
        hps = self._hps
        vsize = self._vocab_size
        # Embedding shared by the input and outputs.
        with tf.variable_scope('embedding'), tf.device('/cpu:0'):
            if not self.disable_extractor:
                self._encoder_embedding = enc_embedding = tf.get_variable(
                    'encoder_embedding', [vsize, hps.emb_dim], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=1e-4))
            else:
                self._encoder_embedding = enc_embedding = tf.get_variable(
                    'encoder_embedding', [vsize, hps.num_hidden], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=1e-4))
            self._decoder_embedding = dec_embedding = tf.get_variable(
                'decoder_embedding', [vsize, hps.emb_dim], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=1e-4))
            self._emb_encoder_inputs = tf.nn.embedding_lookup(enc_embedding, self._records)
            self._emb_decoder_inputs = tf.nn.embedding_lookup(dec_embedding, self._sections)

    def _rnn_cell(self, num_hidden):
        if self._use_lstm:
            return tf.contrib.rnn.LSTMCell(num_hidden,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
        else:
            return tf.contrib.rnn.GRUCell(num_hidden)

    def _add_encoder(self):
        log.debug('Adding encoder component to computation graph...')
        hps = self._hps
        with tf.variable_scope("encoder"):
            if not self.disable_extractor:
                for layer_i in xrange(hps.enc_layers):
                    with tf.variable_scope('layer_%d' % layer_i):
                        cell_fw = self._rnn_cell(hps.num_hidden)
                        cell_bw = self._rnn_cell(hps.num_hidden)
                        (fw_outputs, _), (fw_state, _) = tf.nn.bidirectional_dynamic_rnn(
                            cell_fw, cell_bw, self._emb_encoder_inputs, dtype=tf.float32,
                            sequence_length=self._record_lens,
                            swap_memory=True)
                if not self.disable_word_vectors:
                    # noinspection PyUnboundLocalVariable
                    log.trace('RNN Outputs: %s', fw_outputs)
                    self._encoder_outputs = fw_outputs  # tf.concat(outputs, axis=2)
                else:
                    self._encoder_outputs = self._emb_encoder_inputs
                log.trace('Encoder outputs: %s', self._encoder_outputs)

                if not self.disable_report_vector:
                    # noinspection PyUnboundLocalVariable
                    log.trace('RNN States: %s', fw_state)
                    self._encoder_state = fw_state
                else:
                    self._encoder_state = tf.zeros_like(self._encoder_outputs, dtype=tf.float32)
                log.trace('Encoder state: %s', self._encoder_state)
            else:
                self._encoder_outputs = self._emb_encoder_inputs
                print(self._encoder_outputs)
                self._encoder_state = tf.zeros([hps.batch_size, hps.num_hidden], dtype=tf.float32)

    def _add_decoder(self):
        log.debug('Adding decoder component to computation graph...')
        hps = self._hps
        vsize = self._vocab_size

        with tf.variable_scope('decoder') as scope:
            if not self.disable_attention:
                attn_states = self._encoder_outputs  # tf.concat(self._encoder_outputs, axis=1)
                # Attention states: size [batch_size, hps.max_decoder_timesteps, num_units]
                log.trace('Attention States: %s', attn_states)
                attn_keys, attn_values, attn_score_fn, attn_const_fn = tf.contrib.seq2seq.prepare_attention(
                    attn_states, attention_option=hps.attn_option, num_units=hps.num_hidden)
                log.trace("Attention Keys: %s", attn_keys)
                log.trace('Attention Values: %s', attn_values)
                log.trace('Attention Score Function: %s', attn_score_fn)
                log.trace('Attention Construction Function: %s', attn_const_fn)


            # Define the type of RNN cells the RNLM will use
            if hps.dec_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([self._rnn_cell(hps.num_hidden) for _ in xrange(hps.dec_layers)])
            else:
                cell = self._rnn_cell(hps.num_hidden)
            log.trace('Decoder RNN Cell: %s', cell)

            # Setup weights for computing the final output
            def create_output_fn():

                def output_fn(x):
                    return tf.contrib.layers.linear(x, vsize, scope=scope)

                return output_fn

            output_fn = create_output_fn()


            # We don't need to add the training decoder unless we're training model
            if self._is_training:
                # Setup decoder in (1) training mode (i.e., consider gold previous words) and (2) with attention
                if not self.disable_attention:
                    decoder_fn_train = tf.contrib.seq2seq.attention_decoder_fn_train(
                        encoder_state=self._encoder_state,
                        attention_keys=attn_keys,
                        attention_values=attn_values,
                        attention_score_fn=attn_score_fn,
                        attention_construct_fn=attn_const_fn)
                else:
                    decoder_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train(
                        encoder_state=self._encoder_state)


                # Setup RNLM for training
                decoder_outputs_train, decoder_state_train, _ = \
                    tf.contrib.seq2seq.dynamic_rnn_decoder(cell=cell,
                                                           decoder_fn=decoder_fn_train,
                                                           inputs=self._emb_decoder_inputs,
                                                           sequence_length=hps.dec_timesteps - 1,
                                                           swap_memory=True,
                                                           scope=scope)

                # Project RNLM outputs into vocabulary space
                self._decoder_outputs_train = outputs = output_fn(decoder_outputs_train)
                log.trace('(Projected) RNLM Decoder Training Outputs: %s', outputs)

                # Compute sequence loss
                self._loss = tf.contrib.seq2seq.sequence_loss(outputs, self._targets, self._loss_weights)
                log.trace('RNLM Decoder Loss: %s', self._loss)
                tf.summary.scalar('loss', tf.minimum(12.0, self._loss))

                # If we have created a training decoder, tell the inference decoder to re-use the same weights
                scope.reuse_variables()

            # Inference decoder: use previously generated output to predict next output (e.g., no gold outputs)
            if not self.disable_attention:
                decoder_fn_inference = tf.contrib.seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self._encoder_state,
                    attention_keys=attn_keys,
                    attention_values=attn_values,
                    attention_score_fn=attn_score_fn,
                    attention_construct_fn=attn_const_fn,
                    embeddings=self._decoder_embedding,
                    start_of_sequence_id=record_data.PARAGRAPH_START_ID,
                    end_of_sequence_id=record_data.PARAGRAPH_END_ID,
                    maximum_length=hps.dec_timesteps - 1,
                    num_decoder_symbols=self._vocab_size,
                    dtype=tf.int32)
            else:
                decoder_fn_inference = tf.contrib.seq2seq.simple_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self._encoder_state,
                    embeddings=self._decoder_embedding,
                    start_of_sequence_id=record_data.PARAGRAPH_START_ID,
                    end_of_sequence_id=record_data.PARAGRAPH_END_ID,
                    maximum_length=hps.dec_timesteps - 1,
                    num_decoder_symbols=self._vocab_size,
                    dtype=tf.int32)
            decoder_outputs_inference, decoder_state_inference, _ = \
                tf.contrib.seq2seq.dynamic_rnn_decoder(cell=cell,
                                                       decoder_fn=decoder_fn_inference,
                                                       swap_memory=True,
                                                       scope=scope)
            log.trace('RNLM Decoder Inference Outputs: %s', decoder_outputs_inference)
            self._decoder_outputs_inference = decoder_outputs_inference


    def _add_seq2seq(self):
        with tf.variable_scope("seq2seq"):
            self._add_embedding_ops()
            self._add_encoder()
            self._add_decoder()

    @staticmethod
    def _add_max_tconv(inputs, filters, window, max, activation=None):
        conv = tf.layers.conv1d(inputs, filters, window, max, activation=activation)
        pool = tf.layers.max_pooling1d(conv, max, max)
        return pool

    def _add_discriminator(self, tokens):
        w5 = self._add_max_tconv(tokens, 100, 5, 5)
        w4 = self._add_max_tconv(tokens, 100, 4, 4)
        w3 = self._add_max_tconv(tokens, 100, 3, 3)

        w5 + w4 + w3







    def _add_train_op(self):
        log.debug('Adding training optimization steps to computation graph...')
        """Sets self._train_op, op to run for training."""
        hps = self._hps
        tvars = tf.trainable_variables()
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(self._loss, tvars), hps.max_grad_norm)
        tf.summary.scalar('global_norm', global_norm)
        self.global_step = tf.Variable(0, trainable=False)
        self._lr = tf.maximum(
            hps.min_lr,  # min_lr_rate.
            tf.train.exponential_decay(hps.lr, self.global_step, hps.decay_steps, hps.decay_rate))
        if self._use_adam:
            optimizer = tf.train.AdamOptimizer(self._lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
        tf.summary.scalar('learning_rate', self._lr)
        self._opt_step = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='opt_step')

    def _add_summaries(self):
        self._summaries = tf.summary.merge_all()

    def build_graph(self):
        self._add_placeholders()
        self._add_seq2seq()
        if self._is_training: self._add_train_op()
        self._add_summaries()
        self.saver = tf.train.Saver(tf.global_variables())

    @staticmethod
    def _preprare_sections(sections, section_lens):
        # Input to the RNLM is every word except the last (e.g., everything up to </D>)
        rnlm_inputs = sections[:,:-1]
        log.trace("RNLM Input: %s; Shape: %s", rnlm_inputs, rnlm_inputs.shape)
        # Target output from the RNLM is every word except the first (e.g., everything after <D>)
        rnlm_targets = sections[:,1:]
        log.trace("RNLM Targets: %s; Shape: %s", rnlm_targets, rnlm_targets.shape)
        # Loss weight mask so we don't worry about junk generated after the sequence ends
        target_mask = np.not_equal(rnlm_targets, record_data.PAD_ID)
        rnlm_loss_weights = target_mask.astype(int)
        log.trace("RNLM Loss Mask: %s; Shape: %s", rnlm_loss_weights, rnlm_loss_weights.shape)
        rnlm_lens = section_lens - 1
        log.trace("RNLM Lens: %s", rnlm_lens)
        return rnlm_inputs, rnlm_targets, rnlm_loss_weights, rnlm_lens

    def run_train_step(self, session, records, record_lens, sections, section_lens):
        rnlm_input, rnlm_targets, rnlm_loss_weights, rnlm_lens = self._preprare_sections(sections, section_lens)
        _, loss, output, summaries, step = session.run(
            [self._opt_step, self._loss, self._decoder_outputs_train, self._summaries, self.global_step],
            feed_dict={self._records: records,
                       self._record_lens: record_lens,
                       self._sections: rnlm_input,
                       self._section_lens: rnlm_lens,
                       self._targets: rnlm_targets,
                       self._loss_weights: rnlm_loss_weights})
        return loss, output, summaries, step

    def run_eval_step(self, session, records, record_lens):
        hps = self._hps
        output = session.run(
            [self._decoder_outputs_inference],
            feed_dict={self._records: records,
                       self._record_lens: record_lens,
                       # When evaluating, we don't need gold-standard sections, but tensorflow requires something
                       # to be passed for the placeholders, so, for now, we fill them with zeros.
                       # TODO: find a less insane way to handle this
                       self._sections: np.zeros([hps.batch_size, hps.dec_timesteps - 1], np.int32),
                       self._section_lens: np.zeros([hps.batch_size], np.int32),
                       self._targets: np.zeros([hps.batch_size, hps.dec_timesteps - 1], np.int32),
                       self._loss_weights: np.zeros([hps.batch_size, hps.dec_timesteps - 1], np.float32)})
        # Output is a list containing one tensor: the outputs of the model
        return output[0]

    def run_encode_step(self, session, records, record_lens):
        hps = self._hps
        output = session.run(
            self._encoder_state,
            feed_dict={self._records: records,
                       self._record_lens: record_lens,
                       # When evaluating, we don't need gold-standard sections, but tensorflow requires something
                       # to be passed for the placeholders, so, for now, we fill them with zeros.
                       # TODO: find a less insane way to handle this
                       self._sections: np.zeros([hps.batch_size, hps.dec_timesteps - 1], np.int32),
                       self._section_lens: np.zeros([hps.batch_size], np.int32),
                       self._targets: np.zeros([hps.batch_size, hps.dec_timesteps - 1], np.int32),
                       self._loss_weights: np.zeros([hps.batch_size, hps.dec_timesteps - 1], np.float32)})
        return output
