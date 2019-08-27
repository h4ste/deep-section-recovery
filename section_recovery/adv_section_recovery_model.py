"""Sequence-to-Sequence with attention model for text summarization.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow import contrib, train
from tensorflow.contrib import seq2seq, layers

from . import record_data

from .trace_logger import TraceLogger, TRACE_LEVEL_NUM

logging.setLoggerClass(TraceLogger)
logging.basicConfig()
log = logging.getLogger("model")  # type: TraceLogger
log.setLevel(logging.DEBUG)

HParams = namedtuple('HParams',
                     'batch_size,'
                     'enc_layers, dec_layers, enc_timesteps, dec_timesteps,'
                     'num_hidden, emb_dim,'
                     'min_lr, lr, max_grad_norm,'
                     'attn_option,'
                     'decay_steps, decay_rate,'
                     'cnn_filters, cnn_dropout_rate')


class AdversarialSectionRecoveryModel(object):
    """Wrapper for Tensorflow model graph for section recovery parameters."""

    def __init__(self,
                 hps,
                 vocab_size,
                 use_lstm=False,
                 is_training=True,
                 coupled_discriminator_loss=False):
        """
        Initialize an AdversarialSectionRecoveryModel
        :param hps: model hyper-parameters
        :param vocab_size: size of the vocabulary (i.e., number of unique words)
        :param use_lstm: whether to use an LSTM or GRU in the decoder
        :param is_training: whether the model is training or being used for inference (i.e., testing)
        :param coupled_discriminator_loss: whether to couple the discriminator loss,
               taken from https://arxiv.org/pdf/1610.09038.pdf
        """
        self._hps = hps
        assert hps.enc_layers >= 1, "Must have at least 1 encoding layer"
        assert hps.dec_layers >= 1, "Must have at least 1 decoding layer"
        assert hps.enc_timesteps >= 1, "Records must have at least 1 word"
        assert hps.dec_timesteps >= 1, "Sections must have at least 1 word"
        assert hps.emb_dim >= 1, "Embedding size must be at least 1 dimension"

        self._vocab_size = vocab_size
        log.debug("Setting vocabulary size to %d", vocab_size)

        self._use_lstm = use_lstm
        if use_lstm:
            log.debug("Using LSTM units to build encoder/decoder RNNs")
        else:
            log.debug("Using GRUs to build encoder/decoder RNNs")

        self._is_training = is_training
        if is_training:
            log.debug("Adding training configuration to RNLM decoder")

        self.coupled_discriminator_loss = coupled_discriminator_loss
        if coupled_discriminator_loss:
            log.debug("Using coupled teacher-forced and free-running behavior as discriminator loss")
        else:
            log.debug("Using standard teacher-forced and free-running behavior as discriminator loss")

        self.saver = None

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
            self._embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32,
                                              initializer=tf.truncated_normal_initializer(stddev=1e-4))

    def _rnn_cell(self, num_hidden):
        if self._use_lstm:
            return contrib.rnn.LSTMCell(num_hidden, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
        else:
            return contrib.rnn.GRUCell(num_hidden)

    def _add_encoder(self):
        """
        Add encoder (i.e., "reader" or "extractor") to the computation graph/model
        :return: None
        """
        log.debug('Adding encoder component to computation graph...')
        hps = self._hps
        with tf.variable_scope("encoder"):
            self._emb_encoder_inputs = tf.nn.embedding_lookup(self._embedding, self._records)

            fw_state = None
            for layer_i in xrange(hps.enc_layers):
                with tf.variable_scope('layer_%d' % layer_i):
                    cell_fw = self._rnn_cell(hps.num_hidden)
                    cell_bw = self._rnn_cell(hps.num_hidden)
                    (fw_outputs, _), (fw_state, _) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, self._emb_encoder_inputs, dtype=tf.float32,
                        sequence_length=self._record_lens,
                        swap_memory=True)

                # noinspection PyUnboundLocalVariable
                log.trace('RNN Outputs: %s', fw_outputs)
                self._encoder_outputs = fw_outputs  # tf.concat(outputs, axis=2)
            log.trace('Encoder outputs: %s', self._encoder_outputs)

            assert fw_state is not None
            log.trace('RNN States: %s', fw_state)
            self._encoder_state = fw_state

    def _add_decoder(self):
        """
        Adds decoder (i.e., "writer" or "generator") to the computation graph/model
        :return: None
        """
        log.debug('Adding decoder component to computation graph...')
        hps = self._hps
        vsize = self._vocab_size

        with tf.variable_scope('decoder') as scope:
            attn_states = self._encoder_outputs
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
                cell = contrib.rnn.MultiRNNCell([self._rnn_cell(hps.num_hidden) for _ in xrange(hps.dec_layers)])
            else:
                cell = self._rnn_cell(hps.num_hidden)
            log.trace('Decoder RNN Cell: %s', cell)

            # Setup weights for computing the final output
            def create_output_fn():
                def _output_fn(x):
                    return tf.contrib.layers.linear(x, vsize, scope=scope)

                return _output_fn

            output_fn = create_output_fn()

            # We don't need to add the training decoder unless we're training model
            if self._is_training:
                # Setup decoder in (1) training mode (i.e., consider gold previous words) and (2) with attention
                decoder_fn_train = seq2seq.attention_decoder_fn_train(
                    encoder_state=self._encoder_state,
                    attention_keys=attn_keys,
                    attention_values=attn_values,
                    attention_score_fn=attn_score_fn,
                    attention_construct_fn=attn_const_fn)

                # Setup RNLM for training
                self._emb_decoder_inputs = tf.nn.embedding_lookup(self._embedding, self._sections)
                decoder_outputs_train, decoder_state_train, _ = \
                    contrib.seq2seq.dynamic_rnn_decoder(cell=cell,
                                                        decoder_fn=decoder_fn_train,
                                                        inputs=self._emb_decoder_inputs,
                                                        sequence_length=hps.dec_timesteps - 1,
                                                        swap_memory=True,
                                                        scope=scope)

                self._raw_decoder_outputs = decoder_outputs_train


                # Project RNLM outputs into vocabulary space
                self._decoder_outputs_train = outputs = output_fn(decoder_outputs_train)
                log.trace('(Projected) RNLM Decoder Training Outputs: %s', outputs)

                # Compute sequence loss
                self._decoder_loss = seq2seq.sequence_loss(outputs, self._targets, self._loss_weights)
                log.trace('RNLM Decoder Loss: %s', self._decoder_loss)
                tf.summary.scalar('decoder loss', tf.minimum(12.0, self._decoder_loss))

                # If we have created a training decoder, tell the inference decoder to re-use the same weights
                scope.reuse_variables()

            # Inference decoder: use previously generated output to predict next output (e.g., no gold outputs)
            decoder_fn_inference = tf.contrib.seq2seq.attention_decoder_fn_inference(
                output_fn=output_fn,
                encoder_state=self._encoder_state,
                attention_keys=attn_keys,
                attention_values=attn_values,
                attention_score_fn=attn_score_fn,
                attention_construct_fn=attn_const_fn,
                embeddings=self._embedding,
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

    def _add_discriminator_copy(self, inputs, reuse=None, input_repr='embedded'):
        """
        Creates a copy of the discriminator in the computation graph
        Based on https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/toward-control/model.py
        :param inputs: tensor of size [batch x time x emb] of words to discriminate
        :param reuse: whether to reuse variables
        :param input_repr: whether inputs are represented by 'embedded', 'one_hot', or 'softmax' representations
        :return: probability that inputs are genuine rather than synthetic
        """
        hps = self._hps

        if input_repr == 'embedded':
            emb_inputs = inputs
        elif input_repr == 'one_hot':
            emb_inputs = tf.nn.embedding_lookup(self._embedding, inputs)
        elif input_repr == 'softmax':
            flat_inputs = tf.reshape(inputs, [hps.batch_size * (hps.dec_timesteps - 1), self._vocab_size])
            log.trace('Flattened inputs to %s', flat_inputs)
            emb_flat_inputs = tf.matmul(flat_inputs, self._embedding)
            log.trace('Embedded flattened inputs to %s', flat_inputs)
            emb_inputs = tf.reshape(emb_flat_inputs, [hps.batch_size, hps.dec_timesteps - 1, hps.emb_dim])
        else:
            raise ValueError("input_repr must be 'embedded', 'one_hot' or 'softmax'")
        log.trace('Adding discriminator with %s input %s', input_repr, emb_inputs)
        emb_inputs = tf.layers.dropout(emb_inputs, hps.cnn_dropout_rate, training=self._is_training)

        filter_outputs = []
        for i, k in enumerate([3, 4, 5]):
            conv = tf.layers.conv1d(emb_inputs, hps.cnn_filters, k, activation=tf.nn.elu,
                                    reuse=reuse, name='conv%d' % i)
            log.trace('Added convolutional layer %d', i)
            seq_len = hps.dec_timesteps - k
            pool = tf.layers.max_pooling1d(conv, seq_len, 1)
            outputs = tf.reshape(pool, [hps.batch_size, hps.cnn_filters])
            filter_outputs.append(outputs)
        cnn_output = tf.concat(filter_outputs, axis=-1)

        output = layers.fully_connected(cnn_output, 1, activation_fn=tf.nn.sigmoid, reuse=reuse, scope='out')
        return output

    def _add_discriminator(self):
        """
        Add discriminator module to the model. Creates two copies in the computation graph:
        (1) one to discriminate real/sampled data (stored in _d_s) and
        (2) another to discriminate fake/generated data (stored in _d_g)

        Also adds the discriminator *and generator* loss to the computation graph
        :return: None
        """

        log.trace('Adding discriminator for %s', self._decoder_outputs_train)
        self._d_g = self._add_discriminator_copy(self._decoder_outputs_train, reuse=False, input_repr='softmax')
        log.trace('Adding discriminator for %s', self._emb_decoder_inputs)
        self._d_s = self._add_discriminator_copy(self._emb_decoder_inputs, reuse=True, input_repr='embedded')

        # Define discriminator loss
        self._discriminator_loss = 1 - tf.reduce_mean(tf.log(self._d_s) + tf.log(1 - self._d_g))

        # Define (generator) loss
        self._loss = self._decoder_loss - tf.reduce_mean(tf.log(self._d_g))
        if self.coupled_discriminator_loss:
            self._loss += -tf.reduce_mean(tf.log(1 - self._d_g))

    def _add_train_ops(self):
        """Defines operations to run training the generator/discriminator parts of the model.
        :return: None
        """
        log.debug('Adding training optimization steps to computation graph...')
        hps = self._hps

        train_vars = tf.trainable_variables()

        # Generator training
        g_vars = [v for v in train_vars if v.name.startswith('g/')]
        self.generator_step = tf.Variable(0, trainable=False)
        g_optimizer = tf.train.AdamOptimizer(hps.lr)
        self._g_opt_step = g_optimizer.minimize(self._loss, global_step = self.generator_step, var_list=g_vars)

        # Discriminator training
        d_vars = [v for v in train_vars if v.name.startswith('d/')]
        self.discriminator_step = tf.Variable(0, trainable=False)
        d_optimizer = tf.train.GradientDescentOptimizer(hps.lr)
        self._d_opt_step = d_optimizer.minimize(self._discriminator_loss,
                                                global_step = self.discriminator_step, var_list=d_vars)

    def _add_summaries(self):
        self._summaries = tf.summary.merge_all()

    def build_graph(self):
        """
        Construct the computation graph for the ASRM, including training steps, summaries, and model saver
        :return: None
        """
        log.debug("[1/9] Adding placeholders...")
        self._add_placeholders()
        log.debug("[2/9] Adding embedding operations...")
        self._add_embedding_ops()
        log.debug("[3/9] Adding placeholders...")
        with tf.variable_scope("g"):
                log.debug('[4/9] Adding encoder / "extractor" / "reader"...')
                self._add_encoder()
                log.debug('[5/9] Adding decoder / "generator" / "writer"...')
                self._add_decoder()
        with tf.variable_scope("d"):
            log.debug('[6/9] Adding discriminator / "evaluator" / "teacher"...')
            self._add_discriminator()
        if self._is_training:
            log.debug("[7/9] Adding training operations...")
            self._add_train_ops()
        else:
            log.debug("[7/9] Skipping training operations...")
        log.debug("[8/9] Adding summaries...")
        self._add_summaries()

        log.debug("[9/9] Initializing model saver...")
        self.saver = tf.train.Saver(tf.global_variables())

    @staticmethod
    def _prepare_sections(sections, section_lens):
        # Input to the RNLM is every word except the last (e.g., everything up to </D>)
        rnlm_inputs = sections[:, :-1]
        log.trace("RNLM Input: %s; Shape: %s", rnlm_inputs, rnlm_inputs.shape)
        # Target output from the RNLM is every word except the first (e.g., everything after <D>)
        rnlm_targets = sections[:, 1:]
        log.trace("RNLM Targets: %s; Shape: %s", rnlm_targets, rnlm_targets.shape)
        # Loss weight mask so we don't worry about junk generated after the sequence ends
        target_mask = np.not_equal(rnlm_targets, record_data.PAD_ID)
        rnlm_loss_weights = target_mask.astype(int)
        log.trace("RNLM Loss Mask: %s; Shape: %s", rnlm_loss_weights, rnlm_loss_weights.shape)
        rnlm_lens = section_lens - 1
        log.trace("RNLM Lens: %s", rnlm_lens)
        return rnlm_inputs, rnlm_targets, rnlm_loss_weights, rnlm_lens

    def run_generator_train_step(self, session, records, record_lens, sections, section_lens):
        """
        Runs a single training step for training the generator (i.e. seq2seq, encoder-decoder) part of the model
        :param session: tensorflow session
        :param records: documents used as input, [batch x len] tensor of word ids
        :param record_lens: lengths of documents used as input, [batch] tensor of doc lengths
        :param sections: sections used as gold output, [batch x len] tensor of word ids
        :param section_lens: lengths of sections used as gold output, [batch] tensor of doc lengths
        :return: loss, generated output, summaries, and training step
        """
        rnlm_input, rnlm_targets, rnlm_loss_weights, rnlm_lens = self._prepare_sections(sections, section_lens)
        _, loss, output, summaries, step = session.run(
            [self._g_opt_step, self._loss, self._decoder_outputs_train, self._summaries, self.generator_step],
            feed_dict={self._records: records,
                       self._record_lens: record_lens,
                       self._sections: rnlm_input,
                       self._section_lens: rnlm_lens,
                       self._targets: rnlm_targets,
                       self._loss_weights: rnlm_loss_weights})
        return loss, output, summaries, step

    def run_discriminator_train_step(self, session, records, record_lens, sections, section_lens):
        """
        Runs a single training step for training the discriminator part of the model
        :param session: tensorflow session
        :param records: documents used as input, [batch x len] tensor of word ids
        :param record_lens: lengths of documents used as input, [batch] tensor of doc lengths
        :param sections: sections used as gold output, [batch x len] tensor of word ids
        :param section_lens: lengths of sections used as gold output, [batch] tensor of doc lengths
        :return: loss, discriminator probability of real data, discriminator probability of generated data,
                 summaries, and training step
        """
        rnlm_input, rnlm_targets, rnlm_loss_weights, rnlm_lens = self._prepare_sections(sections, section_lens)
        _, loss, d1, d2, summaries, step = session.run(
            [self._d_opt_step, self._discriminator_loss, self._d_s, self._d_g, self._summaries,
             self.discriminator_step],
            feed_dict={self._records: records,
                       self._record_lens: record_lens,
                       self._sections: rnlm_input,
                       self._section_lens: rnlm_lens,
                       self._targets: rnlm_targets,
                       self._loss_weights: rnlm_loss_weights})
        return loss, d1, d2, summaries, step

    def run_eval_step(self, session, records, record_lens):
        """
        Runs a single inference/evaluation step
        :param session: tensorflow session
        :param records: documents used as input, [batch x len] tensor of word ids
        :param record_lens: lengths of documents used as input, [batch] tensor of doc lengths
        :return: None
        """
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
        """
        Returns the encoded representations of a given batch of records
        :param session: tensorflow session
        :param records: documents used as input, [batch x len] tensor of word ids
        :param record_lens: lengths of documents used as input, [batch] tensor of doc lengths
        :return: None
        """
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
