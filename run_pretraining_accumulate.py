# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization_accumulate
import tensorflow as tf
import numpy as np
import timeit
import time
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse as parse_protobuf_json

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_string("model", "BERT_BASE", "Model.")

try:
    import byteps.tensorflow as bps
    print("Use BytePS as the communication backend.")
except:
    import horovod.tensorflow as bps
    print("Use Horovod as the communication backend.")


def model_fn_builder(bert_config, learning_rate,
                     num_train_steps, num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]
        next_sentence_labels = features["next_sentence_labels"]

        is_training = True

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
             bert_config, model.get_sequence_output(), model.get_embedding_table(),
             masked_lm_positions, masked_lm_ids, masked_lm_weights)

        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
             bert_config, model.get_pooled_output(), next_sentence_labels)

        total_loss = masked_lm_loss + next_sentence_loss

        tvars = tf.trainable_variables()

        # tf.logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #   tf.logging.info("  name = %s, shape = %s", var.name, var.shape)

        output_spec = None
        train_op,  zero_ops, accum_ops = optimization.create_optimizer(
            total_loss, learning_rate, num_train_steps, num_warmup_steps, False)

        return train_op, zero_ops, accum_ops

    return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = - \
            tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.variable_scope("cls/seq_relationship"):
        output_weights = tf.get_variable(
            "output_weights",
            shape=[2, bert_config.hidden_size],
            initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.get_variable(
            "output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        def ret_tensor(a, b, dtype):
            return tf.ones((1, a, b), dtype=dtype)
        d = tf.data.Dataset.from_tensor_slices({
            "masked_lm_positions": ret_tensor(batch_size, 20, tf.int32),
            "input_mask": ret_tensor(batch_size, 128, tf.int32),
            "masked_lm_weights": ret_tensor(batch_size, 20, tf.float32),
            "segment_ids": ret_tensor(batch_size, 128, tf.int32),
            "masked_lm_ids": ret_tensor(batch_size, 20, tf.int32),
            "next_sentence_labels": ret_tensor(batch_size, 1, tf.int32),
            "input_ids": ret_tensor(batch_size, 128, tf.int32)
        })
        d = d.repeat()
        return d

        name_to_features = {
            "input_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels":
                tf.FixedLenFeature([1], tf.int64),
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def add_infer_shape_ops(graph=None):
    # add output_shape ops
    if graph is None:
        graph = tf.compat.v1.get_default_graph()
    # collect tensor shapes
    all_ops = graph.get_operations()
    tensor_shape_ops = []
    tensor_names = []
    with graph.as_default():
        for op in all_ops:
            for output in op.outputs:
                tensor_names.append(output.name)
                name, idx = output.name.split(":")
                tensor_shape_ops.append(
                    tf.shape(output, name="final_shape/"+name+"_"+idx))
    return (tensor_names, tensor_shape_ops)


def train_input_generator(features):
    while True:
        feed_dict = {}
        for input_name, tensor in features.items():
            if "\'" in str(tensor.dtype):
                dtype_as_str = str(tensor.dtype).split("\'")[1]
            else:
                dtype_as_str = str(tensor.dtype)
            feed_dict[tensor] = np.ones(
                shape=tensor.shape).astype(dtype_as_str)
        yield feed_dict


def main(_):
    bps.init()
    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.model.lower() == "bert_base":
        config_file = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "bert_config/bert_base.json")
        bert_config = modeling.BertConfig.from_json_file(config_file)
    elif FLAGS.model.lower() == "bert_large":
        config_file = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "bert_config/bert_large.json")
        bert_config = modeling.BertConfig.from_json_file(config_file)
    else:
        bert_config = modeling.BertConfig(256)
        # raise ValueError("Invalid model {}".format(FLAGS.model))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps)

    max_seq_length = FLAGS.max_seq_length
    max_predictions_per_seq = FLAGS.max_predictions_per_seq

    with tf.name_scope("input"):
        input_ids = tf.placeholder(
            shape=[FLAGS.train_batch_size, max_seq_length], dtype=tf.int32)
        input_mask = tf.placeholder(
            shape=[FLAGS.train_batch_size, max_seq_length], dtype=tf.int32)
        segment_ids = tf.placeholder(
            shape=[FLAGS.train_batch_size, max_seq_length], dtype=tf.int32)
        masked_lm_positions = tf.placeholder(
            shape=[FLAGS.train_batch_size, max_predictions_per_seq], dtype=tf.int32)
        masked_lm_ids = tf.placeholder(
            shape=[FLAGS.train_batch_size, max_predictions_per_seq], dtype=tf.int32)
        masked_lm_weights = tf.placeholder(
            shape=[FLAGS.train_batch_size, max_predictions_per_seq], dtype=tf.float32)
        next_sentence_labels = tf.placeholder(
            shape=[FLAGS.train_batch_size, 1], dtype=tf.int32)

    features = {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids,
                "masked_lm_positions": masked_lm_positions, "masked_lm_ids": masked_lm_ids,
                "masked_lm_weights": masked_lm_weights, "next_sentence_labels": next_sentence_labels}

    train_op, zeros_op, accum_ops = model_fn(features, None, None, None)
    if os.environ.get("BPF_TEST_MEMORY", "") == "1":
        memory_summary = tf.contrib.memory_stats.MaxBytesInUse()

    infer_shape_ops = add_infer_shape_ops()

    hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        bps.BroadcastGlobalVariablesHook(0),

        # Horovod: adjust number of steps based on number of GPUs.
        # tf.train.StopAtStepHook(last_step=100)
    ]

    try:
        hooks.append(bps.TimelineHook(batch_size=FLAGS.train_batch_size))
    except:
        pass

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(bps.local_rank())

    training_batch_generator = train_input_generator(features)
    num_batches_per_iter = 10
    num_iteration = 12
    gradient_accumulation_steps = 2
    with tf.train.MonitoredTrainingSession(hooks=hooks, config=config) as mon_sess:
        # mon_sess = TimelineSession(mon_sess, infer_shape_ops)
        def benchmark_step():
            mon_sess.run(zeros_op)
            for _ in range(gradient_accumulation_steps):
                feed_dict = next(training_batch_generator)
                mon_sess.run([accum_ops], feed_dict=feed_dict)
            mon_sess.run(train_op)
        for x in range(num_iteration):
            # Run a training step synchronously.
            time_s = time.time()
            dur = timeit.timeit(benchmark_step, number=num_batches_per_iter)
            iter_time = (time.time() - time_s) / num_batches_per_iter
            if bps.rank() == 0:
                print('Iter #%d: iteration time %f ms' % (x, iter_time * 1000))
        if os.environ.get("BPF_TEST_MEMORY", "") == "1":
            print("Rank %d: Peak memory: %.2f MB" %
                  (bps.rank(), mon_sess.run(memory_summary) / (1024**2)))


if __name__ == "__main__":
    tf.app.run()