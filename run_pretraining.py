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
import optimization
import tensorflow as tf
import numpy as np
import timeit
import time
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse as parse_protobuf_json

try:
    import byteps.tensorflow as bps
    print("Use BytePS as the communication backend.")
except:
    import horovod.tensorflow as bps
    print("Use Horovod as the communication backend.")

bps.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
  tf.config.experimental.set_visible_devices(gpus[bps.local_rank()], 'GPU')

def log(s, nl=True):
    if bps.rank() != 0:
        return
    print(s, end='\n' if nl else '')

flags = modeling.flags
tf_app = modeling.tf_app
tf_logging = modeling.tf_logging
tf_variable_scope = modeling.tf_variable_scope
tf_get_variable = modeling.tf_get_variable

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

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10, "Number of warmup steps.")

flags.DEFINE_string("model", "BERT_BASE", "Model.")

flags.DEFINE_bool("amp", False, "Whether to use AMP.")

def model_fn_builder(bert_config):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features):  # pylint: disable=unused-argument

    tf_logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf_logging.info("  name = %s, shape = %s" % (name, features[name].shape))

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

    # tf_logging.info("**** Trainable Variables ****")
    # tvars = tf.trainable_variables()
    # for var in tvars:
    #   tf_logging.info("  name = %s, shape = %s", var.name, var.shape)

    # output_spec = None
    # train_op = optimization.create_optimizer(
    #     total_loss, learning_rate, num_train_steps, num_warmup_steps, False, FLAGS.amp)

    return total_loss

  return model_fn

@tf.keras.utils.register_keras_serializable(package='Text')
# Temporary until we can create a Dense layer that ties the embedding.
class Bias(tf.keras.layers.Layer):
  """Adds a bias term to an input."""

  def __init__(self,
               initializer='zeros',
               regularizer=None,
               constraint=None,
               activation=None,
               **kwargs):
    super(Bias, self).__init__(**kwargs)
    self._initializer = tf.keras.initializers.get(initializer)
    self._regularizer = tf.keras.regularizers.get(regularizer)
    self._constraint = tf.keras.constraints.get(constraint)
    self._activation = tf.keras.activations.get(activation)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self._bias = self.add_weight(
        'bias',
        shape=input_shape[1:],
        initializer=self._initializer,
        regularizer=self._regularizer,
        constraint=self._constraint,
        dtype=self._dtype,
        trainable=True)

    super(Bias, self).build(input_shape)

  def get_config(self):
    config = {
        'activation': tf.keras.activations.serialize(self._activation),
        'initializer': tf.keras.initializers.serialize(self._initializer),
        'regularizer': tf.keras.regularizers.serialize(self._regularizer),
        'constraint': tf.keras.constraints.serialize(self._constraint)
    }
    base_config = super(Bias, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    outputs = tf.nn.bias_add(inputs, self._bias)
    if self._activation is not None:
      return self._activation(outputs)  # pylint: disable=not-callable
    else:
      return outputs

def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf_variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf_variable_scope("transform"):
      input_tensor = tf.keras.layers.Dense(
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))(input_tensor)
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = Bias(name='cls/predictions/output_bias', 
        initializer=tf.keras.initializers.Zeros())(logits)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf_variable_scope("cls/seq_relationship"):
    logits = tf.keras.layers.Dense(
        2, activation=None,
        kernel_initializer=modeling.create_initializer(bert_config.initializer_range),
        name="cls/seq_relationship")(input_tensor)
    logits = Bias(name='cls/seq_relationship/output_bias', 
        initializer=tf.keras.initializers.Zeros())(logits)
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
                tensor_shape_ops.append(tf.shape(output, name="final_shape/"+name+"_"+idx))
    return (tensor_names, tensor_shape_ops)
  
def train_input_generator(features):
  feed_dict = {}
  for input_name, tensor in features.items():
    if "\'" in str(tensor.dtype):
      dtype_as_str = str(tensor.dtype).split("\'")[1]
    else:
      dtype_as_str = str(tensor.dtype)
    shape = list(tensor.shape)
    if shape[0] is None:
      shape[0] = FLAGS.train_batch_size
    feed_dict[input_name] = np.ones(shape=shape).astype(dtype_as_str)
  return feed_dict


def main(_):
  tf_logging.set_verbosity(tf_logging.INFO)

  if FLAGS.model.lower() == "bert_base":
    config_file = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "bert_config/bert_base.json")
    bert_config = modeling.BertConfig.from_json_file(config_file)
  elif FLAGS.model.lower() == "bert_large":
    config_file = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "bert_config/bert_large.json")
    bert_config = modeling.BertConfig.from_json_file(config_file)
  else:
    # bert_config = modeling.BertConfig(256)
    raise ValueError("Invalid model {}".format(FLAGS.model))

  model_fn = model_fn_builder(bert_config=bert_config)

  max_seq_length = FLAGS.max_seq_length
  max_predictions_per_seq = FLAGS.max_predictions_per_seq
  
  with tf.name_scope("input"):
    input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
    masked_lm_positions = tf.keras.layers.Input(shape=(max_predictions_per_seq,), dtype=tf.int32, name="masked_lm_positions")
    masked_lm_ids = tf.keras.layers.Input(shape=(max_predictions_per_seq,), dtype=tf.int32, name="masked_lm_ids")
    masked_lm_weights = tf.keras.layers.Input(shape=(max_predictions_per_seq,), dtype=tf.float32, name="masked_lm_weights")
    next_sentence_labels = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="next_sentence_labels")

  features = {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids,
            "masked_lm_positions": masked_lm_positions, "masked_lm_ids": masked_lm_ids,
            "masked_lm_weights": masked_lm_weights, "next_sentence_labels": next_sentence_labels}
  
  total_loss = model_fn(features)
  if os.environ.get("BPF_TEST_MEMORY", "") == "1":
    memory_summary = tf.contrib.memory_stats.MaxBytesInUse()

  infer_shape_ops = add_infer_shape_ops()
  
  model = tf.keras.Model(inputs=features, outputs=total_loss, name=FLAGS.model.lower())
  optimizer = optimization.create_optimizer(
      init_lr=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_amp=False
  )
  # import code
  # code.interact(local=locals())

  data = train_input_generator(features)
  recorder = bps.Recorder(model=model)

  @bps.profile(recorder)
  @tf.function
  def benchmark_step(first_batch):
    """Replicated training step."""
    with tf.GradientTape() as tape:
      loss = model(data, training=True)

    # Horovod: add Horovod Distributed GradientTape.
    log("=================USING DISTRIBUTED OPTIMIZER=================")
    tape = bps.DistributedGradientTape(tape, recorder=recorder)
    if os.environ.get("BPF_ENABLE_RECOMPUTE", "") == '1':
      raise NotImplementedError()
      grads = tf.gradients(loss, tvars)
      tvars = tf.trainable_variables()
    else:
      grads = tape.gradient(loss, model.trainable_variables)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    # trace_dir = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(bps.local_rank()))
    # dump_computation_graph(trace_dir)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if first_batch:
      bps.broadcast_variables(model.variables, root_rank=0)
      bps.broadcast_variables(optimizer.variables(), root_rank=0)

  log('Running warmup...')
  benchmark_step(first_batch=True)
  timeit.timeit(lambda: benchmark_step(first_batch=False),
                number=FLAGS.num_warmup_steps-1)

  log('Running benchmark...')
  device = 'GPU'
  num_batches_per_iter = 10
  num_iteration = 20
  img_secs = []
  with tf.device(device):
    for x in range(num_iteration):
      time = timeit.timeit(lambda: benchmark_step(first_batch=False),
                            number=num_batches_per_iter)
      img_sec = FLAGS.train_batch_size * num_batches_per_iter / time
      iter_time = 1000 * time / num_batches_per_iter
      log('Iter #%d: %.1f img/sec per %s, iteration time %f ms' % (x, img_sec, device, iter_time))
      img_secs.append(img_sec)
      
      if os.environ.get("BPF_TEST_MEMORY", "") == "1":
        raise NotImplementedError()
        print("Rank %d: Peak memory: %.2f MB" % (bps.rank(), mon_sess.run(memory_summary) / (1024**2)))
      
      if x * num_batches_per_iter >= FLAGS.num_train_steps:
        break

  img_sec_mean = np.mean(img_secs)
  img_sec_conf = 1.96 * np.std(img_secs)
  log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
  log('Total img/sec on %d %s(s): %.1f +-%.1f' %
      (bps.size(), device, bps.size() * img_sec_mean, bps.size() * img_sec_conf))

if __name__ == "__main__":
  tf_app.run()
