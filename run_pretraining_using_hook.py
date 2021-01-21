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
import threading
import timeit
import time
import modeling
import optimization
import tensorflow as tf
import numpy as np
try:
    import byteps.tensorflow as bps
    comm_backend = "byteps"
except:
    import horovod.tensorflow as bps
    comm_backend = "hvd"

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

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_string("model", "BERT_BASE", "Model.")

def model_fn_builder(bert_config, learning_rate,
                     num_train_steps, num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

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
    train_op = optimization.create_optimizer(
        total_loss, learning_rate, num_train_steps, num_warmup_steps, False)

    return train_op

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
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
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

from tensorflow.python.client import timeline
import json
import networkx as nx

class _SecondOrStepTimer(tf.train.SecondOrStepTimer):
    def __init__(self, every_secs=None, every_steps=None, step_bound=None):
        if step_bound is not None:
            if not (isinstance(step_bound, list) or isinstance(step_bound, tuple)):
                raise ValueError("step bound must be a list or a tuple, but {} is given".format(step_bound))
            self._start_step = step_bound[0]
            self._end_step = step_bound[1]
            if self._start_step > self._end_step:
                raise ValueError("Profiling start step must be smaller than the end step.")
        else:
            self._start_step = self._end_step = None

        super(_SecondOrStepTimer, self).__init__(every_secs, every_steps)

    def should_trigger_for_step(self, step):
        if self._start_step is not None:
            if step < self._start_step or step > self._end_step:
                return False

        return super(_SecondOrStepTimer, self).should_trigger_for_step(step)

class TimelineHook(tf.train.ProfilerHook):
    def __init__(self, _summary=False, batch_size=None):
        self.trace_dir = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(bps.local_rank()))
        if not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir)

        if os.environ.get("BYTEPS_TRACE_ON", "") != '1':
            self._end_trace = True
            self.start_step = self.end_step = 0
        else:
            self._end_trace = False
            self.start_step = int(os.environ.get("BYTEPS_TRACE_START_STEP", "20"))
            self.end_step = int(os.environ.get("BYTEPS_TRACE_END_STEP", "30"))

        if not self._end_trace and self.start_step < 1:
            raise ValueError("BYTEPS_TRACE_START_STEP must be larger than 1")
        if not self._end_trace and self.end_step <= self.start_step:
            raise ValueError("BYTEPS_TRACE_END_STEP must be larger than BYTEPS_TRACE_START_STEP")

        print("TimelineHook enable: {}  start_step: {} end_step: {}".format(not self._end_trace, self.start_step, self.end_step))
        self.dag = None
        self.has_data = False

        self.shape_dict = {}
        self.run_metadata = None
        self.partition_dag = None
        self.step_stats = []

        self._output_file = os.path.join(self.trace_dir, "timeline-{}.json")
        self._file_writer = tf.summary.FileWriterCache.get(self.trace_dir) if _summary else None
        self._show_dataflow = True
        self._show_memory = False
        self._timer = _SecondOrStepTimer(
            every_secs=None, every_steps=1, step_bound=(self.start_step, self.end_step))
        self.batch_size = batch_size
        assert self.batch_size is not None

    def before_run(self, run_context):
        if not self._end_trace:
            self._request_summary = (
                self._next_step is not None and
                self._timer.should_trigger_for_step(self._next_step))

            if self._request_summary and not self.has_data:
                ### the first step to collect traces, self.has_data tells there are data that need outputing
                self.has_data = True
            if self.has_data and not self._request_summary:
                ### the step after the last trace step, output data
                self._end_trace = True
                partition_graphs = []
                for idx in range(len(self.run_metadata.partition_graphs)):
                    graph_def = self.run_metadata.partition_graphs[idx]
                    partition_graphs.append(graph_def)
                _t = threading.Thread(target=self.output_traces, args=(tf.get_default_graph().get_operations(), partition_graphs))
                _t.start()
        else:
            self._request_summary = False

        requests = {"global_step": self._global_step_tensor}
        opts = (tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
            if self._request_summary else None)

        return tf.train.SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results["global_step"]
        if self._next_step is None:
        # Update the timer so that it does not activate until N steps or seconds
        # have passed.
            self._timer.update_last_triggered_step(stale_global_step)
        global_step = stale_global_step + 1
        if self._request_summary:
            self.run_metadata = run_values.run_metadata
            global_step = run_context.session.run(self._global_step_tensor)
            self._timer.update_last_triggered_step(global_step)
            # _t = multiprocessing.Process(target=self._save, args=(global_step, self._output_file.format(global_step),
            #          run_values.run_metadata.step_stats))
            # _t.start()
            self.step_stats.append(run_values.run_metadata.step_stats)
            # self._save(global_step, self._output_file.format(global_step),
            #         run_values.run_metadata.step_stats)
            # get shapes from step_stats
            if bps.rank() == 0 and bps.local_rank() == 0:
                if not self.shape_dict:
                    for dev_stats in run_values.run_metadata.step_stats.dev_stats:
                        for node_stats in dev_stats.node_stats:
                            for node_outputs in node_stats.output:
                                slot = node_outputs.slot
                                dtype = node_outputs.tensor_description.dtype
                                shape = []
                                if node_outputs.tensor_description.shape.unknown_rank:
                                    shape.append("Unknown")
                                else:
                                    for shape_in_dim in node_outputs.tensor_description.shape.dim:
                                        shape.append(shape_in_dim.size)
                                if node_stats.node_name+":{}".format(slot) not in self.shape_dict:
                                    self.shape_dict[node_stats.node_name+":{}".format(slot)] = {}
                                self.shape_dict[node_stats.node_name+":{}".format(slot)]["shape"] = shape
                                self.shape_dict[node_stats.node_name+":{}".format(slot)]["dtype"] = dtype
            if self._file_writer is not None:
                self._file_writer.add_run_metadata(run_values.run_metadata,
                                         "step_%d" % global_step)
        self._next_step = global_step + 1

    def output_traces(self, ops, partition_graphs):
        self.traces = {"traceEvents":[]}
        ### the ProfilerHook of tensorflow will output the timeline to self.trace_dir/timeline-{global_step}.json
        # for file in sorted(os.listdir(self.trace_dir)):
        #     if file.startswith('timeline-'):
        #         with open(os.path.join(self.trace_dir, file), 'r') as fp:
        #             ctf = json.load(fp)
        #         convert_traces = self.chome_trace_MBE2X(ctf["traceEvents"])
        #         self.traces["traceEvents"] += convert_traces

        for step_stats in self.step_stats:
            trace = timeline.Timeline(step_stats)
            events_str = trace.generate_chrome_trace_format(
                    show_dataflow=self._show_dataflow, show_memory=self._show_memory)
            events = json.loads(events_str)
            self.traces["traceEvents"] += self.chome_trace_MBE2X(events["traceEvents"])

        with open(os.path.join(self.trace_dir, "temp.json"), "w") as fp:
            json.dump(self.traces, fp, indent=4)

        if os.getenv("BYTEPS_PURE_TF_TRACE", '1') == '1':
            ### delete all intermediate redults
            _output_files = os.path.join(self.trace_dir, "timeline-*.json")
            os.system('rm {}'.format(_output_files))

        def serialize_tensor(t):
            _shape = t.shape.as_list() if t.shape.dims is not None else []
            if len(_shape) > 0 and _shape[0] is None:
                _shape[0] = self.batch_size
            return {
                "name": t.name,
                "shape": _shape,
                "dtype": t.dtype.name
            }

        for idx, graph_def in enumerate(partition_graphs):
            graph_json = json.loads(MessageToJson(graph_def))
            with open(os.path.join(self.trace_dir, "partition_def_{}.json".format(idx)), "w") as f:
                json.dump(graph_json, f, indent=4)

            if idx == 0:
                # generate dag
                self.partition_dag = nx.DiGraph()
                # clean node names in graph def
                pruned_node = set()
                all_node_names = set([node["name"] if node["name"][0] != "_" else node["name"][1:] \
                                                                    for node in graph_json["node"]])
                for node in graph_json["node"]:
                    if node["name"][0] == "_":
                        node["name"] = node["name"][1:]
                    last_slash_pos = node["name"].rfind("/")
                    if last_slash_pos != -1 and last_slash_pos < len(node["name"])-1 \
                                            and node["name"][last_slash_pos+1] == "_":
                        if node["name"][:last_slash_pos] in all_node_names:
                            pruned_node.add(node["name"])
                            continue
                        else:
                            node["name"] = node["name"][:last_slash_pos]
                    if "input" in node:
                        for idx, input_node in enumerate(node["input"]):
                            if input_node[0] == "_":
                                node["input"][idx] = input_node[1:]
                                input_node = input_node[1:]
                            last_slash_pos = input_node.rfind("/")
                            if last_slash_pos != -1 and last_slash_pos < len(input_node)-1 \
                                                    and input_node[last_slash_pos+1] == "_":
                                node["input"][idx] = input_node[:last_slash_pos]
                            self.partition_dag.add_edge(node["input"][idx].split(":")[0], node["name"])

        if bps.rank() == 0:
            ### Only dump these info for rank 0
            op_dict = {}
            for op in ops:
                op_dict[op.name] = {
                    "output":[serialize_tensor(e) for e in op.outputs],
                    "input": [serialize_tensor(e) for e in op.inputs._inputs],
                    "op": op.type
                }
            with open(os.path.join(self.trace_dir, "metadata.json"), "w") as f:
                json.dump(op_dict, f, indent=4)

            if self.partition_dag is not None:
                nx.write_gml(self.partition_dag, os.path.join(self.trace_dir, "dag.gml"), lambda x: str(x))

            with open(os.path.join(self.trace_dir, "tensor_shapes.json"), "w") as f:
                json.dump(self.shape_dict, f, indent=4)

        print("Stop tracing, output trace at %s" % self.trace_dir)

    def chome_trace_MBE2X(self, raw_traces):
        ret = []
        pid_table = {}
        if self.dag is None:
            _dag = nx.DiGraph()
        for trace in raw_traces:
            ### Create the DAG
            if self.dag is None:
                if trace["ph"] == "M" or "args" not in trace:
                    continue
                op = trace["args"]["op"]
                name = trace["args"]["name"]
                if name.startswith("^"):
                    name = name[1:]
                ### Add dependency info
                for k, v in trace["args"].items():
                    if "input" in k:
                        if v.startswith("^"):
                            v = v[1:]
                        _dag.add_edge(v, name)

            if trace["ph"] == "M":
                if trace["name"] == "process_name":
                    assert trace["pid"] not in pid_table
                    if trace["args"]["name"] == "":
                        continue
                    process_name = trace["args"]["name"]
                    if "stream:all Compute" in process_name and "device:GPU" in process_name:
                        pid_table[trace["pid"]] = {"process_name": process_name}
                else:
                    pass
            elif trace["ph"] == "i":
                trace["pid"] = trace["tid"] = "mark"
                ret.append(trace)
            elif trace["pid"] in pid_table and trace["ph"] == "X":
                cur_pid = pid_table[trace["pid"]]
                trace["pid"] = cur_pid["process_name"]
                ret.append(trace)
            else:
                pass
        if self.dag is None:
            self.dag = _dag
        return ret

def train_input_generator(features):
  while True:
    feed_dict = {}
    for input_name, tensor in features.items():
      if "\'" in str(tensor.dtype):
        dtype_as_str = str(tensor.dtype).split("\'")[1]
      else:
        dtype_as_str = str(tensor.dtype)
      feed_dict[tensor] = np.ones(shape=tensor.shape).astype(dtype_as_str)
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

#   bert_config = modeling.BertConfig(30522)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps)

  max_seq_length = FLAGS.max_seq_length
  max_predictions_per_seq = FLAGS.max_predictions_per_seq

  with tf.name_scope("input_barrier"):
    barrier_tensor = tf.random_uniform([], maxval=0.1,name="barrier_tensor")
    if comm_backend == "byteps":
        barrier_tensor = bps.push_pull(barrier_tensor)
    else:
        barrier_tensor = bps._allreduce(barrier_tensor)    

  with tf.control_dependencies([barrier_tensor]):
    with tf.name_scope("input"):
        input_ids = tf.random_uniform(shape=[FLAGS.train_batch_size, max_seq_length], dtype=tf.int32, maxval=1)
        input_mask = tf.random_uniform(shape=[FLAGS.train_batch_size, max_seq_length], dtype=tf.int32,maxval=1)
        segment_ids = tf.random_uniform(shape=[FLAGS.train_batch_size, max_seq_length], dtype=tf.int32,maxval=1)
        masked_lm_positions = tf.random_uniform(shape=[FLAGS.train_batch_size, max_predictions_per_seq], dtype=tf.int32,maxval=1)
        masked_lm_ids = tf.random_uniform(shape=[FLAGS.train_batch_size, max_predictions_per_seq], dtype=tf.int32,maxval=1)
        masked_lm_weights = tf.random_uniform(shape=[FLAGS.train_batch_size, max_predictions_per_seq], dtype=tf.float32)
        next_sentence_labels = tf.random_uniform(shape=[FLAGS.train_batch_size, 1], dtype=tf.int32,maxval=1)
  
  #with tf.name_scope("input"):
  #  input_ids = tf.placeholder(shape=[FLAGS.train_batch_size, max_seq_length], dtype=tf.int32)
  #  input_mask = tf.placeholder(shape=[FLAGS.train_batch_size, max_seq_length], dtype=tf.int32)
  #  segment_ids = tf.placeholder(shape=[FLAGS.train_batch_size, max_seq_length], dtype=tf.int32)
  #  masked_lm_positions = tf.placeholder(shape=[FLAGS.train_batch_size, max_predictions_per_seq], dtype=tf.int32)
  #  masked_lm_ids = tf.placeholder(shape=[FLAGS.train_batch_size, max_predictions_per_seq], dtype=tf.int32)
  #  masked_lm_weights = tf.placeholder(shape=[FLAGS.train_batch_size, max_predictions_per_seq], dtype=tf.float32)
  #  next_sentence_labels = tf.placeholder(shape=[FLAGS.train_batch_size, 1], dtype=tf.int32)

    features = {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids,
            "masked_lm_positions": masked_lm_positions, "masked_lm_ids": masked_lm_ids,
            "masked_lm_weights": masked_lm_weights, "next_sentence_labels": next_sentence_labels}
  
    train_op = model_fn(features, None, None, None)

  hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        bps.BroadcastGlobalVariablesHook(0),
        TimelineHook(batch_size=FLAGS.train_batch_size),
        # Horovod: adjust number of steps based on number of GPUs.
        #tf.train.StopAtStepHook(last_step=50),
  ]

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.visible_device_list = str(bps.local_rank())

  #training_batch_generator = train_input_generator(features)
  num_batches_per_iter = 10
  num_iteration = 12
  with tf.train.MonitoredTrainingSession(hooks=hooks, config=config) as mon_sess:
    # mon_sess = TimelineSession(mon_sess, infer_shape_ops)
    def benchmark_step():
        #feed_dict = next(training_batch_generator)
        mon_sess.run([train_op, barrier_tensor])
    for x in range(num_iteration):
      # Run a training step synchronously.
      time_s = time.time()
      dur = timeit.timeit(benchmark_step, number=num_batches_per_iter)
      iter_time = (time.time() - time_s) / num_batches_per_iter
      if bps.rank() == 0:
        print('Iter #%d: iteration time %f ms' % (x, iter_time * 1000))


if __name__ == "__main__":
  tf.app.run()
