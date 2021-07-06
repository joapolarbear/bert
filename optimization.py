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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import json
import tensorflow as tf
try:
  import byteps.tensorflow as bps
except:
  import horovod.tensorflow as bps

from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse as parse_protobuf_json

try:
  tf_get_variable = tf.get_variable
except AttributeError:
  tf_get_variable = tf.compat.v1.get_variable

if os.environ.get("BPF_ENABLE_RECOMPUTE", "") == '1':
    from horovod.tensorflow import memory_saving_gradients # monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
    # tf.__dict__["gradients"] = memory_saving_gradients.gradients_speed
    tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory
    print("================= Enable Re-computation =================")

# def dump_computation_graph(trace_dir):
#     graphdef = tf.compat.v1.get_default_graph().as_graph_def()
#     graph_str = json.loads(MessageToJson(graphdef))
#     if not os.path.isdir(trace_dir):
#       os.makedirs(trace_dir)
#     with open(os.path.join(trace_dir, "graph.json"), "w") as f:
#         json.dump(graph_str, f, indent=4)

class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applies a warmup schedule on a given learning rate decay schedule."""

  def __init__(
      self,
      initial_learning_rate,
      decay_schedule_fn,
      warmup_steps,
      start_warmup_step=0,
      power=1.0,
      name=None):
    super(WarmUp, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.power = power
    self.decay_schedule_fn = decay_schedule_fn
    self.name = name
    self.start_warmup_step = start_warmup_step

  def __call__(self, step):
    with tf.name_scope(self.name or 'WarmUp') as name:
      # Implements polynomial warmup. i.e.,
      # if global_step - start_warmup_step < warmup_steps, the learning rate
      # will be `(global_step - start_warmup_step)/num_warmup_steps * init_lr`.
      step_int = tf.cast(step, tf.int32)
      start_warmup_int = tf.constant(self.start_warmup_step, tf.int32)
      global_step_float = tf.cast(step_int - start_warmup_int, tf.float32)
      warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = (
          self.initial_learning_rate *
          tf.math.pow(warmup_percent_done, self.power))
      return tf.cond(global_step_float < warmup_steps_float,
                     lambda: warmup_learning_rate,
                     lambda: self.decay_schedule_fn(step),
                     name=name)

  def get_config(self):
    return {
        'initial_learning_rate': self.initial_learning_rate,
        'decay_schedule_fn': self.decay_schedule_fn,
        'warmup_steps': self.warmup_steps,
        'power': self.power,
        'name': self.name
    }


def create_optimizer(
  init_lr, num_train_steps, num_warmup_steps, use_amp=False,
  optimizer_type='adamw',
  poly_power=1.0,
  start_warmup_step=0,
  weight_decay_rate=0.01,
  beta_1=0.9,
  beta_2=0.999,
  epsilon=1e-6):
  """Creates an optimizer with learning rate schedule."""
  # Implements linear decay of the learning rate.
  learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=init_lr,
      decay_steps=num_train_steps,
      power=poly_power,
      end_learning_rate=0.0)
  if num_warmup_steps:
    learning_rate_fn = WarmUp(initial_learning_rate=init_lr,
                              decay_schedule_fn=learning_rate_fn,
                              warmup_steps=num_warmup_steps,
                              start_warmup_step=start_warmup_step)
  use_experimental_compile = True if tf.config.list_physical_devices(
  'GPU') else False

  use_experimental_compile = False

  if optimizer_type == 'adamw':
    print('using Adamw optimizer')
    optimizer = AdamWeightDecay(
        learning_rate=learning_rate_fn,
        weight_decay_rate=weight_decay_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        exclude_from_weight_decay=['layer_norm', 'bias'])
  elif optimizer_type == 'lamb':
    print('using Lamb optimizer')
    optimizer = LAMBOptimizer(
        learning_rate=learning_rate_fn,
        weight_decay_rate=weight_decay_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'],
        use_experimental_compile=use_experimental_compile)
  else:
    raise ValueError('Unsupported optimizer type: ', optimizer_type)

  if use_amp:
    # auto mixed precision training
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

  return optimizer

# class AdamWeightDecayOptimizer(tf.keras.optimizers.Adam):
#   """A basic Adam optimizer that includes "correct" L2 weight decay."""

#   def __init__(self,
#                learning_rate,
#                weight_decay_rate=0.0,
#                beta_1=0.9,
#                beta_2=0.999,
#                epsilon=1e-6,
#                exclude_from_weight_decay=None,
#                name="AdamWeightDecayOptimizer",
#                amsgrad=False,
#                **kwargs):
#     """Constructs a AdamWeightDecayOptimizer."""
#     super(AdamWeightDecayOptimizer, self).__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)

#     self.weight_decay_rate = weight_decay_rate
#     self.beta_1 = beta_1
#     self.beta_2 = beta_2
#     self.epsilon = epsilon
#     self.exclude_from_weight_decay = exclude_from_weight_decay

#   def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
#     """See base class."""
#     return super(AdamWeightDecay, self).apply_gradients(
#         grads_and_vars,
#         name=name,
#         experimental_aggregate_gradients=experimental_aggregate_gradients)

#   def _do_use_weight_decay(self, param_name):
#     """Whether to use L2 weight decay for `param_name`."""
#     if not self.weight_decay_rate:
#       return False
#     if self.exclude_from_weight_decay:
#       for r in self.exclude_from_weight_decay:
#         if re.search(r, param_name) is not None:
#           return False
#     return True

#   def _get_variable_name(self, param_name):
#     """Get the variable name from the tensor name."""
#     m = re.match("^(.*):\\d+$", param_name)
#     if m is not None:
#       param_name = m.group(1)
#     return param_name


class AdamWeightDecay(tf.keras.optimizers.Adam):
  """Adam enables L2 weight decay and clip_by_global_norm on gradients.

  Just adding the square of the weights to the loss function is *not* the
  correct way of using L2 regularization/weight decay with Adam, since that will
  interact with the m and v parameters in strange ways.

  Instead we want ot decay the weights in a manner that doesn't interact with
  the m/v parameters. This is equivalent to adding the square of the weights to
  the loss with plain (non-momentum) SGD.
  """

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               weight_decay_rate=0.0,
               include_in_weight_decay=None,
               exclude_from_weight_decay=None,
               name='AdamWeightDecay',
               **kwargs):
    super(AdamWeightDecay, self).__init__(
        learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
    self.weight_decay_rate = weight_decay_rate
    self._include_in_weight_decay = include_in_weight_decay
    self._exclude_from_weight_decay = exclude_from_weight_decay

  @classmethod
  def from_config(cls, config):
    """Creates an optimizer from its config with WarmUp custom object."""
    custom_objects = {'WarmUp': WarmUp}
    return super(AdamWeightDecay, cls).from_config(
        config, custom_objects=custom_objects)

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype,
                                                apply_state)
    apply_state[(var_device, var_dtype)]['weight_decay_rate'] = tf.constant(
        self.weight_decay_rate, name='adam_weight_decay_rate')

  def _decay_weights_op(self, var, learning_rate, apply_state):
    do_decay = self._do_use_weight_decay(var.name)
    if do_decay:
      return var.assign_sub(
          learning_rate * var *
          apply_state[(var.device, var.dtype.base_dtype)]['weight_decay_rate'],
          use_locking=self._use_locking)
    return tf.no_op()

  def apply_gradients(self,
                      grads_and_vars,
                      name=None,
                      experimental_aggregate_gradients=True):
    return super(AdamWeightDecay, self).apply_gradients(
        grads_and_vars,
        name=name,
        experimental_aggregate_gradients=experimental_aggregate_gradients)

  def _get_lr(self, var_device, var_dtype, apply_state):
    """Retrieves the learning rate with the given state."""
    if apply_state is None:
      return self._decayed_lr_t[var_dtype], {}

    apply_state = apply_state or {}
    coefficients = apply_state.get((var_device, var_dtype))
    if coefficients is None:
      coefficients = self._fallback_apply_state(var_device, var_dtype)
      apply_state[(var_device, var_dtype)] = coefficients

    return coefficients['lr_t'], dict(apply_state=apply_state)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
    decay = self._decay_weights_op(var, lr_t, apply_state)
    with tf.control_dependencies([decay]):
      return super(AdamWeightDecay, self)._resource_apply_dense(
          grad, var, **kwargs)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
    decay = self._decay_weights_op(var, lr_t, apply_state)
    with tf.control_dependencies([decay]):
      return super(AdamWeightDecay, self)._resource_apply_sparse(
          grad, var, indices, **kwargs)

  def get_config(self):
    config = super(AdamWeightDecay, self).get_config()
    config.update({
        'weight_decay_rate': self.weight_decay_rate,
    })
    return config

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if self.weight_decay_rate == 0:
      return False

    if self._include_in_weight_decay:
      for r in self._include_in_weight_decay:
        if re.search(r, param_name) is not None:
          return True

    if self._exclude_from_weight_decay:
      for r in self._exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True
