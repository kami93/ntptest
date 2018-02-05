from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.contrib.layers.python.layers import layers


class FCLayers(base.Layer):
  def __init__(self, units,
               activation=nn.relu,
               is_decoder_output=False,
               residual = True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):

    super(FCLayers, self).__init__(trainable=trainable, name=name,
                                activity_regularizer=activity_regularizer,
                                **kwargs)
    self.units = units
    self.num_layers = len(units)
    self.activation = activation
    self.is_decoder_output = is_decoder_output
    self.residual = residual
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint
    self.input_sepc = base.InputSpec(min_ndim=2)
    self.trainable = trainable

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `FCLayers` '
                       'should be defined. Found `None`.')
    self.input_spec = base.InputSpec(min_ndim=2,
                                     axes={-1: input_shape[-1].value})
    self.kernel = []
    self.bias = []


    for idx in range(self.num_layers) :
      if idx == 0:
        self.kernel.append(self.add_variable('kernel'+str(idx),
                                             shape=[input_shape[-1].value, self.units[idx]],
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             constraint=self.kernel_constraint,
                                             dtype=self.dtype,
                                             trainable=self.trainable))

      else:
        self.kernel.append(self.add_variable('kernel'+str(idx),
                                             shape=[self.units[idx-1], self.units[idx]],
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             constraint=self.kernel_constraint,
                                             dtype=self.dtype,
                                             trainable=self.trainable))

      self.bias.append(self.add_variable('bias'+str(idx),
                                             shape=[self.units[idx],],
                                             initializer=self.bias_initializer,
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint,
                                             dtype=self.dtype,
                                             trainable=self.trainable))

    self.built = True

  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    shape = inputs.get_shape().as_list()
    outputs = []
    for idx in range(self.num_layers) :
      if idx == 0:
        if len(shape) > 2:
          # Broadcasting is required for the inputs.
          output = standard_ops.tensordot(inputs, self.kernel[idx], [[len(shape) - 1],
                                                                 [0]])
          # Reshape the output back to the original ndim of the input.
          if context.in_graph_mode():
            output_shape = shape[:-1] + [self.units[idx]]
            output.set_shape(output_shape)
        else:
          output = standard_ops.matmul(inputs, self.kernel[idx])
      else:
        if len(shape) > 2:
          # Broadcasting is required for the inputs.
          output = standard_ops.tensordot(outputs[idx-1], self.kernel[idx], [[len(shape) - 1],
                                                                 [0]])
          # Reshape the output back to the original ndim of the input.
          if context.in_graph_mode():
            output_shape = shape[:-1] + [self.units[idx]]
            output.set_shape(output_shape)
        else:
          output = standard_ops.matmul(outputs[idx-1], self.kernel[idx])

      output = nn.bias_add(output, self.bias[idx])

      if self.residual :
        if self.is_decoder_output :
          if idx == 1 :
            output = output + inputs

        else:
          if idx == 2 :
            output = output + outputs[idx-2]

      if self.activation is not None:
        if not(self.is_decoder_output and idx == self.num_layers - 1):
          output = self.activation(output)

      outputs.append(output)

    return outputs[-1]

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if input_shape[-1].value is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units[-1])