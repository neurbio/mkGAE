
import tensorflow as tf
import numpy as np
from absl import flags

FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

# Define Graph Convolutional Layer
class GraphConvLayer(tf.Module):
    def __init__(self, in_features, out_features):
        self.weights = tf.Variable(tf.random.normal([in_features, out_features]))

    def __call__(self, adjacency, features):
        adjacency_normalized = tf.linalg.normalize(adjacency, ord=1, axis=1)[0]
        transformed_features = tf.matmul(adjacency_normalized, features)
        output = tf.matmul(transformed_features, self.weights)
        return output


class MultiplyLayer(tf.Module):
    def __init__(self, in_features, out_features, bias=False):
        self.bias = bias
        self.weights = tf.Variable(tf.random.normal([in_features, out_features]))
        if self.bias:
            self.vars['bias'] = tf.Variable(tf.zeros([self.in_features], dtype=tf.float32), name='bias')
    def __call__(self, adjacency, features):

        output = tf.multiply(self.weights, adjacency)
        output = tf.matmul(tf.transpose(features), output)
        if self.bias:
            output = tf.add(output, self.vars['bias'])
        return output