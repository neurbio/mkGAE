
import tensorflow as tf

from layers import GraphConvLayer, MultiplyLayer
from absl import flags

FLAGS = flags.FLAGS

# Define 3-layer GCN model
class mkGAE(tf.Module):
    def __init__(self, num_features, num_nodes):
        self.layer1 = GraphConvLayer(num_features, FLAGS.hidden1)
        self.layer2 = GraphConvLayer(FLAGS.hidden1, FLAGS.hidden2)
        self.layer3 = GraphConvLayer(FLAGS.hidden2, num_nodes)
        self.express_inner = MultiplyLayer(num_nodes, num_nodes)
        self.layer4 = tf.keras.layers.Dense(FLAGS.hidden1, activation='relu')
        self.layer5 = tf.keras.layers.Dense(FLAGS.hidden1, activation='relu')
        self.layer6 = tf.keras.layers.Dense(num_nodes)

    def __call__(self, adjacency, features):
        hidden11 = tf.nn.tanh(self.layer1(adjacency, features))
        hidden21 = tf.nn.relu(self.layer2(adjacency, hidden11))
        z_adj_mean1 = self.layer3(adjacency, hidden21)
        z_adj_std1 = self.layer3(adjacency, hidden21)
        z_adj1 = z_adj_mean1 + tf.random.normal(adjacency.shape) * tf.exp(z_adj_std1)
        express_inner = self.express_inner(z_adj1, features)
        encoded1 = self.layer4(express_inner)
        encoded2 = self.layer5(encoded1)
        decode_pi = tf.nn.sigmoid(self.layer6(encoded2))
        decode_disp = tf.clip_by_value(tf.nn.softplus((self.layer6(encoded2))), 1e-4, 1e4)
        decode_mean = tf.clip_by_value(tf.exp((self.layer6(encoded2))), 1e-5, 1e6)
        z_express = tf.transpose(decode_mean)

        return z_express


