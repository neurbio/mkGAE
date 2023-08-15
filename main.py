from network import mkGAE
import tensorflow as tf
import numpy as np
from absl import flags
import sys

# Define the adjacency matrix (binary connections between nodes)
adjacency_matrix = np.array([[0, 1, 0, 1],
                             [1, 0, 1, 1],
                             [0, 1, 0, 0],
                             [1, 1, 0, 0]], dtype=np.float32)

# Define node features
node_features = np.array([[0.1, 0.2],
                          [0.3, 0.4],
                          [0.5, 0.6],
                          [0.7, 0.8]], dtype=np.float32)

# Hyperparameters3
FLAGS = flags.FLAGS
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.FLAGS(sys.argv)

num_nodes = adjacency_matrix.shape[0]
num_features = node_features.shape[1]
hidden_units = 16
learning_rate = 0.01
num_epochs = 100

# Create TensorFlow variables
adjacency_tensor = tf.constant(adjacency_matrix, dtype=tf.float32)
print(adjacency_tensor.shape)
node_features_tensor = tf.constant(node_features, dtype=tf.float32)

# Instantiate GCN model
gcn_model = mkGAE(num_features, num_nodes)

# Define loss function and optimizer
loss_fn = tf.losses.mean_squared_error
optimizer = tf.optimizers.Adam(learning_rate)

# Training loop
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        predicted_adjacency = gcn_model(adjacency_tensor, node_features_tensor)
        loss = loss_fn(node_features, predicted_adjacency)
    gradients = tape.gradient(loss, gcn_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, gcn_model.trainable_variables))
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.numpy()}")

# Get the final predicted adjacency matrix
final_predicted_adjacency = gcn_model(adjacency_tensor, node_features_tensor)
print("Final Predicted Adjacency Matrix:")
print(final_predicted_adjacency.numpy())