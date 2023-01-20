import tensorflow as tf
from src.Utils import *
from src.EncoderLayer import *
from tensorflow.keras import layers
from src.PositionalEncoding import positional_encoding


class encoder(tf.keras.layers.Layer):
   def __init__(self, input_vocab_size, num_layers, d_model, num_heads, dff, maximum_position_encoding = 10000, dropout = 0.0):

    super(encoder, self).__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, mask_zero=True)
    self.conv1 = layers.Conv1D(d_model, 3, padding="same", activation="relu")
    self.conv2 = layers.Conv1D(d_model, 3, padding="same", activation="relu")
    self.conv3 = layers.Conv1D(d_model, 3, padding="same", activation="relu")

    self.dense = layers.Dense(d_model)
    self.concat = layers.Concatenate(axis=-1)
    self.repeat = layers.RepeatVector(100)
    self.pos = positional_encoding(maximum_position_encoding, d_model)
    self.encoder_layers = [EncoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout) for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout)

   def call(self, F0, mask=None, training=None):
    x = self.embedding(F0)

    # positional encoding
    x *= (tf.math.sqrt(tf.cast(self.d_model, tf.float32)))
    x += (self.pos[: , :tf.shape(x)[1], :])
    x = self.dropout(x, training=training)

    #Encoder layer
    embedding_mask = self.embedding.compute_mask(F0)
    for encoder_layer in self.encoder_layers:
      x = encoder_layer(x, mask = embedding_mask)
    return x

   def compute_mask(self, inputs, mask=None):
    return self.embedding.compute_mask(inputs)
