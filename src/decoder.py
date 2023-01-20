import tensorflow as tf
from src.Utils import *
from src.DecoderLayer import *
from tensorflow.keras import layers

class Decoder(tf.keras.layers.Layer):
  def __init__(self, target_vocab_size, num_layers, d_model, num_heads, dff, maximum_position_encoding, dropout):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model, mask_zero=True)
    self.pos = positional_encoding(maximum_position_encoding, d_model)
    self.decoder_layers = [ DecoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout)  for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, inputs, mask=None, training=None):
    x = self.embedding(inputs[0])

    # positional encoding
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos[: , :tf.shape(x)[1], :]
    x = self.dropout(x, training=training)

    #Decoder layer
    embedding_mask = self.embedding.compute_mask(inputs[0])
    for decoder_layer in self.decoder_layers:
      x = decoder_layer([x,inputs[1]], mask = [embedding_mask, mask])
    return x


  # Comment this out if you want to use the masked_loss()
  def compute_mask(self, inputs, mask=None):
    return self.embedding.compute_mask(inputs[0])

