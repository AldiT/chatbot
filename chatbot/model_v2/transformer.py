# -*- coding: utf-8 -*-
"""
This file is the inference code for the transformer model which is later used from the chatbot API.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import os
import sys
import re
import pickle as pkl

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def preprocess_sentence(w):
  w = w.lower().strip()
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!]+", " ", w)
  w = w.strip()
  w = decontracted(w)

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w

#Read the tokenizer from the saved pickle
with open("./model_v2/model_2/tokenizer.pkl", "rb") as handle:
    tokenizer = pkl.load(handle)

#Read the encoder_embedding layer --> this is probably already saved with the weights but
#to make sure we save it twice
with open("./model_v2/model_2/encoder_embedding_layer.pkl", "rb") as handle:
    encoder_embedding_layer_weights = pkl.load(handle)
#Same here
with open("./model_v2/model_2/decoder_embedding_layer.pkl", "rb") as handle:
    decoder_embedding_layer_weights = pkl.load(handle)

#Set the vocabulary size + 1 to account for the padding token which has index 0
vocab_size = 30000+1

#Maximum length of a sentence
MAX_LENGTH = 15

# From here and below the code will be creating the transformer model

def pointwise_ffn(embedding_dims, expanded_dims):
    return tf.keras.Sequential([
           tf.keras.layers.Dense(expanded_dims, activation='relu'),  
           tf.keras.layers.Dense(embedding_dims)  
    ])

# padding tokens are explicitly ignored by masking 
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :]

# in the decoder we want to mask out all of the future tokens
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(input_data, target):
    # Encoder padding mask
    encoder_padding_mask = create_padding_mask(input_data)
  
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    decoder_padding_mask = create_padding_mask(input_data)
  
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    decoder_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(decoder_target_padding_mask, look_ahead_mask)
  
    return encoder_padding_mask, decoder_padding_mask, combined_mask

def calculate_self_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    
    # scaling it by the sqrt of the last dimension of k
    scaled_scores = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_scores += (mask * -1e9)  

    # softmax is normalized on the last axis (len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_scores, axis=-1)  

    output = tf.matmul(attention_weights, v)  

    return output

"""Multi-Head Attention"""

num_heads = 10
embedding_dims = 100



class MultiHeadAttention(tf.keras.layers.Layer):
    
    def __init__(self, embedding_dims, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dims = embedding_dims
        self.depth = embedding_dims // num_heads
    
        self.wq = tf.keras.layers.Dense(embedding_dims)
        self.wk = tf.keras.layers.Dense(embedding_dims)
        self.wv = tf.keras.layers.Dense(embedding_dims)
    
        self.dense = tf.keras.layers.Dense(embedding_dims)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
    
        q = self.wq(q)  # (batch_size, seq_len, embedding_dims)
        k = self.wk(k)  
        v = self.wv(v)  
    
        q = self.split_heads(q, batch_size)# (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)  
        v = self.split_heads(v, batch_size)  
    
        # self_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        self_attention = calculate_self_attention(q, k, v, mask)
    
        self_attention = tf.transpose(self_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, depth)

        concat_attention = tf.reshape(self_attention, 
                                     (batch_size, -1, self.embedding_dims))  # (batch_size, seq_len, embedding_dims)

        output = self.dense(concat_attention)
        
        return output

"""Encoder Layer"""

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, embedding_dims, num_heads, expanding_dims, rate=0.1):
    super().__init__()

    self.mha = MultiHeadAttention(embedding_dims, num_heads)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.ffn = pointwise_ffn(embedding_dims, expanding_dims)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, x, training, padding_mask):

    attn_output = self.mha(x, x, x, padding_mask)  
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  
    
    ffn_output = self.ffn(out1) 
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)   # (batch_size, input_seq_len, embedding_dims)
    
    return out2

"""Decoder Layer"""

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, embedding_dims, num_heads, expanded_dims, rate=0.1):
    super().__init__()

    self.mha1 = MultiHeadAttention(embedding_dims, num_heads)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.mha2 = MultiHeadAttention(embedding_dims, num_heads)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.ffn = pointwise_ffn(embedding_dims, expanded_dims)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6) 
    
  def call(self, x, encoder_output, training, padding_mask, look_ahead_mask):

    attn1 = self.mha1(x, x, x, look_ahead_mask)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    
    attn2 = self.mha2( encoder_output, encoder_output, out1, padding_mask)  
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  
    
    ffn_output = self.ffn(out2)  
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, embedding_dims)
    
    return out3

"""Positional Encoding"""

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, dimensions):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(dimensions)[np.newaxis, :],
                            dimensions)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

"""Encoder"""

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dims, num_heads, expanded_dims, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super().__init__()

        self.embedding_dims = embedding_dims
        self.num_layers = num_layers
    
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, embedding_dims, weights=encoder_embedding_layer_weights)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                embedding_dims)
    
        self.dropout = tf.keras.layers.Dropout(rate)

        self.encoder_layers = [EncoderLayer(embedding_dims, num_heads, expanded_dims, rate) 
                              for i in range(num_layers)]
  
    
        
    def call(self, x, training, padding_mask):

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  
        x *= tf.math.sqrt(tf.cast(self.embedding_dims, tf.float32)) # Technicality that  is used in the original paper
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
    
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, padding_mask)
    
        return x  # (batch_size, input_seq_len, embedding_dims)

"""Decoder"""

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dims, num_heads, expanded_dims, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super().__init__()

        self.embedding_dims = embedding_dims
        self.num_layers = num_layers
    
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, embedding_dims, weights=decoder_embedding_layer_weights)
        self.pos_encoding = positional_encoding(maximum_position_encoding, embedding_dims)
    
        self.decoder_layers = [DecoderLayer(embedding_dims, num_heads, expanded_dims, rate) 
                       for i in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, encoder_output, training, 
             padding_mask, look_ahead_mask):

        seq_len = tf.shape(x)[1]
    
        x = self.embedding(x)  
        x *= tf.math.sqrt(tf.cast(self.embedding_dims, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
    
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, encoder_output, training,
                                       padding_mask, look_ahead_mask)
  
        return x  # (batch_size, target_seq_len, d_model)

"""Transformer Model"""

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, embedding_dims, num_heads, expanded_dims, input_vocab_size, 
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super().__init__()

        self.encoder = Encoder(num_layers, embedding_dims, num_heads, expanded_dims, 
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, embedding_dims, num_heads, expanded_dims, 
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, input, target, training, encoder_padding_mask, 
             decoder_padding_mask, look_ahead_mask):

        encoder_output = self.encoder(input, training, encoder_padding_mask)  # (batch_size, inp_seq_len, embedding_dims)
    
    
        dec_output = self.decoder(target, encoder_output, training, decoder_padding_mask, look_ahead_mask)
        # (batch_size, target_seq_len, embedding_dims)

        final_output = self.final_layer(dec_output)  # (batch_size, target_seq_len, target_vocab_size)
    
        return final_output

num_layers = 4
expanded_dims = 512

transformer = Transformer(num_layers, embedding_dims, num_heads, expanded_dims,
                          vocab_size, vocab_size, 
                          pe_input=vocab_size, 
                          pe_target=vocab_size)

def loss_function(real, pred):
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = cross_entropy(y_true=real, y_pred=pred)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = loss * mask
    loss = tf.reduce_mean(loss)
    return loss

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_dims, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
    
        self.embedding_dims = embedding_dims
        self.embedding_dims = tf.cast(self.embedding_dims, tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step += PRE_TRAINED_STEPS
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
    
        return tf.math.rsqrt(self.embedding_dims) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(embedding_dims)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)


#load model
transformer = Transformer(num_layers, embedding_dims, num_heads, expanded_dims,
                          vocab_size, vocab_size, 
                          pe_input=vocab_size, 
                          pe_target=vocab_size)

transformer.load_weights("./model_v2/model_2/final_ckpt")



initialize_optimizer = False
#with open(ckpt_dir + "/epoch280.pkl", 'rb') as f:
#    initial_optimizer_weights = pkl.load(f)
#len(initial_optimizer_weights)



def answer(sentence, preprocess=True):
    if preprocess:
        sentence = preprocess_sentence(sentence)
        input = tokenizer.texts_to_sequences([sentence])
    else:
        input = sentence
    input = tf.convert_to_tensor(input)
  
    # as the target is German, the first word to the transformer should be the
    # German start token
    start_token = tf.convert_to_tensor([tokenizer.word_index['<start>']])
    start_token = tf.expand_dims(start_token, 0)

    # And it should stop wiht the end token
    end_token = tf.convert_to_tensor(tokenizer.word_index['<end>'])
    end_token = tf.expand_dims(end_token, 0)

    output = start_token
    for i in range(15):
        enc_padding_mask, dec_padding_mask, combined_mask, = create_masks(
                                                                input, output)

        predictions = transformer(input, 
                                    output,
                                    False,
                                    enc_padding_mask,
                                    dec_padding_mask,
                                    combined_mask
                                    )

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == end_token:
            result = tf.squeeze(output, axis=0)
            #remove the start symbol
            output = output[:, 1:]
            return tokenizer.sequences_to_texts(output.numpy())[0]

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    result = tf.squeeze(output, axis=0)
    output = output[:, 1:]
    return tokenizer.sequences_to_texts(output.numpy())[0]

print(answer("Ok I am glad, but can you tell me please what you are searching for?"))
print("HERE")
print(answer("What should I consider it again?"))
