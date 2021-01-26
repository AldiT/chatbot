#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import pickle as pkl
import unicodedata
import spacy
from spacy_langdetect import LanguageDetector
import time
import os
import sys
import re

with open("../from_drive/english_human.pkl", "rb") as handle:
    human_lines = pkl.load(handle)

with open("../from_drive/english_robot.pkl", "rb") as handle:
    robot_lines = pkl.load(handle)


import re

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


for i, line in tqdm(enumerate(human_lines)):
    human_lines[i] = decontracted(line)

for i, line in tqdm(enumerate(robot_lines)):
    robot_lines[i] = decontracted(line)



human_lines = [re.sub(r"\[\w+\]",'hi',line) for line in human_lines]
human_lines = [" ".join(re.findall(r"\w+",line)) for line in human_lines]
robot_lines = [re.sub(r"\[\w+\]",'',line) for line in robot_lines]
robot_lines = [" ".join(re.findall(r"\w+",line)) for line in robot_lines]
# grouping lines by response pair
pairs = list(zip(human_lines,robot_lines))
#random.shuffle(pairs)
len(pairs)


# In[10]:


input_docs = []
target_docs = []
input_tokens = set()
target_tokens = set()

for line in pairs:
    input_doc, target_doc = line[0], line[1]
    # Appending each input sentence to input_docs
    input_docs.append(input_doc)
    # Splitting words from punctuation  
    target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
    # Redefine target_doc below and append it to target_docs
    target_doc = '<START> ' + target_doc + ' <END>'
    target_docs.append(target_doc)
  
    # Now we split up each sentence into words and add each unique word to our vocabulary set
    for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
        if token not in input_tokens:
            input_tokens.add(token)
    for token in target_doc.split():
        if token not in target_tokens:
            target_tokens.add(token)
input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)
num_tokens = len(set(input_tokens + target_tokens)) + 2 # [UNK]
pairs = list(zip(input_docs, target_docs))



vocab_size = 30000 + 1
units = 1024
embedding_dim = 100



tokenizer = Tokenizer(filters='', oov_token="<unk>")
tokenizer.fit_on_texts(input_docs + target_docs)

tokenizer.num_words = vocab_size
input_docs_tokenized = tokenizer.texts_to_sequences(input_docs)
target_docs_tokenized = tokenizer.texts_to_sequences(target_docs)


final_in_docs_tokenized = []
final_tar_docs_tokenized = []

for i in range(len(input_docs_tokenized)):
  if len(input_docs_tokenized[i]) <= 15 and len(target_docs_tokenized[i]) <= 15:
    final_in_docs_tokenized.append(input_docs_tokenized[i])
    final_tar_docs_tokenized.append(target_docs_tokenized[i])
len(final_in_docs_tokenized), len(final_tar_docs_tokenized)




max_len = 0
for r in final_tar_docs_tokenized:
    if len(r) > max_len:
        max_len = len(r)
  
max_len


X = final_in_docs_tokenized
Y = final_tar_docs_tokenized

X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='pre')
Y = tf.keras.preprocessing.sequence.pad_sequences(Y, padding='pre')

del final_in_docs_tokenized
del final_tar_docs_tokenized
del input_docs
del target_docs
del input_docs_tokenized
del target_docs_tokenized
#del human_lines
#del robot_lines


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)


print("Shape of train and test: ", X_train.shape, X_test.shape)


BUFFER_SIZE = len(X_train)
BATCH_SIZE = 64
steps_per_epoch = len(X_train)//BATCH_SIZE

with open("./model_v1/model_2_v3/encoder_embedding_layer.pkl", "rb") as handle:
    encoder_embedding_layer_weights = pkl.load(handle)

with open("./model_v1/model_2_v3/decoder_embedding_layer.pkl", "rb") as handle:
    decoder_embedding_layer_weights = pkl.load(handle)

dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, 
                                                                                            drop_remainder=True)

def max_len(sentence):
    return max(len(s) for s in sentence)

max_length_input = max_len(X_train)
max_length_output = max_len(Y_train)



for example in dataset.take(1):
    example_x, example_y = example
    
print(example_x.shape) 
print(example_y.shape) 




class EncoderAttention(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dims, hidden_units):
        super().__init__()
        self.hidden_units = hidden_units
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dims, weights=encoder_embedding_layer_weights)

        self.lstm_layer = tf.keras.layers.LSTM(hidden_units, return_sequences=True, 
                                                     return_state=True ) # We need the lstm outputs 
                                                                         # to calculate attention!
    
    def initialize_hidden_state(self): 
        return [tf.zeros((BATCH_SIZE, self.hidden_units)), 
                tf.zeros((BATCH_SIZE, self.hidden_units))] 
                                                               
    def call(self, inputs, hidden_state):
        embedding = self.embedding_layer(inputs)
        output, h_state, c_state = self.lstm_layer(embedding, initial_state = hidden_state)
        return output, h_state, c_state


encoder = EncoderAttention(vocab_size, embedding_dim, units)


# Test  the encoder
sample_initial_state = encoder.initialize_hidden_state()
sample_output, sample_h, sample_c = encoder(example_x, sample_initial_state)
print(sample_output.shape)
print(sample_h.shape)


class DecoderAttention(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super().__init__()
        
        
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=decoder_embedding_layer_weights)

        self.lstm_cell = tf.keras.layers.LSTMCell(hidden_units)

        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        self.attention_mechanism = tfa.seq2seq.LuongAttention(hidden_units, memory_sequence_length=BATCH_SIZE*[len(X_train[0])]) #N

        self.attention_cell = tfa.seq2seq.AttentionWrapper(cell=self.lstm_cell, # N
                                      attention_mechanism=self.attention_mechanism, 
                                      attention_layer_size=hidden_units)

        self.output_layer = tf.keras.layers.Dense(vocab_size)
        self.decoder = tfa.seq2seq.BasicDecoder(self.attention_cell, # N
                                                sampler=self.sampler, 
                                                output_layer=self.output_layer)

    def build_initial_state(self, batch_size, encoder_state): #N
        decoder_initial_state = self.attention_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state


    def call(self, inputs, initial_state):
        embedding = self.embedding_layer(inputs)
        outputs, _, _ = self.decoder(embedding, initial_state=initial_state, sequence_length=BATCH_SIZE*[len(Y_train[0])-1])
        return outputs

decoder = DecoderAttention(vocab_size, embedding_dim, units)



# Test the decoder
sample_y = tf.random.uniform((BATCH_SIZE, len(X_train)))
decoder.attention_mechanism.setup_memory(sample_output) # Attention needs the last output of the Encoder
                                                        # as starting point
initial_state = decoder.build_initial_state(BATCH_SIZE, [sample_h, sample_c]) # N


sample_decoder_output = decoder(example_y, initial_state)

print(sample_decoder_output.rnn_output.shape)



optimizer = tf.keras.optimizers.Adam()

def loss_function(real, pred):
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = cross_entropy(y_true=real, y_pred=pred)
    mask = tf.logical_not(tf.math.equal(real,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)  # mask and loss have to have the same Tensor type
    loss = mask * loss
    loss = tf.reduce_mean(loss) # you need one loss scalar number for the mini batch
    return loss 


print(tf.config.list_physical_devices("GPU"))
import time


encoder_model_save_path = "./model_v1/model_2_v3/encoder/weights.ckpt"
decoder_model_save_path = "./model_v1/model_2_v3/decoder/weights.ckpt"


encoder.load_weights(encoder_model_save_path)
decoder.load_weights(decoder_model_save_path)


import unicodedata
def preprocess_sentence(w):
    w = w.lower().strip()
    # This next line is confusing!
    # We normalize unicode data, umlauts will be converted to normal letters
    #w = w.replace("ß", "ss")
    #w = ''.join(c for c in unicodedata.normalize('NFD', w) if unicodedata.category(c) != 'Mn')

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"\[\w+\]",'', w)
    w = " ".join(re.findall(r"\w+",w))
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

def reply(sentence, preprocess=True):

    if preprocess:
        sentence = preprocess_sentence(sentence)
        sentence_tokens = tokenizer.texts_to_sequences([sentence])
        input = tf.keras.preprocessing.sequence.pad_sequences(sentence_tokens, maxlen=max_length_input, padding='post')
    else:
        input = sentence
    input = tf.convert_to_tensor(input)

    encoder_hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
    encoder_output, encoder_h, encoder_c = encoder(input, encoder_hidden)
    start_token = tf.convert_to_tensor([tokenizer.word_index['<start>']])
    end_token = tokenizer.word_index['<end>']

    # This time we use the greedy sampler because we want the word with the highest probability!
    # We are not generating new text, where a probability sampling would be better
    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

    # Instantiate a BasicDecoder object
    decoder_instance = tfa.seq2seq.BasicDecoder(cell=decoder.attention_cell, # N
                                                sampler=greedy_sampler, output_layer=decoder.output_layer)
    # Setup Memory in decoder stack
    decoder.attention_mechanism.setup_memory(encoder_output) # N

    # set decoder_initial_state
    decoder_initial_state = decoder.build_initial_state(batch_size=1, encoder_state=[encoder_h, encoder_c]) # N

    ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder 
    ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this. 
    ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

    decoder_embedding_matrix = decoder.embedding_layer.variables[0]

    outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_token, end_token= end_token, initial_state=decoder_initial_state)

    result_sequence  = outputs.sample_id.numpy()
    result_sequence = result_sequence[:, :-1] #remove <end> from the sentence
    return tokenizer.sequences_to_texts(result_sequence)[0]
print(reply("Hi"))



print(reply("How about we go hiking?"))



