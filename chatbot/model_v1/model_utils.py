import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import os
import sys
import pickle as pkl
import re

weight_path_encoder = "./model_v1/model_1/encoder/weights.ckpt"
weight_path_decoder = "./model_v1/model_1/encoder/weights.ckpt"
tokenizer_path = "./model_v1/model_1/tokenizer.pkl"
encoder_embedding_layer_path = "./model_v1/model_1/encoder_embedding_layer.pkl"
decoder_embedding_layer_path = "./model_v1/model_1/decoder_embedding_layer.pkl"

vocab_size = 30000 + 1
units = 1024
embedding_dim = 100
BATCH_SIZE=32

with open(tokenizer_path, "rb") as handle:
    tokenizer = pkl.load(handle)

with open(encoder_embedding_layer_path, "rb") as handle:
    encoder_embedding_variables = pkl.load(handle)

with open(decoder_embedding_layer_path, "rb") as handle:
    decoder_embedding_variables = pkl.load(handle)



class EncoderAttention(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dims, hidden_units):
        super().__init__()
        self.hidden_units = hidden_units
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dims, tf.keras.initializers.Constant(encoder_embedding_variables),
                trainable=True)
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


class DecoderAttention(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super().__init__()
        
        
        self.embedding_layer = self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, tf.keras.initializers.Constant(decoder_embedding_variables),
                trainable=True)

        self.lstm_cell = tf.keras.layers.LSTMCell(hidden_units)

        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        self.attention_mechanism = tfa.seq2seq.LuongAttention(hidden_units, memory_sequence_length=BATCH_SIZE*[15]) #N

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
        outputs, _, _ = self.decoder(embedding, initial_state=initial_state, sequence_length=BATCH_SIZE*[15-1])
        return outputs


example_x, example_y = tf.random.uniform((BATCH_SIZE, 15)), tf.random.uniform((BATCH_SIZE, 15))

##ENCODER
encoder = EncoderAttention(vocab_size, embedding_dim, units)
# Test  the encoder
sample_initial_state = encoder.initialize_hidden_state()
sample_output, sample_h, sample_c = encoder(example_x, sample_initial_state)
print(sample_output.shape)
print(sample_h.shape)


##DECODER
decoder = DecoderAttention(vocab_size, embedding_dim, units)
decoder.attention_mechanism.setup_memory(sample_output) # Attention needs the last output of the Encoder as starting point
initial_state = decoder.build_initial_state(BATCH_SIZE, [sample_h, sample_c]) # N
sample_decoder_output = decoder(example_y, initial_state)


encoder.load_weights(weight_path_encoder)
decoder.load_weights(weight_path_decoder)

print("All set")

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
        input = tf.keras.preprocessing.sequence.pad_sequences(sentence_tokens, maxlen=15, padding='post')
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
    return tokenizer.sequences_to_texts(result_sequence)[0]

print(reply("So now you are running on my machine?"))