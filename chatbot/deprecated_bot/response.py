import json
import numpy as np
import re

from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model


class Response_Generator():
    def __init__(self, user_input):
        self.user_input = user_input
        self.encoder_model = build_encoder()
        self.decoder_model = build_decoder()

    def generate_response(self):
        input_matrix = string_to_matrix(self.user_input)
        chatbot_response = self.decode_response(input_matrix)
        #Remove <START> and <END> tokens from chatbot_response
        chatbot_response = chatbot_response.replace("<START>",'')
        chatbot_response = chatbot_response.replace("<END>",'')
        return chatbot_response

    def decode_response(self, test_input):

        num_decoder_tokens = 1003 # taken from training
        max_decoder_seq_length = 50 # taken from training

        # import target_features_dict
        with open('feature_dicts/target_features.json', 'r') as fp:
            target_features_dict = json.load(fp)

        # import reverse_target_features_dict
        with open('feature_dicts/reverse_target_features.json', 'r') as fp:
            reverse_target_features_dict = json.load(fp)

        #Getting the output states to pass into the decoder
        states_value = self.encoder_model.predict(test_input)
        #Generating empty target sequence of length 1
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        #Setting the first token of target sequence with the start token
        target_seq[0, 0, target_features_dict['<START>']] = 1.
        
        #A variable to store our response word by word
        decoded_sentence = ''
        
        stop_condition = False
        while not stop_condition:
            #Predicting output tokens with probabilities and states
            output_tokens, hidden_state, cell_state = self.decoder_model.predict([target_seq] + states_value)
        #Choosing the one with highest probability
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token_index = str(sampled_token_index)

            sampled_token = reverse_target_features_dict[sampled_token_index]
            decoded_sentence += " " + sampled_token
        #Stop if hit max length or found the stop token
            if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True
        #Update the target sequence
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, int(sampled_token_index)] = 1.
            #Update states
            states_value = [hidden_state, cell_state]
        return decoded_sentence


def build_encoder():
    # load training model 
    training_model = load_model('training_model.h5')

    # create encoder
    encoder_inputs = training_model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    return encoder_model


def build_decoder():
    # create decoder input
    latent_dim = 256
    num_decoder_tokens = 1003 # taken from training

    decoder_state_input_hidden = Input(shape=(latent_dim,))
    decoder_state_input_cell = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

    # decoder output
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')

    decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_hidden, state_cell]
    decoder_outputs = decoder_dense(decoder_outputs)

    # setup decoder model
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return decoder_model


#Method to convert user input into a matrix
def string_to_matrix(user_input):

    # parameters taken from training
    num_encoder_tokens = 981
    max_encoder_seq_length = 51

    # load input_features_dict
    with open('feature_dicts/input_features.json', 'r') as fp:
        input_features_dict = json.load(fp)

    tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
    user_input_matrix = np.zeros(
        (1, max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    for timestep, token in enumerate(tokens):
        if token in input_features_dict:
            user_input_matrix[0, timestep, input_features_dict[token]] = 1.
    return user_input_matrix 