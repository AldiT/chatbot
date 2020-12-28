import numpy as np
import re

from tensorflow.keras.preprocessing.text import Tokenizer
from typing import Dict

def remove_emojis(sentence: str) -> str:
    """
    For given sentence removes the emojis.
    """
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F" # emoticons
                           u"\U0001F300-\U0001F5FF" # symbols & pictographs
                           u"\U0001F680-\U0001F6FF" # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r'', sentence)


def preprocess_data(data_path: str) -> [str]:
    """
    Function which takes the data path as input and returns the preprocessed data.
    """

    # create file
    data_file = open(data_path, 'r')

    # read in the data from the file
    raw_text = data_file.read()

    # get sentences
    sentences = raw_text.split("\n")

    preprocessed_data = []

    for sentence in sentences:
        sentence = "<start> " + remove_emojis(sentence) + " <end>"
    
        preprocessed_data.append(sentence)

    return preprocessed_data


def glove_vectors(glove_path: str) -> Dict:

    # define dictionary
    glove_vectors = dict()

    # open file
    glove_file = open(glove_path, encoding="utf-8")

    for line in glove_file:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:])
        glove_vectors[word] = vectors

    glove_file.close()

    return glove_vectors

def word_vector_matrix(text: [str], glove_vectors: Dict):

    token = Tokenizer()
    token.fit_on_texts(text)

    vocab_size = len(token.index_word) 

    word_vector_matrix = np.zeros((vocab_size + 1, 200))

    for word, index in token.word_index.items():
        vector = glove_vectors.get(word)

        if vector is not None:
            word_vector_matrix[index] = vector
        else:
            print(word)