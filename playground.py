import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer


human_data = preprocessing.preprocess_data("data/human_text.txt")
robot_data = preprocessing.preprocess_data("data/robot_text.txt")

# get glove vectors (takes about 55s)
glove_vectors = preprocessing.glove_vectors("data/glove/glove.twitter.27B.200d.txt")

preprocessing.word_vector_matrix(human_data, glove_vectors)
