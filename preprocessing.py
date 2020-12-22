import re

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


