from preprocessing import remove_emojis

# open the files
human_file = open("data/human_text.txt", 'r')
robot_file = open("data/robot_text.txt", 'r')

# read the files
human_text = human_file.read()
robot_text = robot_file.read()

# create list of sentences
human_sentences = human_text.split("\n")
robot_sentences = robot_text.split("\n")

print(type(human_sentences))
    