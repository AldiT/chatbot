
from model_v1.seq2seq_with_attention_glove import reply

def start_callback(update, context):
    msg = "Welcome to this chat! My name is Dwayne but some call me rDany (cause I am a robot, and I like the name Dany). How can I help you!"
    return context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

def help_callback(update, context):
    msg = "Some text from the help callback!"
    return context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

def info_callback(update, context):
    msg = "Some text from the info callback"
    return context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

def message_callback(update, context):
    msg = "Some text from the message callback"
    return context.bot.send_message(chat_id=update.effective_chat.id, text=reply(update.message.text))