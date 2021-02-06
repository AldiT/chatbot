from bot import Telegram_Chatbot
from chat_controller import Chat_Controller

# Instantiate bot with token specified in the config
my_bot = Telegram_Chatbot("config.cfg")
chat_controller = Chat_Controller()

def make_reply(message):
    return "Okay cool"

update_id = None
while True:
    updates = my_bot.get_updates(offset=update_id)
    updates = updates['result']

    if updates:
        if chat_controller.state == "Deactivated":
            chat_controller.state = "Activated"
            
        update_id, msg, sender_id = chat_controller.process_input(updates)
        reply = chat_controller.make_reply(msg)
        my_bot.send_message(reply, sender_id)