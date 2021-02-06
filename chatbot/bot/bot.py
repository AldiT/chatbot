"""
This file is the main bot file. Here is where everything is started, i.e. the chatbot
"""


from telegram.ext import Updater, Dispatcher, Filters, CommandHandler, MessageHandler
from bot.credentials import bot_token, bot_user_name
from bot.handlers import start_callback, help_callback, info_callback, message_callback
import logging
import os
import sys

print("Creating the bot...", flush=True)
"""
We need an updater and a dispatcher to link this instance of the chatbot with the telegram servers to send 
and receive messages
"""
updater = Updater(token=bot_token, use_context=True)
dispatcher = updater.dispatcher ## updater <-> dispatcher

##logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

start_handler   = CommandHandler("start", start_callback) #Register the start handler --> /start
help_handler    = CommandHandler("help", help_callback) # Register the help handler --> /help
info_handler    = CommandHandler("info", info_callback) # Register the info handler --> /info
message_handler = MessageHandler(Filters.text & (~Filters.command), message_callback) # Register the free text handler

# Add all handlers to the distpacher so that it knows what to call when each of this events occurs
dispatcher.add_handler(start_handler)
dispatcher.add_handler(help_handler)
dispatcher.add_handler(info_handler)
dispatcher.add_handler(message_handler)

print("Starting the bot...", flush=True)
updater.start_polling() ##Start the bot