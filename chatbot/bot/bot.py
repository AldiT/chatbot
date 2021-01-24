from telegram.ext import Updater, Dispatcher, Filters, CommandHandler, MessageHandler
from bot.credentials import bot_token, bot_user_name
from bot.handlers import start_callback, help_callback, info_callback, message_callback
import logging
import os
import sys

print("Creating the bot...", flush=True)

updater = Updater(token=bot_token, use_context=True)
dispatcher = updater.dispatcher ## updater <-> dispatcher

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

start_handler   = CommandHandler("start", start_callback)
help_handler    = CommandHandler("help", help_callback)
info_handler    = CommandHandler("info", info_callback)
message_handler = MessageHandler(Filters.text & (~Filters.command), message_callback)

dispatcher.add_handler(start_handler)
dispatcher.add_handler(help_handler)
dispatcher.add_handler(info_handler)
dispatcher.add_handler(message_handler)

print("Starting the bot...", flush=True)
updater.start_polling()