import datetime
import json
import logging
import queue
from functools import update_wrapper
from logging import Logger, getLogger

from flask import Flask, request
from telegram import (CallbackQuery, InlineKeyboardButton,
                      InlineKeyboardMarkup, ParseMode, Update, replymarkup)
from telegram.ext import (CommandHandler, Filters, MessageHandler, Updater,
                          updater)
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.callbackqueryhandler import CallbackQueryHandler
from telegram.utils.request import Request

from auth_data import token
from main import answer_bot

#from echo_v.utils import logger_factory
#args_answer = ""
#answer_func = answer_bot(args_answer)

"""
logger = getLogger(__name__)
logger_requests = logger_factory(logger=logger)
"""

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', #level,
)
logger = logging.getLogger(__name__)

CALLBACK_BUTTON1_LEFT = "callback_button1_left"
CALLBACK_BUTTON2_RIGHT = "callback_button2_right"

TITLES = {
    CALLBACK_BUTTON1_LEFT: "Помощь ⚡️",
    CALLBACK_BUTTON2_RIGHT: "УЗНАТЬ ЦЕНУ 💰",
}


def start(update: Update, context: CallbackContext):
    #answer_help = help_command()

    update.message.reply_text('Привет, этот бот с помощью нейросети\nподскажет тебе сколько стоит товар с сайта\nhttps://dental-medix.ru\nНеобходимо ввести нтересующую товарную позицию\nзатем нажать кнопку\n"УЗНАТЬ ЦЕНУ 💰"',
    #reply_markup = key_board1()
    )
    
#@logging
def key_board1():
    keyboard = [
    [InlineKeyboardButton(TITLES[CALLBACK_BUTTON1_LEFT], callback_data=CALLBACK_BUTTON1_LEFT),
    InlineKeyboardButton(TITLES[CALLBACK_BUTTON2_RIGHT], callback_data=CALLBACK_BUTTON2_RIGHT)]
    ]
    #reply_markup = InlineKeyboardMarkup(keyboard)
    return InlineKeyboardMarkup(keyboard)


def new_func(client_text):
    preFIX = 'сколько стоит '
    args_answer = preFIX + client_text 
    ans_bot = (answer_bot(args_answer))
    #An_Data = ans_bot['data']
    return ans_bot
    

def keyboard_handler_all(update: Update, context: CallbackContext):
    
    query = update.callback_query
    data = query.data
    chat_id = update.effective_message.chat_id
    current_text = update.effective_message.text

    if data == CALLBACK_BUTTON1_LEFT:
        query.edit_message_text(   # запрос на редактирование клавиатуры
            text=current_text,
            parse_mode=ParseMode.MARKDOWN,
        )
        context.bot.send_message(
            chat_id=chat_id,
            text= "Для запуска с самого начала нажми /start\nдля продолжения введите свой вопрос и нажмите отправить!",                         # "Новое сообщение\ncallback_query.data={}".format(data),
            reply_markup=key_board1(),
        )
    elif data == CALLBACK_BUTTON2_RIGHT:
        
        ans_bot = new_func(client_text)
        
        #an_b = (ans_bot['data'])
        text = "{}".format(ans_bot['data'])
        context.bot.send_message(
        chat_id=chat_id,
        text = text
        )
        '''else:
            error_message = "Вы не ввели вопрос"
        context.bot.send_message(
            chat_id=chat_id,
            text = error_message
        )'''



def help_command(update: Update, context: CallbackContext):
    update.message.reply_text("Для запуска с самого начала нажми /start\nдля продолжения введите свой вопрос и нажмите отправить!",
    reply_markup = key_board1()
    )
#@logging

    
#@logging
def button(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()
    query.edit_message_text(text=f"Выбран параметр: {query.data}",
    reply_markup = key_board1()
    )
'''логига обработки сообщения пользователя
1) при стартовом сообщении '''    

def message_handler(update: Update, context: CallbackContext):
    global client_text
    client_text = ''
    client_text = update.effective_message.text
    
    update.message.reply_text(text='Вы ввели вопрос, и нажали отправить, затем нажмите кнопку\n"УЗНАТЬ ЦЕНУ 💰"\nи дождитесь ответа нейросети(примерно 10 сек)! ',
    reply_markup = key_board1(),        
    )
    


def main():
    logger.info("запускаем бота...")
    print('start')

    reg = Request(
        connect_timeout=0.5,
        read_timeout=1.0,
    )
    updater=Updater(
        token,
        use_context=True
    )

    handler1 = CommandHandler('start', start)
    updater.dispatcher.add_handler(handler1)

    handler_button = CallbackQueryHandler(callback=keyboard_handler_all)
    updater.dispatcher.add_handler(handler_button)

    handler2 = CallbackQueryHandler(callback=button)
    updater.dispatcher.add_handler(handler2)

    handler3 = CommandHandler('help', help_command)
    updater.dispatcher.add_handler(handler3)

    handler = MessageHandler(Filters.update, message_handler) #_Command
    updater.dispatcher.add_handler(handler)

    updater.start_polling()
    updater.idle()

if __name__=='__main__':
    main()
