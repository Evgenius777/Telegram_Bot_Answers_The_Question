# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# #Импорт библиотек

# %%
#from google.colab import files # модуль для загрузки файлов в colab
import numpy as np #библиотека для работы с массивами данных
import os
from tensorflow.keras.models import Model, load_model # из кераса подгружаем абстрактный класс базовой модели, метод загрузки предобученной модели
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Flatten # из кераса загружаем необходимые слои для нейросети
from tensorflow.keras.optimizers import RMSprop, Adadelta # из кераса загружаем выбранный оптимизатор
from tensorflow.keras.preprocessing.sequence import pad_sequences # загружаем метод ограничения последовательности заданной длиной
from tensorflow.keras.preprocessing.text import Tokenizer # загружаем токенизатор кераса для обработки текста
from tensorflow.keras import utils # загружаем утилиты кераса для one hot кодировки
from tensorflow.keras.utils import plot_model # удобный график для визуализации архитектуры модели
import matplotlib.pyplot as plt
#import yaml # импортируем модуль для удобной работы с файлами
import tensorflow as tf
import json
import requests
from flask import Flask, request
import os


# %%
ques = np.load('questions.npy')
ans = np.load('answers.npy')
questions1 = list(ques)
answers1 = list(ans)


# %%
######################
# Подключаем керасовский токенизатор и собираем словарь индексов
######################
tokenizer = Tokenizer(num_words=10000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,  char_level=False, oov_token='unknown')
tokenizer.fit_on_texts(questions1 + answers1) # загружаем в токенизатор список вопросов-ответов для сборки словаря частотности
vocabularyItems = list(tokenizer.word_index.items()) # список с cодержимым словаря
vocabularySize = len(vocabularyItems)+1 # размер словаря
#print( 'Фрагмент словаря : {}'.format(vocabularyItems[:150]))
#print( 'Размер словаря : {}'.format(vocabularySize))


# %%
maxLenQuestions = 11
maxLenAnswers = 11

# %% [markdown]
# #Нейросеть и Модель обучения

# %%
######################
# Первый входной слой, кодер, выходной слой
######################
encoderInputs = Input(shape=(11, )) # размеры на входе сетки (здесь будет encoderForInput)
# Эти данные проходят через слой Embedding (длина словаря, размерность) 
encoderEmbedding = Embedding(vocabularySize, 200,  mask_zero=True) (encoderInputs) # x_train  maskzero=rue при обучнии не берет нули
# Затем выход с Embedding пойдёт в LSTM слой, на выходе у которого будет два вектора состояния - state_h , state_c
# Вектора состояния - state_h , state_c зададутся в LSTM слое декодера в блоке ниже
encoderOutputs, state_h , state_c = LSTM(200, return_state=True)(encoderEmbedding)
encoderStates = [state_h, state_c]


# %%
######################
# Второй входной слой, декодер, выходной слой
######################
decoderInputs = Input(shape=(11, )) # размеры на входе сетки (здесь будет decoderForInput)
# Эти данные проходят через слой Embedding (длина словаря, размерность) 
# mask_zero=True - игнорировать нулевые padding при передаче в LSTM. Предотвратит вывод ответа типа: "У меня все хорошо PAD PAD PAD PAD PAD PAD.."
decoderEmbedding = Embedding(vocabularySize, 200, mask_zero=True) (decoderInputs) 
# Затем выход с Embedding пойдёт в LSTM слой, которому передаются вектора состояния - state_h , state_c

decoderLSTM = LSTM(200, return_state=True, return_sequences=True)

decoderOutputs , _ , _ = decoderLSTM (decoderEmbedding, initial_state=encoderStates)
# И от LSTM'а сигнал decoderOutputs пропускаем через полносвязный слой с софтмаксом на выходе
decoderDense = Dense(vocabularySize, activation='softmax') 
output = decoderDense (decoderOutputs)


# %%
model = load_model('model2.h5')
#filename = "weights-improvement-13-0.0002.hdf5"
#model.load_weights(filename)
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy')
#model.summary()

# %%
######################
# Создадим функцию, которая преобразует вопрос пользователя в последовательность индексов
######################
def strToTokens(sentence: str): # функция принимает строку на вход (предложение с вопросом)
  words = sentence.lower().split() # приводит предложение к нижнему регистру и разбирает на слова
  tokensList = list() # здесь будет последовательность токенов/индексов
  for word in words: # для каждого слова в предложении
    tokensList.append(tokenizer.word_index[word]) # определяем токенизатором индекс и добавляем в список

    # Функция вернёт вопрос в виде последовательности индексов, ограниченной длиной самого длинного вопроса из нашей базы вопросов
  return pad_sequences([tokensList], maxlen=maxLenQuestions , padding='post')


# %%
######################
# Устанавливаем связи между слоями рабочей модели и предобученной
######################
def loadInferenceModels():
  encoderInputs = model.input[0]   # входом энкодера рабочей модели будет первый инпут предобученной модели(input_1)
  encoderEmbedding = model.layers[2] # связываем эмбединг слои(model.layers[2] это embedding_1)
  encoderOutputs, state_h_enc, state_c_enc = model.layers[4].output # вытягиваем аутпуты из первого LSTM слоя обуч.модели и даем энкодеру(lstm_1)
  encoderStates = [state_h_enc, state_c_enc] # ложим забранные состояния в состояния энкодера
  encoderModel = Model(encoderInputs, encoderStates) # формируем модель

  decoderInputs = model.input[1]   # входом декодера рабочей модели будет второй инпут предобученной модели(input_2)
  decoderStateInput_h = Input(shape=(200 ,)) # обозначим размерность для входного слоя с состоянием state_h
  decoderStateInput_c = Input(shape=(200 ,)) # обозначим размерность для входного слоя с состоянием state_c

  decoderStatesInputs = [decoderStateInput_h, decoderStateInput_c] # возьмем оба inputs вместе и запишем в decoderStatesInputs

  decoderEmbedding = model.layers[3] # связываем эмбединг слои(model.layers[3] это embedding_2)
  decoderLSTM = model.layers[5] # связываем LSTM слои(model.layers[5] это lstm_2)
  decoderOutputs, state_h, state_c = decoderLSTM(decoderEmbedding.output, initial_state=decoderStatesInputs)
  decoderStates = [state_h, state_c] # LSTM даст нам новые состояния

  decoderDense = model.layers[6] # связываем полносвязные слои(model.layers[6] это dense_1)
  decoderOutputs = decoderDense(decoderOutputs) # выход с LSTM мы пропустим через полносвязный слой с софтмаксом

    # Определим модель декодера, на входе далее будут раскодированные ответы (decoderForInputs) и состояния
    # на выходе предсказываемый ответ и новые состояния
  decoderModel = Model([decoderInputs] + decoderStatesInputs, [decoderOutputs] + decoderStates)
  return encoderModel , decoderModel


# %%
#import warnings
#warnings.simplefilter('ignore')


# %%
######################
# Устанавливаем окончательные настройки и запускаем рабочую модель над предобученной
######################
def answer_bot(args_answer):
    encModel , decModel = loadInferenceModels() # запускаем функцию для построения модели кодера и декодера

    #for _ in range(1): # задаем количество вопросов, и на каждой итерации в этом диапазоне:
    # Получаем значения состояний, которые определит кодер в соответствии с заданным вопросом
    statesValues = encModel.predict(strToTokens(args_answer)) #(input( 'Задайте вопрос : ' )
    # Создаём пустой массив размером (1, 1)
    emptyTargetSeq = np.zeros((1, 1))    
    emptyTargetSeq[0, 0] = tokenizer.word_index['start'] # положим в пустую последовательность начальное слово 'start' в виде индекса

    stopCondition = False # зададим условие, при срабатывании которого, прекратится генерация очередного слова
    decodedTranslation = '' # здесь будет собираться генерируемый ответ
    while not stopCondition : # пока не сработало стоп-условие
      # В модель декодера подадим пустую последовательность со словом 'start' и состояния предсказанные кодером по заданному вопросу.
      # декодер заменит слово 'start' предсказанным сгенерированным словом и обновит состояния
      decOutputs , h , c = decModel.predict([emptyTargetSeq] + statesValues)
      
      #argmax пробежит по вектору decOutputs'а[0,0,15104], найдет макс.значение, и вернёт нам номер индекса под которым оно лежит в массиве
      sampledWordIndex = np.argmax(decOutputs, axis=-1) # argmax возьмем от оси, в которой 15104 элементов. Получили индекс предсказанного слова.
      sampledWord = None # создаем переменную, в которую положим слово, преобразованное на естественный язык
      for word , index in tokenizer.word_index.items():
        if sampledWordIndex == index: # если индекс выбранного слова соответствует какому-то индексу из словаря
          decodedTranslation += ' {}'.format(word) # слово, идущее под этим индексом в словаре, добавляется в итоговый ответ 
          sampledWord = word # выбранное слово фиксируем в переменную sampledWord
      
      # Если выбранным словом оказывается 'end' либо если сгенерированный ответ превышает заданную максимальную длину ответа
      if sampledWord == 'end' or len(decodedTranslation.split()) > maxLenAnswers:
        stopCondition = True # то срабатывает стоп-условие и прекращаем генерацию

      emptyTargetSeq = np.zeros((1, 1)) # создаем пустой массив
      emptyTargetSeq[0, 0] = sampledWordIndex # заносим туда индекс выбранного слова
      statesValues = [h, c] # и состояния, обновленные декодером
      # и продолжаем цикл с обновленными параметрами
      data = decodedTranslation[:-3]
      answers = json.dumps({"data":data})
      ans_b = json.loads(answers)
    return ans_b                #print(decodedTranslation[:-3])#decodedTranslation[:-3] # выводим ответ сгенерированный декодером










