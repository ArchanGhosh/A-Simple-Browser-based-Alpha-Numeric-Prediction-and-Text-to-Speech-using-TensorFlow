# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:51:43 2020

@author: gharc
"""

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#ALPHABET DATA LOADING
data_char = pd.read_csv('E:\Projects\hand writing Recognition/A_Z Handwritten Data.csv').astype('float32')
data_char.rename(columns={'0':'label'}, inplace=True)

#alphabets_mapper = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
alphabets_mapper = {0:10,1:11,2:12,3:13,4:14,5:15,6:16,7:17,8:18,9:19,10:20,11:21,12:22,13:23,14:24,15:25,16:26,17:27,18:28,19:29,20:30,21:31,22:32,23:33,24:34,25:35}
data_char['label'] = data_char['label'].map(alphabets_mapper)

X_char = data_char.drop('label',axis = 1)
y_char = data_char['label']

#NUMERIC TRAIN DATASET LOAD
data_num_train= pd.read_csv('E:\Projects\hand writing Recognition/mnist_train.csv').astype('float32')
data_num_train.rename(columns={'0':'label'}, inplace=True)

X_num_train = data_num_train.drop('label',axis = 1)
y_num_train = data_num_train['label']

#NUMERIC TEST DATASET LOAD
data_num_test= pd.read_csv('E:\Projects\hand writing Recognition/mnist_test.csv').astype('float32')
data_num_test.rename(columns={'0':'label'}, inplace=True)


X_num_test = data_num_test.drop('label',axis = 1)
y_num_test = data_num_test['label']


(X_char_train, X_char_test, y_char_train, y_char_test) = train_test_split(X_char, y_char)

X_train = np.vstack([X_char_train,X_num_train])
X_test = np.vstack([X_char_test,X_num_test])

y_char_train_labels=np.unique(y_char_train)
y_num_train_labels=np.unique(y_num_train)

y_train = pd.concat([y_char_train,y_num_train], axis=0, ignore_index=True)
y_test = pd.concat([y_char_test,y_num_test], axis=0, ignore_index=True)


standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train)
standard_scaler.fit(X_test)

X_train = standard_scaler.transform(X_train)
X_test = standard_scaler.transform(X_test)

X_train = X_train.reshape(X_train.shape[0], 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28).astype('float32')

X_train= X_train[..., np.newaxis]
X_test= X_test[..., np.newaxis]


y_train_labels=np.unique(y_train)
len(y_train_labels)


y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model=keras.models.Sequential([keras.layers.Conv2D(32,3,activation='relu', input_shape=[28,28,1]),
                               keras.layers.Conv2D(64, (3, 3), activation='relu'),
                               keras.layers.MaxPooling2D(pool_size=2),
                               keras.layers.Dropout(.4),
                               keras.layers.Flatten(),
                               keras.layers.Dense(128, activation='relu'),
                               keras.layers.Dense(36, activation='softmax'),
])

model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=[X_test,y_test], epochs=10)

model.save('model_name.h5')

