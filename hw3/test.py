#! /usr/bin/env python
import pickle
import numpy as np
import keras
import sys
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K

def new_model(shape):
        model = Sequential()

        model.add(Convolution2D(32, 3, 3, border_mode='same',
                                dim_ordering = 'tf',
                                input_shape=shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering = 'tf'))
        model.add(Dropout(0.25))

        model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering = 'tf'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering = 'tf'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

batch_size = 100
nb_classes = 10
nb_epoch = 40


filePath = sys.argv[1]
modelName = sys.argv[2]
outputName = sys.argv[3]

f = open( filePath + 'test.p', 'rb')

test = pickle.load(f)
test = np.asarray(test['data'])
shape = test.shape
test = test.reshape(shape[0], 3, 32, 32)
test = test.swapaxes(1,2).swapaxes(2,3)

test = test.astype('float32') / 255

#model = new_model(test.shape[1:])
#model.load_weights(modelName)

model = keras.models.load_model(modelName)

result = model.predict_classes(test,batch_size=batch_size,verbose=2)

out = open(outputName,'w+')

out.write("ID,class\n")

for i in range(test.shape[0]):
        out.write(str(i) + ',' + str(result[i]) + '\n')

