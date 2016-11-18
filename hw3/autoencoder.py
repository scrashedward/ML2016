#! /usr/bin/env python

import pickle
import numpy as np
#import matplotlib.pyplot as plt
import sys
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, Activation, Flatten
from keras.models import Model, Sequential
from scipy.spatial import distance

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
        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

def new_datagen():
 # this will do preprocessing and realtime data augmentation
	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=True,  # randomly flip images
		vertical_flip=False)  # randomly flip images
        return datagen

filePath = sys.argv[1]
modelName = sys.argv[2]

f = open(filePath+'all_label.p' , 'r')

all_label = pickle.load(f)
all_label = np.asarray(all_label)
all_label = all_label.astype('float32')/255

shape = all_label.shape

label_train = all_label.reshape(shape[0]*shape[1], shape[2])
all_label.resize(shape[0] *shape [1], shape[2])
label_train.resize(shape[0]*shape[1], 3, 32, 32)
label_train = label_train.swapaxes(1,2).swapaxes(2,3)
#label_train = label_train[...,:3].dot([0.299, 0.587, 0.114])
#label_train.resize(label_train.shape[0], 32, 32, 1)

label = np.zeros((500,1))

for i in range(1,10):
        label = np.concatenate((label, np.ones((500, 1)) * i))
        #print label.shape

label = np_utils.to_categorical(label.astype(int), 10)

#reshape_all = label_train.reshape(label_train.shape[0], 32, 32)

input_img = Input(shape=(32,32,3))

x = Convolution2D(32, 3, 3, border_mode='same',dim_ordering = 'tf', activation = 'relu')(input_img)
x = Convolution2D(32, 3, 3, border_mode='same',dim_ordering = 'tf', activation = 'relu')(x)
x = MaxPooling2D((2, 2), border_mode = 'same', dim_ordering = 'tf')(x)
x = Dropout(0.25)(x)
x = Convolution2D(64, 3, 3, border_mode='same',dim_ordering = 'tf', activation = 'relu')(x)
x = Convolution2D(64, 3, 3, border_mode='same',dim_ordering = 'tf', activation = 'relu')(x)
x = MaxPooling2D((2, 2), border_mode = 'same', dim_ordering = 'tf')(x)
x = Dropout(0.25)(x)
x = Flatten()(x)

extractor = Model(input_img, x)
extractor.compile(optimizer = 'adam', loss = 'binary_crossentropy')
label_feature = extractor.predict(label_train)
print label_feature.shape

inputShape = Input(shape=(label_feature.shape[1],))

encoded = Dense(256, activation='relu')(inputShape)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(4096, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(inputShape, output=decoded)


encoder = Model(inputShape, encoded)
autoencoder = Model(inputShape, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


autoencoder.fit(label_feature, label_feature,
                nb_epoch=40,
                batch_size=512,
                shuffle=True)

trained = encoder.predict(label_feature)
print trained.shape
trained.resize(10, 500, 256)
trained =  trained.mean(axis = 1)

f = open(filePath + 'all_unlabel.p', 'r')
all_unlabel = pickle.load(f)
all_unlabel = np.asarray(all_unlabel)
all_unlabel = all_unlabel.astype('float32') / 255
shape = all_unlabel.shape
print shape
unlabel_train = all_unlabel.reshape(shape[0], 3, 32, 32)
unlabel_train = unlabel_train.swapaxes(1,2).swapaxes(2,3)
#unlabel_train = unlabel_train[...,:3].dot([0.299, 0.587, 0.114])
#unlabel_train.resize(unlabel_train.shape[0], 32, 32)

unlabel_feature = extractor.predict(label_train)
unlabel_trained = encoder.predict(unlabel_feature)

#print trained[0]
#print unlabel_trained[0]
result =  distance.cdist(unlabel_trained, trained, 'sqeuclidean')

mark = []
for i in range(result.shape[0]):
	#print result[i]
	#raw_input()
	if result[i].min() < 0.01:
		mark.append(i)
		argmin = result[i].argmin()
		result[i] = np.zeros((1,10))
		result[i][argmin] = 1
print len(mark)
all_label = np.concatenate((all_label, all_unlabel[mark]))
label = np.concatenate((label, result[mark]))

#print all_label.shape
#print label.shape

all_label.resize(all_label.shape[0], 3, 32, 32)
all_label = all_label.swapaxes(1,2).swapaxes(2,3)

model = new_model(all_label.shape[1:])
datagen = new_datagen()
datagen.fit(all_label)
model.fit_generator(datagen.flow(all_label, label,
                        batch_size=64),
                        samples_per_epoch=all_label.shape[0],
                        nb_epoch=100,
                        verbose = 2)

model.save(modelName)
#f = open('./data/test.p', 'r')
#test = pickle.load(f)
#test = np.asarray(test['data'])
#shape = test.shape
#test = test.reshape(shape[0], 3, 32, 32)
#test = test.swapaxes(1,2).swapaxes(2,3)
#test = test.astype('float32')/255

#result = model.predict_classes(test,batch_size=64,verbose=2)

#out = open('result_auto.csv','w+')

#out.write("ID,class\n")

#for i in range(10000):
#        out.write(str(i) + ',' + str(result[i]) + '\n')


