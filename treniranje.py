import os
import keras
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import pickle

X = pickle.load(open("X.pickle","rb"))
Y = pickle.load(open("y.pickle","rb"))

checkpoint_path = 'training_1/cp-{epoch:04d}.h5'

X = X/255.0

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
callback = [ModelCheckpoint(checkpoint_path)]

model.fit(X,Y, batch_size=40,epochs=10,validation_split=0.1,callbacks=callback)
 