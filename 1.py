# import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

# from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout, AveragePooling2D
from keras import Sequential
import matplotlib.pyplot as plt

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

import time

LOG_DIR = f"{int(time.time())}"

import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt

import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras import backend as K
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.compat.v1.Session(config=config)


DATA_PATH = "/media/td/B4DAA25FDAA21E1C1/snore_dataset/snore_train_dataset/test (1)/0112ACtest"

import scipy.io.wavfile as wav
import python_speech_features


########################################################### THESE FUNCTIONS HAVE BEEN BORROWED FROM Weights & Biases video
# convert file to wav2mfcc
# Mel-frequency cepstral coefficients


def wav2mfcc(file_path, n_mfcc=64, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    # sr, wave = wav.read(factile_path)

    wave = np.asfortranarray(wave[::3])

    mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=n_mfcc)
    # mfcc = python_speech_features.mfcc(wave, sr)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc


def save_data_to_array(path=DATA_PATH, max_len=11, n_mfcc=64):
    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + '/' + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len, n_mfcc=n_mfcc)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)


def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    indices = range(0, len(labels))

    # Getting first arrays
    X = np.load(labels[0] + '.npy', allow_pickle=True)
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy', allow_pickle=True)
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)


# print(prepare_dataset(DATA_PATH))

###################################################################################


wandb.init(project="csy")
config = wandb.config

config.max_len = 11
config.buckets = 64

# Save data to array file first


commandsdataset = '/media/td/B4DAA25FDAA21E1C1/snore_dataset/snore_train_dataset/test (1)/0112ACtest'

labels = []
for each in os.listdir(commandsdataset):
    if each != '.DS_Store':
        labels.append(each)

save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)

X_train, X_test, y_train, y_test = get_train_test()

import math


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def create_datagen():
    return ImageDataGenerator(
 )

channels = 1
config.update({"epochs": 50, "batch_size": int(round_down(len(X_train) / 50))}, allow_val_change=True)

num_classes = 2

X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len, channels)
X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len, channels)

plt.imshow(X_train[100, :, :, 0])
print(y_train[100])

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(256, kernel_size=(7, 7), padding='same', input_shape=(config.buckets, config.max_len, channels),
                 activation='relu', strides=[1, 1]))  # strides = 1, what shape? square? 1,11? 64,1?
model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=[2, 2]))  # , strides = 2
# model.add(Dropout(0.01))

model.add(Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same', strides=[1, 1]))  # strides = 1
model.add(MaxPooling2D(pool_size=(2, 2), strides=[2, 2]))
# model.add(Dropout(0.01))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', strides=[1, 1]))  # strides = 1
model.add(MaxPooling2D(pool_size=(2, 2), strides=[2, 2]))

model.add(Flatten())  # input_shape = (config.buckets, config.max_len, channels)

model.add(Dense(8192, activation='relu'))
# model.add(Dropout(0.01))

model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.01))

model.add(Dense(2048, activation='relu'))
# model.add(Dropout(0.01))

model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.01))

model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.01))

model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.01))

model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.01))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.summary()

wandb.init()
data_generator = create_datagen().flow(X_train,
                                       y_train_hot,
                                       batch_size=64,
                                       seed=77)


model.fit_generator(data_generator, epochs=50, validation_data=(X_test, y_test_hot),
          callbacks=[WandbCallback(data_type="image", labels=labels)])


predictions = model.predict(X_test)


# X_test.shape
# predictions.shape
# labels = [8,9,3,1,0,7,2,6,5,4]
labels = [0, 1]
preds = []
for each in predictions:
    index = np.argmax(each)
    preds.append(labels[index])

correct = []
for each in range(0, len(preds)):
    if preds[each] == y_test[each]:
        correct.append(1)
    else:
        correct.append(0)

accuracy = sum(correct) / len(correct)
print(accuracy)

print(tf.math.confusion_matrix(y_test, preds))

