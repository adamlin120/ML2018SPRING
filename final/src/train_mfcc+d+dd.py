# -*- coding: utf-8 -*-
# Change this to True to replicate the result

import numpy as np
import pickle as pk
import os
import shutil
import pandas as pd
import librosa
from sklearn.cross_validation import StratifiedKFold
from keras import losses, models, optimizers
from keras.models import load_model
from keras.activations import softmax
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import MaxPool2D, Activation
from utils import ensembleModels
from utils import Config


def get_2d_conv_model(config):

    nclass = config.n_classes

    inp = Input(shape=(config.dim[0], config.dim[1], config.dim[2],))
    x = Convolution2D(128, (4, 10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(64, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(.5)(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(.5)(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(.5)(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(.5)(x)

    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt,
                  loss=losses.categorical_crossentropy,
                  metrics=['acc'])
    return model


def prepare_data(df, config, data_dir, save=False, save_path=""):
    X = np.empty(shape=(df.shape[0],
                        config.dim[0], config.dim[1], config.dim[2]))
    input_length = config.audio_length
    for i, fname in enumerate(df.fname):
        print(fname)
        file_path = data_dir + fname
        data, _ = librosa.core.load(file_path, sr=config.sampling_rate,
                                    res_type="kaiser_fast")

        # Random offset / Padding
        # file longer than the target audio length / offsetting
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        elif len(data) < input_length:
            # shorter / padding
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
            data = np.pad(data, (offset, input_length - len(data) - offset),
                          "constant")
        else:
            offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset),
                          "constant")

        mfcc = librosa.feature.mfcc(data, sr=config.sampling_rate,
                                    n_mfcc=config.n_mfcc)
        X[i, :, :, 0] = mfcc
        if config.use_delta:
            mfcc_delta = librosa.feature.delta(mfcc)
            X[i, :, :, 1] = mfcc_delta
            if config.use_Ddelta:
                mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
                X[i, :, :, 2] = mfcc_delta_delta

    if save:
        np.save(save_path, X)

    return X


config = Config(sampling_rate=44100, audio_duration=2, n_folds=10,
                use_mfcc=True, use_delta=True, use_Ddelta=True, n_mfcc=60,
                learning_rate=1e-3, batch_size=32, max_epochs=500,
                model_name="MFCC_d_dd")
# pk.dump(config, open("config_"+config.model_name+".pk", 'wb'))

np.random.seed(config.random_seed)

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/sample_submission.csv")

LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
# train.set_index("fname", inplace=True)
# test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])

PATH_train_mfcc60_d_dd = "./model/train_mfcc60+d+dd.npy"
PATH_test_mfcc60_d_dd = "./model/test_mfcc60+d+dd.npy"
X_train = prepare_data(train, config, './data/audio_train/',
                       save=True, save_path=PATH_train_mfcc60_d_dd)
X_test = prepare_data(test, config, './data/audio_test/',
                      save=True, save_path=PATH_test_mfcc60_d_dd)
X_train = np.load(PATH_train_mfcc60_d_dd)
X_test = np.load(PATH_test_mfcc60_d_dd)
y_train = to_categorical(train.label_idx, num_classes=config.n_classes)

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

MODEL_FOLDER = "./model/" + config.model_name + "/"
if not os.path.exists(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)

models_list = []
skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)
for i, (train_split, val_split) in enumerate(skf):
    X, y = X_train[train_split], y_train[train_split]
    X_val, y_val = X_train[val_split], y_train[val_split]

    checkpoint = ModelCheckpoint(
        MODEL_FOLDER + config.model_name+'_best_{}.h5'.format(i),
        monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False)
    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=10)
    rLR = ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                            patience=6, verbose=1, mode='min')
    callbacks_list = [checkpoint, early, rLR]
    print("#"*50)
    print("Fold: ", i)
    model = get_2d_conv_model(config)
    print(model.summary())
    history = model.fit(X, y,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list,
                        batch_size=config.batch_size,
                        epochs=config.max_epochs)
    model = load_model(config.model_name+"_best_{}.h5".format(i))
    models_list.append(model)

ens_model = ensembleModels(
    models_list, Input(shape=(config.dim[0], config.dim[1], config.dim[2],)))
ens_model.save('./model/' + config.model_name+"_10fold.h5")
