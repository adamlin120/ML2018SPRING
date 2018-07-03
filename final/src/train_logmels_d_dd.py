# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import librosa
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import class_weight
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
from keras.layers import average
from keras.models import Model


class Config(object):
    def __init__(self,
                 model_name,
                 sampling_rate=16000, audio_duration=2, n_classes=41,
                 use_mfcc=False, use_delta=False, use_Ddelta=False,
                 n_folds=10, n_mfcc=20,
                 learning_rate=0.0001, max_epochs=50, batch_size=512,
                 random_seed=13):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.model_name = model_name
        self.use_delta = use_delta
        self.use_Ddelta = use_Ddelta

        if self.use_delta:
            if self.use_Ddelta:
                self.n_channel = 3
            else:
                self.n_channel = 2
        else:
            self.n_channel = 1

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc,
                        1 + int(np.floor(self.audio_length/(512))),
                        self.n_channel)
        else:
            self.dim = (self.audio_length, 1)


def ensembleModels(models, model_input, model_name=None):
    # collect outputs of models in a list
    yModels = [model(model_input) for model in models]
    # averaging outputs
    yAvg = average(yModels)
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name=model_name)

    return modelEns


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


def prepare_data(df, config, data_dir, save=False, save_path="",
                 seed=None, debug=False):
    np.random.seed(seed)
    X = np.empty(shape=(df.shape[0],
                        config.dim[0], config.dim[1], config.dim[2]))
    input_length = config.audio_length
    for i, fname in enumerate(df.fname):
        if debug:
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

        # mfcc = librosa.feature.mfcc(data, sr=config.sampling_rate,
        #                             n_mfcc=config.n_mfcc)

        mfcc = librosa.feature.melspectrogram(data, sr=config.sampling_rate,
                                              n_mels=config.n_mfcc)
        mfcc = np.log(mfcc + 1e-10)

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
                use_mfcc=True, use_delta=True, use_Ddelta=True, n_mfcc=128,
                learning_rate=1e-3, batch_size=32, max_epochs=500,
                model_name="logMels_d_dd_ten")
# pk.dump(config, open("config_"+config.model_name+".pk", 'wb'))

# np.random.seed(config.random_seed)

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/sample_submission.csv")

LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
# train.set_index("fname", inplace=True)
# test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])

MODEL_FOLDER = "./model/" + config.model_name + "/"
if not os.path.exists(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)

models_list = []
skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)
for i, (train_split, val_split) in enumerate(skf):
    print("#"*50)
    print("Fold: ", i)

    PATH_train = "./model/train_logMels128+d+dd_fold{}.npy".format(i)
    PATH_test = "./model/test_logMels128+d+dd_fold{}.npy".format(i)
    X_train = prepare_data(train, config, './data/audio_train/',
                           save=False, save_path=PATH_train, seed=i)
    X_test = prepare_data(test, config, './data/audio_test/',
                          save=False, save_path=PATH_test, seed=i)
    # X_train = np.load(PATH_train)
    # X_test = np.load(PATH_test)
    y_train = to_categorical(train.label_idx, num_classes=config.n_classes)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std

    X, y = X_train[train_split], y_train[train_split]
    X_val, y_val = X_train[val_split], y_train[val_split]

    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(np.argmax(y, axis=1)),
        np.argmax(y, axis=1).reshape((-1,)))

    checkpoint = ModelCheckpoint(
        MODEL_FOLDER + config.model_name+'_best_{}.h5'.format(i),
        monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False)
    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=12)
    rLR = ReduceLROnPlateau(monitor='val_loss', factor=0.7,
                            patience=6, verbose=1, mode='min')
    callbacks_list = [checkpoint, early, rLR]

    model = get_2d_conv_model(config)
    print(model.summary())
    history = model.fit(X, y,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list,
                        batch_size=config.batch_size,
                        epochs=config.max_epochs,
                        class_weight=class_weights)
    model = load_model(MODEL_FOLDER+config.model_name+"_best_{}.h5".format(i))

    pred = model.predict(X_test)
    np.save("./model/raw_prediction_"+config.model_name+"fold_{}.npy".format(i), pred)

#     models_list.append(model)
#
# ens_model = ensembleModels(
#     models_list, Input(shape=(config.dim[0], config.dim[1], config.dim[2],)))
# ens_model.save(config.model_name+"_10fold.h5")
