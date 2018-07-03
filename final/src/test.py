# -*- coding: utf-8 -*-
# Change this to True to replicate the result

import numpy as np
import pandas as pd
import pickle as pk
from keras.models import load_model
from keras.models import Model
from keras.layers import average
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Convolution1D
from keras.layers import MaxPool1D
from keras.layers import Dropout
from keras.layers import GlobalMaxPool1D
from keras.activations import relu
from keras.activations import softmax
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from utils import ensembleModels


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
                        1 + int(np.floor(self.audio_length/512)),
                        self.n_channel)
        else:
            self.dim = (self.audio_length, 1)


def get_1d_conv_model():
    nclass = 41
    input_length = 32000

    inp = Input(shape=(input_length, 1))
    x = Convolution1D(16, 9, activation=relu, padding="valid")(inp)
    x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
    x = MaxPool1D(16)(x)
    x = Dropout(rate=0.1)(x)

    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)

    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)

    x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=0.2)(x)

    x = Dense(64, activation=relu)(x)
    x = Dense(1028, activation=relu)(x)
    out = Dense(nclass, activation=softmax)(x)

    model = Model(inputs=inp, outputs=out)
    opt = Adam(0.001)

    model.compile(optimizer=opt,
                  loss=categorical_crossentropy,
                  metrics=['acc'])
    return model


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/sample_submission.csv")

LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
train["label_idx"] = train.label.apply(lambda x: label_idx[x])


mfcc_d_dd_config = pk.load(open("./model/config_MFCC_d_dd.pk", "rb"))

X_train_mfcc60_d_dd = np.load("./model/train_mfcc60+d+dd.npy")
X_test_mfcc60_d_dd = np.load("./model/test_mfcc60+d+dd.npy")
mean_mfcc60_d_dd = np.mean(X_train_mfcc60_d_dd, axis=0)
std_mfcc60_d_dd = np.std(X_train_mfcc60_d_dd, axis=0)
X_test_mfcc60_d_dd = (X_test_mfcc60_d_dd - mean_mfcc60_d_dd) / std_mfcc60_d_dd

X_train_con1d = np.load("./model/train_1d.npy")
X_test_con1d = np.load("./model/test_1d.npy")

# mean_con1d = np.mean(X_train_con1d, axis=0)
# std_con1d = np.std(X_train_con1d, axis=0)
#
# X_train_con1d = (X_train_con1d - mean_con1d)/std_con1d
# X_test_con1d = (X_test_con1d - mean_con1d)/std_con1d

X_train_con1d = np.expand_dims(X_train_con1d, -1)
X_test_con1d = np.expand_dims(X_test_con1d, -1)


model_mfcc_d_dd = load_model("./model/MFCC_d_dd_10fold.h5")

model_con1d_list = [load_model("./model/best_{}.h5".format(int(i))) for i in range(10)]
model_con1d = ensembleModels(model_con1d_list, Input(shape=(32000, 1)))
# model_con1d.save('model_con1d_10fold.h5')

# collect outputs of models in a list
input_mfcc_d_dd = Input(shape=(mfcc_d_dd_config.dim[0],
                               mfcc_d_dd_config.dim[1],
                               mfcc_d_dd_config.dim[2],))
input_con1d = Input(shape=(32000, 1))

yModels = []
yModels.append(model_mfcc_d_dd(input_mfcc_d_dd))
yModels.append(model_con1d(input_con1d))
yAvg = average(yModels)

modelEns = Model(inputs=[input_mfcc_d_dd, input_con1d],
                 outputs=yAvg,
                 name='MODEL ensemble')
print(modelEns.summary())
# modelEns.save('model_enseble_con1d_mfcc60_d_dd.h5')
# modelEns = load_model('model_enseble_con1d_mfcc60_d_dd.h5')
pred_1d_2d = modelEns.predict([X_test_mfcc60_d_dd, X_test_con1d], verbose=1)

num_logMels = 5
pred_logMels_list = [np.load('./model/raw_prediction_logMels_d_ddfold_{}.npy'.format(i)) for i in range(num_logMels)]
tmp = pred_logMels_list[0]
for i in range(num_logMels-1):
    tmp = tmp * pred_logMels_list[i+1]
pred_logMels = np.float_power(tmp, (1/num_logMels))


pred = np.float_power(pred_logMels * pred_1d_2d, 1/2)

# Make a submission file
top_3 = np.array(LABELS)[np.argsort(-pred, axis=1)[:, :3]]
predicted_labels = [' '.join(list(x)) for x in top_3]
test = pd.read_csv('./data/sample_submission.csv')
test['label'] = predicted_labels
test[['fname', 'label']].to_csv("./prediction_final.csv", index=False)
