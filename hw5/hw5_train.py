import os
import pickle as pk
import multiprocessing
import numpy as np
import argparse


import keras.preprocessing.text as Text_prep
from keras import layers
from keras.models import load_model, Model, Input
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
import gensim

from hw5_models import BLSTM


def read_data(data_path, train=False, semi=False, test=False):
    with open(data_path, 'r', encoding='utf8') as f:
        if train:
            X_train, Y_train = [], []
            for line in f:
                line = line.strip().split(' +++$+++ ')
                X_train.append(line[1])
                Y_train.append(int(line[0]))
            return (X_train, Y_train)
        elif semi:
            X_semi = []
            for line in f:
                X_semi.append(line.strip())
            return (X_semi)
        elif test:
            X_test, ID = [], []
            for line in f:
                line = line.strip().split(',', 1)
                if line[0] != 'id':
                    X_test.append(line[1])
                    ID.append(int(line[0]))
            return (X_test, ID)
        else:
            raise Exception('NOT SPECIFY READING DATA CATEGORY')


def ensembleModels(models, model_input, version_name):
    # collect outputs of models in a list
    yModels = [model(model_input) for model in models]
    # averaging outputs
    yAvg = layers.average(yModels)
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name=version_name)

    return modelEns


parser = argparse.ArgumentParser(description='TRAINING--Sentiment classify')
parser.add_argument('--test_data', default='./dataset/testing_data.txt')
args = parser.parse_args()

model_type = 'BLSTM'
model_name = 'BLSTM_big_ens_iter25'
save_dir = os.path.join('./model/', model_name)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_save_path = os.path.join(save_dir, model_name+'.h5')
semi_model_save_path = os.path.join(save_dir, model_name+'_semi.h5')
tokenizer_save_path = os.path.join(save_dir, 'tokenizer.pk')
w2v_save_path = os.path.join(save_dir, 'w2v_model')
# Load data
train_data_path = args.test_data

(X_train, Y_train) = read_data(train_data_path, train=True)

# Create tokenizer
no_punc = False
if no_punc:
    flt = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'
else:
    flt = '\t\n'
tokenizer = Tokenizer(filters=flt)

tokenizer.fit_on_texts(X_train)
seq = tokenizer.texts_to_sequences(X_train)
max_len = 39
pk.dump(tokenizer, open(tokenizer_save_path, 'wb'))

# =============================================================================
# bow = True
# if bow:
#     from sklearn.feature_extraction.text import CountVectorizer
#     cv = CountVectorizer(dtype=np.int8)
#     X_bow_train = cv.fit_transform(X_train).toarray()
# =============================================================================
for i in range(len(X_train)):
    X_train[i] = Text_prep.text_to_word_sequence(X_train[i], filters=flt)

# train word2vec model
embed_dim = 250
w2v_model = gensim.models.Word2Vec(X_train, iter=25,
                                   size=embed_dim, min_count=10,
                                   workers=multiprocessing.cpu_count())
w2v_model.save(w2v_save_path)

# create embedding matrix
num_word = len(w2v_model.wv.vocab)
emb_matrix = np.zeros((num_word+1, embed_dim))
out = 0
for word, i in tokenizer.word_index.items():
    try:
        vec = w2v_model.wv[word]
        emb_matrix[i] = vec
    except:
        out += 1

# prepare fitting data
X_train = pad_sequences(seq, maxlen=max_len)
Y_train = to_categorical(Y_train)


# =============================================================================
# if bow:
#     X_fit = X_bow_train[2000:]
#     Y_fit = Y_train[2000:]
#
#     X_val = X_bow_train[:2000]
#     Y_val = Y_train[:2000]
#
#     bow_md = BOW_DNN()
#     print(bow_md.summary())
#     checkpoint = ModelCheckpoint('./model/bow.h5', monitor='val_acc', save_best_only=True, verbose=1)
#     hist = bow_md.fit(X_fit, Y_fit, validation_data=(X_val, Y_val),
#                      epochs=10, batch_size=1024, callbacks=[checkpoint])
# =============================================================================

# =============================================================================
# # basic training
# X_fit = X_train[2000:]
# Y_fit = Y_train[2000:]
#
# X_val = X_train[:2000]
# Y_val = Y_train[:2000]
#
# model = eval(model_type)(num_word, embed_dim, max_len, emb_matrix)
# print(model.summary())
# checkpoint = ModelCheckpoint(model_save_path, monitor='val_acc',
#                              save_best_only=True, verbose=1)
# earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
# hist = model.fit(X_fit, Y_fit, validation_data=(X_val, Y_val),
#                  epochs=30, batch_size=768,
#                  callbacks=[checkpoint, earlystopping])
# model = load_model(model_save_path)
# =============================================================================

# ensemble training -- 10-fold
val_idx = np.arange(0, len(X_train))
np.random.shuffle(val_idx)
val_idx = list(val_idx.reshape((10, -1)))
train_idx = [np.delete(np.arange(0, len(X_train)), obj) for obj in val_idx]
for fold in range(10):
    print('\nENSEMBLE TRAINING: Iteration: {}'.format(fold+1))
    print('initial model...')
    model = eval(model_type)(num_word, embed_dim, max_len, emb_matrix)
    print(model.summary())
    print('spliting data...')
    X_fit = X_train[train_idx[fold]]
    Y_fit = Y_train[train_idx[fold]]
    X_val = X_train[val_idx[fold]]
    Y_val = Y_train[val_idx[fold]]

    ens_model_save_path = os.path.join(save_dir,
                                       model_name+'_ens-'+str(fold+1)+'.h5')
    checkpoint = ModelCheckpoint(ens_model_save_path, monitor='val_acc',
                                 save_best_only=True, verbose=1)
    earlystopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
    hist = model.fit(X_fit, Y_fit, validation_data=(X_val, Y_val),
                     epochs=30, batch_size=128,
                     callbacks=[checkpoint, earlystopping])

ens_model_paths = [os.path.join(save_dir, model_name+'_ens-'+str(fold+1)+'.h5')
                   for fold in range(10)]
version_name = '10fold'
aggregated_model_save_path = os.path.join(
                             save_dir, model_name+'_ens-'+version_name+'.h5')
models = []

for i, path in enumerate(ens_model_paths):
    modelTemp = load_model(path)  # load model
    modelTemp.name = model_name+str(i+1)  # change name to be unique
    models.append(modelTemp)

ins = Input(shape=(max_len,))
modelEns = ensembleModels(models, ins, version_name)
modelEns.summary()
modelEns.save(aggregated_model_save_path)


# =============================================================================
# # self learning
# self_train = False
# if self_train:
#     (X_semi) = read_data(semi_data_path, semi=True)
#     X_semi = pad_sequences(tokenizer.texts_to_sequences(X_semi), maxlen=max_len)
#     checkpoint = ModelCheckpoint(semi_model_save_path, monitor='val_acc',
#                                  save_best_only=True, verbose=1)
#     for num_self_learn in range(5):
#         print('Self-learning Iteration: {}'.format(num_self_learn+1))
#         pred_semi = model.predict(X_semi, batch_size=1024, verbose=1)
#         pred_semi_class = np.argmax(pred_semi, axis=-1)
#         threshold = .05
#         semi_filter = np.logical_or(pred_semi[:, 0] > 1-threshold,
#                                     pred_semi[:, 0] < threshold).astype(np.int32)
#         (semi_idx, ) = np.where(semi_filter)
#         print('# data added: {}'.format(semi_idx.shape[0]))
#
#         X_add = X_semi[semi_idx, ]
#         Y_add = to_categorical(pred_semi_class[semi_idx, ])
#
#
#         X_aug = np.concatenate((X_fit, X_add), axis=0)
#         Y_aug = np.concatenate((Y_fit, Y_add), axis=0)
#
#         earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
#         model = eval(model_type)(num_word, embed_dim, max_len, emb_matrix)
#         hist = model.fit(X_aug, Y_aug, validation_data=(X_val, Y_val),
#                          epochs=30, batch_size=512,
#                          callbacks=[checkpoint, earlystopping])
#         model = load_model(semi_model_save_path)
# =============================================================================
