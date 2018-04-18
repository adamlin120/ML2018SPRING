import numpy as np
import pandas as pd
import csv
from keras import layers
from keras.models import Model
from keras.utils import np_utils


def load_train_data(path_data, validation_split, shuffle=True, seed=None):
    if(seed is None):
        np.random.seed(seed=1)
    else:
        np.random.seed(seed)

    df = pd.read_csv(path_data)

    n_obs = len(df)
    n_train = n_obs - int(n_obs * validation_split)

    idx = np.arange(n_obs)
    if(shuffle):
        np.random.shuffle(idx)
    idx_train = idx[:n_train]
    idx_test = idx[n_train:]

    df_train = df.iloc[idx_train]
    df_val = df.iloc[idx_test]

    X = np.array([list(map(float, df_train["feature"].iloc[i].split(' '))) for i in range(len(df_train))]).reshape(-1, 48, 48, 1).astype('float32') / 255.0
    X_val = np.array([list(map(float, df_val["feature"].iloc[i].split(' '))) for i in range(len(df_val))]).reshape(-1, 48, 48, 1).astype('float32') / 255.0

    y = np_utils.to_categorical(np.array(df_train["label"]), 7)
    y_val = np_utils.to_categorical(np.array(df_val["label"]), 7)

    return (X, y), (X_val, y_val)


def load_test_data(path_data):
    test_df = pd.read_csv(path_data)
    X_test = np.array([list(map(float, test_df["feature"][i].split())) for i in range(len(test_df))]).reshape(-1, 48, 48, 1).astype('float32') / 255.0

    return X_test


def submission(Y_pred, name):
    Y_pred = Y_pred.astype('int')
    file = open(name, 'w')
    csvCursor = csv.writer(file)

    # write header to csv file
    csvHeader = ['id', 'label']
    csvCursor.writerow(csvHeader)

    ans = []
    for idx, value in enumerate(Y_pred, 0):
        ans.append([str(idx), value])

    csvCursor.writerows(ans)
    file.close()


def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels = [model(model_input) for model in models]
    # averaging outputs
    yAvg = layers.average(yModels)
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')

    return modelEns


def pred2pred_class(pred):
    pred_class = np.array([np.argmax(pred[i]) for i in range(len(pred))])
    return pred_class
