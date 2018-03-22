#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:32:47 2018

@author: adamlin
"""
import csv

import numpy as np
import pandas as pd


def add_bias(X):
    X_bias = np.append(np.ones((X.shape[0], 1)), X, axis=1)

    return X_bias


def standardize(X_un, bias):
    if(bias):
        X_un = np.delete(X_un, 0, 1)
    colMean = np.mean(X_un, axis=0)
    colStd = np.var(X_un, axis=0)**(1/2)
    X_std = (X_un-colMean)/colStd
    if(bias):
        X_std = np.append(np.ones((X_std.shape[0], 1)), X_std, axis=1)
    return X_std


def load_train(path="./dataset/train.csv", transform=False):
    # loading and cleaning data
    df_train = pd.read_csv(path, delimiter=',', encoding='big5')
    # remove first THREE columns
    df_train = df_train.drop(df_train.columns[[0, 1, 2]], axis=1)
    # replace 'NR' to 0
    df_train = df_train.replace(to_replace='NR', value=0)
    # change to numeric
    df_train = df_train.astype('float64')

    df_train = np.hstack(np.split(df_train, df_train.shape[0]//18, 0))

    df_train[9, 1229:1239] = 65
    df_train[9, 1466+9] = 5

    n_fea_day = 18
    n_feature = n_fea_day * 9  # 18measure per day, 9 days
    row_delete = ()
    if(transform):
        # transform 4 variable: wind speed and wind dicrection
        wd_dr = df_train[14]  # wind direc in hour
        wd_sp = df_train[17]  # wind speed in hout
        wd_min_dr = df_train[15]  # wind direc in 10min
        wd_min_sp = df_train[16]  # wind speed in 10min
        # transfrom to x and y two dimesion
        wd_x = wd_sp * np.cos(wd_dr*np.pi/180)
        wd_y = wd_sp * np.sin(wd_dr*np.pi/180)
        wd_min_x = wd_min_sp * np.cos(wd_min_dr*np.pi/180)
        wd_min_y = wd_min_sp * np.sin(wd_min_dr*np.pi/180)
        # replace
        df_train[14] = wd_x
        df_train[15] = wd_y
        df_train[16] = wd_min_x
        df_train[17] = wd_min_y

        # row_delete = (0, 1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17)
        df_train = np.delete(df_train, row_delete, axis=0)

        n_fea_day = 18-len(row_delete)
        n_feature = n_fea_day*9

    # extract feature X_train, _train
    X_train = np.empty((n_feature, 1), float)
    for i in range(5760):
        if (i+9 >= 5760):
            break
        X_train = np.hstack((X_train,
                            df_train[:, i:(9+i)].T.reshape((n_feature, 1))))
    X_train = np.delete(X_train, 0, 1)  # delete the first empty col
    X_train = X_train.T

    num_fea_before_pm25 = sum(f < 9 for f in row_delete)
    Y_train = df_train[9-num_fea_before_pm25, 9:]

    return X_train, Y_train


def load_test(path='./dataset/test.csv', transform=False):
    df_test = pd.read_csv('./dataset/test.csv', delimiter=',', header=None)
    # remove first TWO columns
    df_test = df_test.drop(df_test.columns[[0, 1]], axis=1)
    # replace 'NR' to 0
    df_test = df_test.replace(to_replace='NR', value=0)
    # change to numeric
    df_test = df_test.astype('float64')

    df_test = np.hstack(np.split(df_test, df_test.shape[0]//18, 0))

    n_fea_day = 18
    n_feature = n_fea_day * 9
    row_delete = ()
    if(transform):
        # transform 4 variable: wind speed and wind dicrection
        wd_dr = df_test[14]  # wind direc in hour
        wd_sp = df_test[17]  # wind speed in hout
        wd_min_dr = df_test[15]  # wind direc in 10min
        wd_min_sp = df_test[16]  # wind speed in 10min
        # transfrom to x and y two dimesion
        wd_x = wd_sp * np.cos(wd_dr*np.pi/180)
        wd_y = wd_sp * np.sin(wd_dr*np.pi/180)
        wd_min_x = wd_min_sp * np.cos(wd_min_dr*np.pi/180)
        wd_min_y = wd_min_sp * np.sin(wd_min_dr*np.pi/180)
        # replace
        df_test[14] = wd_x
        df_test[15] = wd_y
        df_test[16] = wd_min_x
        df_test[17] = wd_min_y

        # row_delete = (0, 1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17)
        df_test = np.delete(df_test, row_delete, axis=0)

        n_fea_day = 18-len(row_delete)
        n_feature = n_fea_day*9

    X_test = np.reshape(df_test, (n_feature, 260), order='F').T

    return X_test


def k_fold_cv(X, y, model, n_fold=3, shuffle=True):
    def rmse(y, y_prd):
        residual = y - y_prd
        return (np.sum(residual**2)/len(y))**(1/2)

    idx = np.arange(len(X))
    if(shuffle):
        np.random.seed(int(X.size*1e5*np.random.random()) % (2**32 - 1))
        np.random.shuffle(idx)

    batch_size = len(X)//n_fold

    idx_split = []
    for n in range(n_fold):

        test_start = n*batch_size
        test_end = test_start + batch_size

        if(n == n_fold-1):
            test_end = len(X)

        test_idx = idx[test_start:test_end]
        train_idx = np.delete(idx, test_idx)

        idx_split.append([train_idx,  test_idx])

    err = 0
    for train_idx, test_idx in idx_split:
        model.fit(X[train_idx], y[train_idx])
        err = err + (rmse(model.predict(X[test_idx]),
                          y[test_idx])*(len(test_idx)/len(idx)))

    return err


def predict(model, X_test, name):
    Y_pred = model.predict(X_test)

    file = open(name, 'w')
    csvCursor = csv.writer(file)

    # write header to csv file
    csvHeader = ['id', 'value']
    csvCursor.writerow(csvHeader)

    ans = []
    i = 0
    for value in Y_pred:
        ans.append(['id_'+str(i), value])
        i = i + 1

    csvCursor.writerows(ans)
    file.close()


class linear_regression_adgrad():
    def __init__(self, lr=0.1, lda=0, max_itr=1e4):
        self.lr = lr
        self.max_itr = int(max_itr)
        self.lda = lda  # for L2 ridge reg
        self.w = None

    def fit(self, X, y, gd=False, verbose=1):
        if(gd):
            self.w = np.zeros(X.shape[1])
            sum_sq_gra = 0

            for it in range(self.max_itr):
                residual = X.dot(self.w) - y
                gradient = X.T.dot(residual) + self.lda * self.w
                sum_sq_gra = sum_sq_gra + gradient**2

                self.w = self.w - self.lr * gradient/(np.sqrt(sum_sq_gra))

                if(verbose and not it % 10):
                    rmse = (np.sum(residual**2)/X.shape[0])**(1/2)
                    print('Iteration: {}\tRMSE Cost = {}'.format(it, rmse))

        else:
            w_hat = np.linalg.inv(
                X.T.dot(X) +
                np.eye(np.shape(X)[1]).dot(self.lda)).dot(X.T).dot(y)

            self.w_closed = w_hat
            self.w = w_hat

        # self.evaluate(X, y, True)

    def evaluate(self, X, y, verbose=True):
        residual = X.dot(self.w) - y

        GD = 'Gradient lr={} lambda={} max_iteration={}'.format(self.lr,
                                                                self.lda,
                                                                self.max_itr)

        rmse = (np.sum(residual**2)/X.shape[0])**(1/2)

        if(verbose):
            print('{}\nRMSE Cost = {}'.format(GD, rmse))

        return rmse

    def predict(self, X):
        pred = X.dot(self.w)
        return pred

    def set_w(self, w_new):
        self.w = w_new

    def save_md(self, path):
        np.save(path, self.w)

    def load_md(self, path):
        self.set_w(np.load(path))

    def coef_(self):
        return self.w
