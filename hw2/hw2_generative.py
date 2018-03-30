import numpy as np
import pandas as pd
import csv
import sys

def add_bias(X):
    X_bias = np.append(np.ones((X.shape[0], 1)), X, axis=1)

    return X_bias

def standardize(X_un, bias):
    if(bias):
        X_un = np.delete(X_un, 0, 1)
    colMean = np.mean(X_un, axis=0)
    colStd = np.var(X_un, axis=0)**(1/2)
    #avoid devided by zero
    colStd[np.abs(colStd)<=1e-10] = 1
    X_std = (X_un-colMean)/colStd
    if(bias):
        X_std = np.append(np.ones((X_std.shape[0], 1)), X_std, axis=1)
    return X_std

def normalize(X_un, bias):
    if(bias):
        X_un = np.delete(X_un, 0, 1)
    colMean = np.mean(X_un, axis=0)
    colMax = np.max(X_un, axis=0)
    colMin = np.min(X_un, axis=0)
    colRange = colMax-colMin
    #avoid devided by zero
    colRange[np.abs(colRange)<=1e-10] = 1
    X_nor = (X_un-colMean)/(colRange)
    if(bias):
        X_nor = np.append(np.ones((X_nor.shape[0], 1)), X_nor, axis=1)
    return X_nor


class gaussianGen:
    def __init__(self, X, y):
        self.num_class = 2
        self.mean = []
        self.mean.append(np.mean(X[(y==0).reshape((len(y)))], axis=0))
        self.mean.append(np.mean(X[(y==1).reshape((len(y)))], axis=0))
        self.all_std = np.cov(X.T)
        self.D = len(self.mean[0])
        self.demonimator = (np.power(2*np.pi, self.D/2)*np.power(abs(np.linalg.det(self.all_std)), 1/2))
        self.std_inv = np.linalg.pinv(self.all_std)

        self.p1 = np.sum((y==0).reshape((len(y)))) / float(len(y))
        self.p2 = np.sum((y==1).reshape((len(y)))) / float(len(y))

    def PDF(self, x, cls):
        class_mean = self.mean[cls]
        diff = x-class_mean

        numerator = np.power(np.e, diff.T.dot(self.std_inv).dot(diff)/-2)

        ans = numerator/self.demonimator

        return ans

    def predict_one(self, x_test):
        p_0 = self.PDF(x_test, 0) * self.p1
        p_1 = self.PDF(x_test, 1) * self.p2
        if(p_0>p_1):
            return int(0)
        else:
            return int(1)

    def predict(self, X_test):
        pred = np.empty((len(X_test)))
        for idx, x in enumerate(X_test):
            pred[idx] = self.predict_one(x)

        return pred.astype(int)

    def evaluate(self, X, y, verbose=True):
        pred = self.predict(X)
        accu = np.sum(y.reshape((len(y)))==pred) / len(X)

        if(verbose):
            print('Accuracy: {}'.format(accu))

        return accu

    def pred_eval(self, X, y, verbose=True):
        pred = self.predict(X)
        accu = np.sum(y.reshape((len(y)))==pred) / len(X)

        if(verbose):
            print('Accuracy: {}'.format(accu))

        return pred, accu



def evaluate(pred, y, verbose=True):
    accu = np.sum(y.reshape((len(y)))==pred) / len(y)

    if(verbose):
        print('Accuracy: {}'.format(accu))

    return accu


def submission(Y_pred, name):
    file = open(name, 'w')
    csvCursor = csv.writer(file)

    # write header to csv file
    csvHeader = ['id', 'label']
    csvCursor.writerow(csvHeader)

    ans = []
    for idx, value in enumerate(Y_pred, 1):
        ans.append([str(idx), value])

    csvCursor.writerows(ans)
    file.close()


#df = pd.read_csv('./dataset/train.csv')
df = pd.read_csv(sys.argv[1])
df = df.drop(['fnlwgt', 'education'], axis=1)
df = pd.get_dummies(df)
y = df['income_ >50K'].values.reshape((-1,1)).astype('float64')
X = df.drop(['income_ >50K', 'income_ <=50K'], axis=1).values.astype('float64')
# normalize
X = normalize(X, False)


#df_test = pd.read_csv('./dataset/test.csv')
df_test = pd.read_csv(sys.argv[2])
df_test = df_test.drop(['fnlwgt', 'education'], axis=1)
df_test = pd.get_dummies(df_test)
df_test.insert(76, column = 'native_country_ Holand-Netherlands',
               value=np.zeros(len(df_test)))
X_test = df_test.values.astype('float64')
# normalize
X_test = normalize(X_test, False)


model = gaussianGen(X, y)
pred = model.predict(X_test).astype(int)

# submission(pred, 'generative_noBias_std.csv')
submission(pred, sys.argv[6])
