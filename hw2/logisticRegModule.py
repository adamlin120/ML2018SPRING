import numpy as np
import random
import csv

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


class logistic_reg:
    def __init__(self, learning_rate=0.01, alpha=0):
        self.learning_rate = float(learning_rate)
        self.alpha = alpha


    def logistic_func(self, s):
        ret = np.exp(s)/(1+np.exp(s))
        ret = 1/(1+np.exp(-s))
        return ret

    def err_func(self, X, Y):
        y_hat = X.dot(self.w_log)
        y_hat[y_hat > 0] = 1
        y_hat[y_hat <= 0] = 0
        e_in = float(np.sum(y_hat==Y)) / np.shape(Y)[0]

        return e_in

    def fit(self, X, y, bias=True, epochs=1e5, split_ratio=0, verbose=True):
        self.err_list = []
        self.epochs = int(epochs)
        self.verbose = True

        idx_train = np.random.randint(0, len(X), int((1-split_ratio)*len(X)))
        idx_test = np.delete(np.arange(len(X)), idx_train)
        X_train = X[idx_train]
        y_train = y[idx_train]
        X_test = X[idx_test]
        y_test = y[idx_test]

        N, d = np.shape(X_train)
        self.w_log = np.zeros((d, 1))
        self.w_log = np.random.rand(d).reshape((d, 1))
        sum_sq_gra = 1e-8
        for i in range(self.epochs):
            idx = np.random.randint(0, X_train.shape[0], 150)
            X_batch = X_train[idx]
            y_batch = y_train[idx]
            
            w_regularization = np.array(self.w_log)
            if(bias):
                w_regularization[0] = 0

            gradient = (1/len(X_batch)) * -X_batch.T.dot(y_batch-self.logistic_func(X_batch.dot(self.w_log))) + self.alpha * w_regularization

            sum_sq_gra = sum_sq_gra + gradient**2
            # sum_sq_gra = 1

            self.w_log += -self.learning_rate * gradient / (np.sqrt(sum_sq_gra))
            # self.w_log += -self.learning_rate * gradient

            self.err_list.append(self.err_func(X_train, y_train))

            if(self.verbose and not i%100):
                if(split_ratio!=0):
                    print('# Iteration: {}\tTrain Accu = {}\tTest Accu = {}'.format(i, self.err_list[i], self.err_func(X_test, y_test)))
                else:
                    print('# Iteration: {}\tTrain Accu = {}'.format(i, self.err_list[i]))
                          

        return self.w_log, self.err_list

    def predict(self, X_test):
        pred = X_test.dot(self.w_log)
        pred[pred>0] = 1
        pred[pred<=0] = 0
        return pred.astype(int).reshape((len(pred)))

    def evaluate(self, X, y, verbose=True):
        pred = X.dot(self.w_log)
        pred[pred>0] = 1
        pred[pred<=0] = 0

        accu = float(np.sum(pred==y))/len(y)

        if(verbose):
            print('Accuracy: {}'.format(accu))

        return accu


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


def perceptron(X, Y, lr=1e-10, updates = 1e5, pocket = True) :
    def test(X, y, w) :
       pred = X.dot(w)
       accu = np.sum(y.reshape((len(y)))==pred) / len(y)
       return accu

    updates = int(updates)
    col = len(X[0])
    n = len(X)
    w = np.zeros(col)
    wg = w
    error = test(X, Y, w);

    idx = np.arange(n)
    for k in range(updates) :
        np.random.shuffle(idx)
        for i in idx :
            if np.sign(X[i].dot(w)) != Y[i] :
                w = w + lr * (Y[i] * X[i])
                e = test(X, Y, w)
                if e < error :
                    error = e
                    wg = w
                break
        print('# itration: {}\t Accu: {}'.format(k, error))


    if(pocket):
        return wg
    return w

def evaluate(pred, y, verbose=True):
    accu = np.sum(y.reshape((len(y)))==pred) / len(y)

    if(verbose):
        print('Accuracy: {}'.format(accu))

    return accu

def k_fold_cv(X, y, model, n_fold=3, loss='accu', fit_arg='None', shuffle=True):
    def rmse(y, y_prd):
        residual = y - y_prd
        return (np.sum(residual**2)/len(y))**(1/2)

    def accu(y, y_prd):
        err = float(np.sum(y!=y_prd))
        return err


    idx = np.arange(len(X))
    if(shuffle):
        np.random.seed(int(X.size*1e5*np.random.random()) % (2**32 - 1))
        np.random.shuffle(idx)
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

    err = list()
    for train_idx, test_idx in idx_split:
        model.fit(X[train_idx], y[train_idx], exec(fit_arg))
        err.append(model.evaluate(X[test_idx], y[test_idx]))

    return err


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
