import logisticRegModule as logReg
import numpy as np
import pandas as pd
import sys
from sklearn import svm


#df = pd.read_csv('../dataset/train.csv')
df = pd.read_csv(sys.argv[1])
df = df.drop(['fnlwgt', 'education'], axis=1)
df = pd.get_dummies(df)
y = df['income_ >50K'].values.reshape((-1,1)).astype('float64')
y = y.reshape((len(y)))
X = df.drop(['income_ >50K', 'income_ <=50K'], axis=1).values.astype('float64')
# normalize
X = logReg.normalize(X, False)
# add X_bias
X = logReg.add_bias(X)

#df_test = pd.read_csv('../dataset/test.csv')
df_test = pd.read_csv(sys.argv[2])
df_test = df_test.drop(['fnlwgt','education'], axis=1)
df_test = pd.get_dummies(df_test)

df_test.insert(76, column = 'native_country_ Holand-Netherlands',
               value=np.zeros(len(df_test)))
X_test = df_test.values.astype('float64')
# normalize
X_test = logReg.normalize(X_test, False)
# add X_bias
X_test = logReg.add_bias(X_test)

# features transorm: conti variables square terms
var_sq = (1,5)

X = np.c_[X, np.power(X[:, var_sq], 2)]
X_test = np.c_[X_test, np.power(X_test[:, var_sq], 2)]

idx = np.random.randint(0, len(X), int(0.8*len(X)))
X_tune = X[idx, ]
y_tune = y[idx]
X_val = X[np.delete(np.arange(len(X)), idx), ]
y_val = y[np.delete(np.arange(len(X)), idx)]

#target = pd.read_csv('~/Desktop/svm_sqTerm.csv')['label'].values

#linear
linearSVM = svm.LinearSVC(penalty='l2', loss='squared_hinge',
                          fit_intercept=True, intercept_scaling=1,
                          verbose=True, max_iter=int(1e5), dual=False, C=1)
linearSVM.fit(X, y)

pred = linearSVM.predict(X_test).astype(int)

logReg.submission(pred, sys.argv[6])

# =============================================================================
# C_s = np.logspace(0, 2, 20)
# scores = list()
# scores_std = list()
# for C in C_s:
#     linearSVM.C = C
#     this_scores = cross_val_score(linearSVM, X, y, n_jobs=1, cv=8)
#     scores.append(np.mean(this_scores))
#     scores_std.append(np.std(this_scores))
# print(scores)
# print(C_s)
# # Do the plotting
# import matplotlib.pyplot as plt
# plt.figure(1, figsize=(11, 11))
# plt.clf()
# plt.semilogx(C_s, scores)
# plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
# plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
# locs, labels = plt.yticks()
# plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
# plt.ylabel('CV score')
# plt.xlabel('Parameter C')
# plt.ylim(.8375, .861)
# plt.show()
# =============================================================================
