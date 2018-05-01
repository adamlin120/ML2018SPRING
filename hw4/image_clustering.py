#!/usr/bin/env python
import sys
import csv
import pickle
import numpy as np
from keras.models import Input
from keras.models import Model
from keras.layers import Dense
from keras.regularizers import l1
from keras.optimizers import Adam
from sklearn.cluster import KMeans


def submission(Y_pred, name):
    file = open(name, 'w')
    csvCursor = csv.writer(file)

    # write header to csv file
    csvHeader = ['ID', 'Ans']
    csvCursor.writerow(csvHeader)

    ans = []
    for idx, value in enumerate(Y_pred, 0):
        ans.append([str(idx), value])

    csvCursor.writerows(ans)
    file.close()

# loading data and normalization
X = np.load(sys.argv[1]).astype(np.float64) / 255.0

num_train = int(0.9 * len(X))

x_train = X[:num_train]
x_val = X[num_train:]

input_dim = 28*28
encoding_dim = 32
# build model
input_img = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu', kernel_regularizer=l1(1e-8))(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

encoded_input = Input(shape=(encoding_dim,))
decoded = Dense(64, activation='relu')(encoded_input)
decoded = Dense(128, activation='relu', kernel_regularizer=l1(1e-8))(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# build encoder
encoder = Model(input=input_img, output=encoded)
# bulid decoder
decoder = Model(encoded_input, decoded)

# build autoencoder
adam = Adam(lr=1e-3)
autoencoder = Model(input=input_img, output=decoder(encoder(input_img)))
autoencoder.compile(optimizer=adam, loss='mse')
autoencoder.summary()

EPOCHS = 100
BATCH_SIZE = 256
autoencoder.fit(x=x_train, y=x_train, validation_data=(x_val, x_val),
                shuffle=True, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

# after training, use encoder to encode image, and feed it into Kmeans
encoded_imgs = encoder.predict(X)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)

# saving model
encoder.save('./encoder.h5')
s = pickle.dump(kmeans, open('kmeans.sav', 'wb'))
