#!/usr/bin/env python
import csv
import sys
import numpy as np
import pandas as pd
from keras.models import load_model
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


encoder = load_model('./encoder.h5')

X = np.load(sys.argv[1]).astype(np.float64) / 255.0

# after training, use encoder to encode image, and feed it into Kmeans
encoded_imgs = encoder.predict(X)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)

# get test cases
f = pd.read_csv(sys.argv[2])
IDs = np.array(f['ID'])
idx1 = np.array(f['image1_index'])
idx2 = np.array(f['image2_index'])

# predict
prediction = np.empty(len(IDs)).astype(np.int)
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        prediction[idx] = 1  # two images in same cluster
    else:
        prediction[idx] = 0  # two images not in same cluster
submission(prediction, sys.argv[3])

print(np.sum(prediction)/len(prediction))
