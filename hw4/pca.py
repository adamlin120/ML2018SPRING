from skimage import io
from skimage.viewer import ImageViewer
import numpy as np
import sys


def convert2plot(im):
    im = im - np.min(im)
    im = im / np.max(im)
    im = (im*255).astype(np.uint8).reshape(600, 600, 3)

    return im


def reconstruct(y, mean_X, U, n_component=4):
    y = y - mean_X
    weight = y.dot(U[:, :n_component])

    ret = weight.dot(U[:, :n_component].T) + mean_X

    return ret


DIR_PATH = sys.argv[1]+'/'
IMAGE_FILE = sys.argv[2]
    
#DIR_PATH = '../dataset/Aberdeen/'
#IMAGE_FILE = '10.jpg'

num_image = 415
X = np.empty((num_image, 600, 600, 3))
for i in range(num_image):
    X[i] = io.imread(DIR_PATH+str(i)+'.jpg').astype(np.float64)

X = X.reshape((num_image, -1)) / 255.
mean_X = np.mean(X, axis=0)
X_std = X - mean_X
U, sigma, V = np.linalg.svd(X_std.T, full_matrices=False)

# testing
ID = int(IMAGE_FILE.split('.')[0])
y = X[ID]
recon = reconstruct(y, mean_X, U, n_component=4)

image = convert2plot(recon)
viewer = ImageViewer(image)
viewer.save_to_file('./reconstruction.png')
