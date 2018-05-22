import csv
import argparse
import pickle
import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


def submission(ID, pred, output_name):
    print('=========writing result=========')
    file = open(output_name, 'w')
    csvCursor = csv.writer(file)

    # write header to csv file
    csvHeader = ['id', 'label']
    csvCursor.writerow(csvHeader)

    ans = []
    for idx in range(len(ID)):
        ans.append([int(idx), int(pred[idx])])

    csvCursor.writerows(ans)
    file.close()


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

# Parse arguments
parser = argparse.ArgumentParser(description='TESTING--Sentiment classify')
parser.add_argument('-md', '--model')
parser.add_argument('-sub', '--submission', default='submission.csv')
parser.add_argument('--test_data', default='./dataset/testing_data.txt')
parser.add_argument('--tokenizer', default='./tokenizer.pk')
parser.add_argument('--maxlen', default=39, type=int)
args = parser.parse_args()

test_data_path = args.test_data
tokenizer_save_path = args.tokenizer
submission_path = args.submission
model_path = args.model
maxlen = args.maxlen

# Load data
(X_test, ID) = read_data(test_data_path, test=True)

# Load tokenizer
tokenizer = pickle.load(open(tokenizer_save_path, 'rb'))

# Prepare Data
seq = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(seq, maxlen=maxlen)

# Load RNN model
model = load_model(model_path)
print(model.summary())

# Predict
predict = model.predict(X_test, batch_size=1024, verbose=1)
predict = np.argmax(predict, axis=-1)

# Create prediction csv file
submission(ID, predict, submission_path)
