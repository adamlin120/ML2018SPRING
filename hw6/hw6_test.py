import csv
import argparse
from keras.models import load_model
import ut as u

parser = argparse.ArgumentParser(description='TESTING--Matrix Factorization')
parser.add_argument('--test')
parser.add_argument('--prediction')
parser.add_argument('--movies')
parser.add_argument('--users')
args = parser.parse_args()


model_path = './model.h5'
sub_path = args.prediction
movie_path = args.movies
user_path = args.users
test_path = args.test

n_item, movieid_mapping = u.movie_input(movie_path)
n_user, userid_mapping = u.user_input(user_path)

test_id, test_movie, test_user = u.test_input(test_path, movieid_mapping, userid_mapping)

model = load_model(model_path)
Y_pred = model.predict([test_user, test_movie], verbose=1)

with open(sub_path, 'w') as f:
    csvw = csv.writer(f)
    csvw.writerow(["TestDataID", "Rating"])
    for i in range(0, len(test_id)):
        csvw.writerow([test_id[i], max(1.0, min(Y_pred[i][0], 5.0))])
