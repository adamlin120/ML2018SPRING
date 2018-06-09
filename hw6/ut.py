import csv
import random
import numpy as np


def movie_input(movie_path):
    with open(movie_path, 'r', encoding="Windows-1252") as f:
        raw_data = [line.split("::")
                    for line in f.read().strip().split("\n")][1:]
    n_movie = len(raw_data)
    movie_id_mapping = dict((int(raw_data[i][0]), i) for i in range(n_movie))

    type_id_list = []
    for i in range(n_movie):
        type_str = raw_data[i][2]
        if type_str not in type_id_list:
            type_id_list.append(type_str)

    type_id_list = sorted(type_id_list)


    return n_movie, movie_id_mapping


def user_input(user_path):
    with open(user_path, 'r', encoding="Windows-1252") as f:
        raw_data = [line.split("::")
                    for line in (f.read().strip().replace("F", "1").
                                 replace("M", "0")).split("\n")][1:]
    n_user = len(raw_data)
    userid_mapping = dict((int(raw_data[i][0]), i)
                          for i in range(n_user))

    user_list = []
    for i in range(0, len(raw_data)):
        if raw_data[i][1:] not in user_list:
            user_list.append(raw_data[i][1:])

    return n_user, userid_mapping


def train_input(train_path, movieid_mapping, userid_mapping, split_ratio):

    raw_data = list(csv.reader(open(train_path, 'r')))[1:]
    split_num = int(len(raw_data)*split_ratio) + 850
    random.shuffle(raw_data)

    val = raw_data[-split_num:]
    train = raw_data[:-split_num]

    train_X = [line[:-1] for line in train]
    train_Y = [line[-1] for line in train]
    val_X = [line[:-1] for line in val]
    val_Y = [line[-1] for line in val]

    train_user = [userid_mapping[int(line[1])] for line in train_X]
    train_movie = [movieid_mapping[int(line[2])] for line in train_X]

    val_user = [userid_mapping[int(line[1])] for line in val_X]
    val_movie = [movieid_mapping[int(line[2])] for line in val_X]

    return (np.array(train_movie), np.array(train_user)),(np.array(val_movie), np.array(val_user)),(np.array(train_Y), np.array(val_Y))


def test_input(test_path, movieid_mapping, userid_mapping):
    with open(test_path, 'r') as f:
        raw_data = list(csv.reader(f))[1:]


    test_id = np.array([int(line[0]) for line in raw_data])
    test_user = np.array([userid_mapping[int(line[1])] for line in raw_data])
    test_movie = np.array([movieid_mapping[int(line[2])] for line in raw_data])

    return test_id, test_movie, test_user
