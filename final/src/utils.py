import numpy as np
from keras.layers import average
from keras.models import Model
from keras.utils import Sequence
from keras.utils import to_categorical
import librosa


class Config(object):
    def __init__(self,
                 model_name,
                 sampling_rate=16000, audio_duration=2, n_classes=41,
                 use_mfcc=False, use_delta=False, use_Ddelta=False,
                 n_folds=10, n_mfcc=20,
                 learning_rate=0.0001, max_epochs=50, batch_size=512,
                 random_seed=13):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.model_name = model_name
        self.use_delta = use_delta
        self.use_Ddelta = use_Ddelta

        if self.use_delta:
            if self.use_Ddelta:
                self.n_channel = 3
            else:
                self.n_channel = 2
        else:
            self.n_channel = 1

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc,
                        1 + int(np.floor(self.audio_length/(512))),
                        self.n_channel)
        else:
            self.dim = (self.audio_length, 1)


def ensembleModels(models, model_input, model_name=None):
    # collect outputs of models in a list
    yModels = [model(model_input) for model in models]
    # averaging outputs
    yAvg = average(yModels)
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name=model_name)

    return modelEns


class DataGenerator(Sequence):
    def __init__(self, config, data_dir, list_IDs, labels=None,
                 batch_size=64, preprocessing_fn=lambda x: x):
        self.config = config
        self.data_dir = data_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.preprocessing_fn = preprocessing_fn
        self.on_epoch_end()
        self.dim = self.config.dim

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        cur_batch_size = len(list_IDs_temp)
        X = np.empty((cur_batch_size, *self.dim))

        input_length = self.config.audio_length
        for i, ID in enumerate(list_IDs_temp):
            file_path = self.data_dir + ID

            # Read and Resample the audio
            data, _ = librosa.core.load(file_path,
                                        sr=self.config.sampling_rate,
                                        res_type='kaiser_fast')

            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data,
                              (offset, input_length - len(data) - offset),
                              "constant")

            # Normalization + Other Preprocessing
            if self.config.use_mfcc:
                data = librosa.feature.mfcc(data,
                                            sr=self.config.sampling_rate,
                                            n_mfcc=self.config.n_mfcc)
                data = np.expand_dims(data, axis=-1)
            else:
                data = self.preprocessing_fn(data)[:, np.newaxis]
            X[i, ] = data

        if self.labels is not None:
            y = np.empty(cur_batch_size, dtype=int)
            for i, ID in enumerate(list_IDs_temp):
                y[i] = self.labels[ID]
            return X, to_categorical(y, num_classes=self.config.n_classes)
        else:
            return X


def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5
