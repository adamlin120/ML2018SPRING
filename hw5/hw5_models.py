from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.layers import Input, Dense, Embedding, LSTM, SpatialDropout1D, GRU
from keras.layers import Bidirectional, Dropout


def BRNN(num_word, embed_dim, max_len, emb_matrix, dropout_rate=.2):
    inputs = Input(shape=(max_len,))

    # Embedding layer
    embedding_inputs = Embedding(input_dim=num_word+1,
                                 output_dim=embed_dim,
                                 input_length=max_len,
                                 weights=[emb_matrix],
                                 mask_zero=True,
                                 trainable=False)(inputs)
    embedding_inputs = SpatialDropout1D(0.4)(embedding_inputs)
    # RNN
    RNN_output = Bidirectional(
                    LSTM(64, return_sequences=True, dropout=dropout_rate,
                         recurrent_dropout=dropout_rate))(embedding_inputs)
    RNN_output = Bidirectional(
                    LSTM(64, return_sequences=False, dropout=dropout_rate,
                         recurrent_dropout=dropout_rate))(RNN_output)
    # DNN layer
    outputs = Dense(32, activation='relu',
                    kernel_regularizer=l2(0.01))(RNN_output)
    outputs = Dense(2, activation='softmax')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])

    return model


def BLSTM(num_word, embed_dim, max_len, emb_matrix, dropout_rate=.2,
          spatial_dropout_rate=.4):
    inputs = Input(shape=(max_len,))

    # Embedding layer
    embedding_inputs = Embedding(input_dim=num_word+1,
                                 output_dim=embed_dim,
                                 input_length=max_len,
                                 weights=[emb_matrix],
                                 mask_zero=True,
                                 trainable=False)(inputs)
    embedding_inputs = SpatialDropout1D(spatial_dropout_rate)(embedding_inputs)
    # RNN
    RNN_output = Bidirectional(
                    LSTM(64, return_sequences=True, dropout=dropout_rate,
                         recurrent_dropout=dropout_rate))(embedding_inputs)
    RNN_output = Bidirectional(
                    LSTM(64, return_sequences=False, dropout=dropout_rate,
                         recurrent_dropout=dropout_rate))(RNN_output)
    # DNN layer
    outputs = Dense(32, activation='relu',
                    kernel_regularizer=l2(0.01))(RNN_output)
    outputs = Dense(2, activation='softmax')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])

    return model


def BGRU(num_word, embed_dim, max_len, emb_matrix, dropout_rate=.2):
    inputs = Input(shape=(max_len,))

    # Embedding layer
    embedding_inputs = Embedding(input_dim=num_word+1,
                                 output_dim=embed_dim,
                                 input_length=max_len,
                                 weights=[emb_matrix],
                                 mask_zero=True,
                                 trainable=False)(inputs)
    embedding_inputs = SpatialDropout1D(0.4)(embedding_inputs)
    # RNN
    RNN_output = Bidirectional(
                    GRU(64, return_sequences=True, dropout=dropout_rate,
                         recurrent_dropout=dropout_rate))(embedding_inputs)
    RNN_output = Bidirectional(
                    GRU(64, return_sequences=False, dropout=dropout_rate,
                         recurrent_dropout=dropout_rate))(RNN_output)
    # DNN layer
    outputs = Dense(32, activation='relu',
                    kernel_regularizer=l2(0.01))(RNN_output)
    outputs = Dense(2, activation='softmax')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])

    return model


def BOW_DNN():
    model = Sequential()
    # Dense layers
    model.add(Dense(128, activation='relu', input_dim=82117))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])

    return model