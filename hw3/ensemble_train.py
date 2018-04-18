import hw3_utilities as my
import keras
import sys
from keras import layers
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator


def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels = [model(model_input) for model in models]
    # averaging outputs
    yAvg = layers.average(yModels)
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')

    return modelEns

#DATA_FILE = "~/Desktop/machine_learning/hw/hw3/dataset/train.csv"
DATA_FILE = sys.argv[1]

# ----------------------------MODEL-DEEPER-REG23v2------------------------------
MODEL_NAME_reg23 = 'deeper_reg23v2'
# loading and splitting data to training and validation set
(X, y), (X_val, y_val) = my.load_train_data(DATA_FILE, validation_split=0)

model_reg22v2 = Sequential()
# Conv1
model_reg22v2.add(Conv2D(filters=128, kernel_size=3, padding='same',
                         input_shape=(48, 48, 1)))
model_reg22v2.add(Activation('selu'))
model_reg22v2.add(BatchNormalization())
model_reg22v2.add(Conv2D(filters=128, kernel_size=3, padding='same'))
model_reg22v2.add(Activation('selu'))
model_reg22v2.add(BatchNormalization())
model_reg22v2.add(Conv2D(filters=128, kernel_size=3, padding='same'))
model_reg22v2.add(Activation('selu'))
model_reg22v2.add(BatchNormalization())
model_reg22v2.add(MaxPooling2D((2, 2)))
model_reg22v2.add(Dropout(0.5))
# Conv2
model_reg22v2.add(Conv2D(filters=256, kernel_size=3, padding='same'))
model_reg22v2.add(Activation('selu'))
model_reg22v2.add(BatchNormalization())
model_reg22v2.add(Conv2D(filters=256, kernel_size=3, padding='same'))
model_reg22v2.add(Activation('selu'))
model_reg22v2.add(BatchNormalization())
model_reg22v2.add(Dropout(0.3))
model_reg22v2.add(Conv2D(filters=256, kernel_size=3, padding='same'))
model_reg22v2.add(Activation('selu'))
model_reg22v2.add(BatchNormalization())
model_reg22v2.add(MaxPooling2D((2, 2)))
model_reg22v2.add(Dropout(0.55))
# Conv3
model_reg22v2.add(Conv2D(filters=512, kernel_size=3, padding='same'))
model_reg22v2.add(Activation('selu'))
model_reg22v2.add(BatchNormalization())
model_reg22v2.add(Conv2D(filters=512, kernel_size=3, padding='same'))
model_reg22v2.add(Activation('selu'))
model_reg22v2.add(BatchNormalization())
model_reg22v2.add(Dropout(0.3))
model_reg22v2.add(Conv2D(filters=512, kernel_size=3, padding='same'))
model_reg22v2.add(Activation('selu'))
model_reg22v2.add(BatchNormalization())
model_reg22v2.add(MaxPooling2D((2, 2)))
model_reg22v2.add(Dropout(0.58))
# Conv4
model_reg22v2.add(Conv2D(filters=1024, kernel_size=3, padding='same'))
model_reg22v2.add(Activation('selu'))
model_reg22v2.add(BatchNormalization())
model_reg22v2.add(Conv2D(filters=1024, kernel_size=3, padding='same'))
model_reg22v2.add(Activation('selu'))
model_reg22v2.add(BatchNormalization())
model_reg22v2.add(Dropout(0.3))
model_reg22v2.add(Conv2D(filters=1024, kernel_size=3, padding='same'))
model_reg22v2.add(Activation('selu'))
model_reg22v2.add(BatchNormalization())
model_reg22v2.add(MaxPooling2D((2, 2)))
model_reg22v2.add(Dropout(0.58))
# FC1
model_reg22v2.add(Flatten())
model_reg22v2.add(Dense(units=512))
model_reg22v2.add(Activation('selu'))
model_reg22v2.add(BatchNormalization())
model_reg22v2.add(Dropout(0.6))
# Output Layer
model_reg22v2.add(Dense(units=7))
model_reg22v2.add(Activation('softmax'))

# Image data augmentation
datagen = ImageDataGenerator(
    rotation_range=39,
    shear_range=0.27,
    width_shift_range=0.16,
    height_shift_range=0.16,
    horizontal_flip=True
    )
datagen.fit(X)

# Model HYPER-parameters
EPOCHS = 75
BATCH_SIZE = 32
cb_adaLR = ReduceLROnPlateau(monitor='loss', factor=0.75, patience=4,
                             verbose=1)
cb_tb = TensorBoard(log_dir='./logs/'+MODEL_NAME_reg23, batch_size=BATCH_SIZE)

# Model compilation and fitting
model_reg22v2.summary()
model_reg22v2.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
hist = model_reg22v2.fit_generator(
        initial_epoch=0, epochs=EPOCHS, callbacks=[cb_tb, cb_adaLR],
        generator=datagen.flow(X, y, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X)//BATCH_SIZE,
        validation_data=(X_val, y_val),
        validation_steps=len(X_val)//BATCH_SIZE,
        workers=0, use_multiprocessing=True
        )
model_reg22v2.save('model_'+MODEL_NAME_reg23+'.h5')
# ----------------------------MODEL-DEEPER-REG23v2------------------------------

# loading and splitting data to training and validation set
# (X, y), (X_val, y_val) = my.load_train_data(DATA_FILE,
#                                             validation_split=0.2)
(X, y), (X_val, y_val) = my.load_train_data(DATA_FILE,
                                            validation_split=0.1)

# ----------------------------MODEL-DEEPER-REG11--------------------------------
MODEL_NAME_reg11 = 'deeper_reg11'

model_reg11 = Sequential()
# Conv1
model_reg11.add(Conv2D(filters=128, kernel_size=3, padding='same',
                       input_shape=(48, 48, 1)))
model_reg11.add(Activation('selu'))
model_reg11.add(BatchNormalization())
model_reg11.add(Conv2D(filters=128, kernel_size=3, padding='same'))
model_reg11.add(Activation('selu'))
model_reg11.add(BatchNormalization())
model_reg11.add(Conv2D(filters=128, kernel_size=3, padding='same'))
model_reg11.add(Activation('selu'))
model_reg11.add(BatchNormalization())
model_reg11.add(MaxPooling2D((2, 2)))
model_reg11.add(Dropout(0.4))
# Conv2
model_reg11.add(Conv2D(filters=256, kernel_size=3, padding='same'))
model_reg11.add(Activation('selu'))
model_reg11.add(BatchNormalization())
model_reg11.add(Conv2D(filters=256, kernel_size=3, padding='same'))
model_reg11.add(Activation('selu'))
model_reg11.add(BatchNormalization())
model_reg11.add(Conv2D(filters=256, kernel_size=3, padding='same'))
model_reg11.add(Activation('selu'))
model_reg11.add(BatchNormalization())
model_reg11.add(MaxPooling2D((2, 2)))
model_reg11.add(Dropout(0.45))
# Conv3
model_reg11.add(Conv2D(filters=512, kernel_size=3, padding='same'))
model_reg11.add(Activation('selu'))
model_reg11.add(BatchNormalization())
model_reg11.add(Conv2D(filters=512, kernel_size=3, padding='same'))
model_reg11.add(Activation('selu'))
model_reg11.add(BatchNormalization())
model_reg11.add(Conv2D(filters=512, kernel_size=3, padding='same'))
model_reg11.add(Activation('selu'))
model_reg11.add(BatchNormalization())
model_reg11.add(MaxPooling2D((2, 2)))
model_reg11.add(Dropout(0.45))
# Conv4
model_reg11.add(Conv2D(filters=1024, kernel_size=3, padding='same'))
model_reg11.add(Activation('selu'))
model_reg11.add(BatchNormalization())
model_reg11.add(Conv2D(filters=1024, kernel_size=3, padding='same'))
model_reg11.add(Activation('selu'))
model_reg11.add(BatchNormalization())
model_reg11.add(Conv2D(filters=1024, kernel_size=3, padding='same'))
model_reg11.add(Activation('selu'))
model_reg11.add(BatchNormalization())
model_reg11.add(MaxPooling2D((2, 2)))
model_reg11.add(Dropout(0.5))
# FC1
model_reg11.add(Flatten())
model_reg11.add(Dense(units=512))
model_reg11.add(Activation('selu'))
model_reg11.add(BatchNormalization())
model_reg11.add(Dropout(0.5))
# Output Layer
model_reg11.add(Dense(units=7))
model_reg11.add(Activation('softmax'))

# Image data augmentation
datagen = ImageDataGenerator(
    rotation_range=39,
    shear_range=0.27,
    width_shift_range=0.16,
    height_shift_range=0.16,
    horizontal_flip=True
    )
datagen.fit(X)

# Model HYPER-parameters
EPOCHS = 100
BATCH_SIZE = 32
cb_adaLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                             verbose=1)
cb_tb = TensorBoard(log_dir='./logs/'+MODEL_NAME_reg11, batch_size=BATCH_SIZE)

# Model compilation and fitting
model_reg11.summary()
model_reg11.compile(loss='categorical_crossentropy', optimizer='adam',
                    metrics=['accuracy'])
hist = model_reg11.fit_generator(
        initial_epoch=0, epochs=EPOCHS, callbacks=[cb_adaLR, cb_tb],
        generator=datagen.flow(X, y, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X)//BATCH_SIZE,
        validation_data=(X_val, y_val),
        validation_steps=len(X_val)//BATCH_SIZE,
        workers=0, use_multiprocessing=True
        )
model_reg11.save('model_'+MODEL_NAME_reg11+'.h5')
# ----------------------------MODEL-DEEPER-REG11--------------------------------

# ----------------------------MODEL-DEEPER-REG5---------------------------------
MODEL_NAME_reg5 = 'deeper_reg5'

model_reg5 = Sequential()
# Conv1
model_reg5.add(Conv2D(filters=128, kernel_size=3, padding='same',
                      input_shape=(48, 48, 1)))
model_reg5.add(Activation('selu'))
model_reg5.add(BatchNormalization())
model_reg5.add(Conv2D(filters=128, kernel_size=3, padding='same'))
model_reg5.add(Activation('selu'))
model_reg5.add(BatchNormalization())
model_reg5.add(MaxPooling2D((2, 2)))
model_reg5.add(Dropout(0.15))
# Conv2
model_reg5.add(Conv2D(filters=256, kernel_size=3, padding='same'))
model_reg5.add(Activation('selu'))
model_reg5.add(BatchNormalization())
model_reg5.add(Conv2D(filters=256, kernel_size=3, padding='same'))
model_reg5.add(Activation('selu'))
model_reg5.add(BatchNormalization())
model_reg5.add(MaxPooling2D((2, 2)))
model_reg5.add(Dropout(0.25))
# Conv3
model_reg5.add(Conv2D(filters=512, kernel_size=3, padding='same'))
model_reg5.add(Activation('selu'))
model_reg5.add(BatchNormalization())
model_reg5.add(Conv2D(filters=512, kernel_size=3, padding='same'))
model_reg5.add(Activation('selu'))
model_reg5.add(BatchNormalization())
model_reg5.add(MaxPooling2D((2, 2)))
model_reg5.add(Dropout(0.35))
# Conv4
model_reg5.add(Conv2D(filters=1024, kernel_size=3, padding='same'))
model_reg5.add(Activation('selu'))
model_reg5.add(BatchNormalization())
model_reg5.add(Conv2D(filters=1024, kernel_size=3, padding='same'))
model_reg5.add(Activation('selu'))
model_reg5.add(BatchNormalization())
model_reg5.add(MaxPooling2D((2, 2)))
model_reg5.add(Dropout(0.4))
# Conv5
model_reg5.add(Conv2D(filters=2048, kernel_size=3, padding='same'))
model_reg5.add(Activation('selu'))
model_reg5.add(BatchNormalization())
model_reg5.add(Conv2D(filters=2048, kernel_size=3, padding='same'))
model_reg5.add(Activation('selu'))
model_reg5.add(BatchNormalization())
model_reg5.add(MaxPooling2D((2, 2)))
model_reg5.add(Dropout(0.45))
# FC1
model_reg5.add(Flatten())
model_reg5.add(Dense(units=2048))
model_reg5.add(Activation('selu'))
model_reg5.add(BatchNormalization())
model_reg5.add(Dropout(0.5))
# FC2
model_reg5.add(Dense(units=1024))
model_reg5.add(Activation('selu'))
model_reg5.add(BatchNormalization())
model_reg5.add(Dropout(0.5))
# FC3
model_reg5.add(Dense(units=512))
model_reg5.add(Activation('selu'))
model_reg5.add(BatchNormalization())
model_reg5.add(Dropout(0.5))
# Output Layer
model_reg5.add(Dense(units=7))
model_reg5.add(Activation('softmax'))

# Image data augmentation
datagen = ImageDataGenerator(
    rotation_range=39,
    shear_range=0.27,
    width_shift_range=0.16,
    height_shift_range=0.16,
    horizontal_flip=True
    )

datagen.fit(X)

# Model HYPER-parameters
EPOCHS = 100
BATCH_SIZE = 32
cb_adaLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                             verbose=1)
cb_tb = TensorBoard(log_dir='./logs/'+MODEL_NAME_reg5, batch_size=BATCH_SIZE)

# Model compilation and fitting
model_reg5.summary()
model_reg5.compile(loss='categorical_crossentropy', optimizer='adam',
                   metrics=['accuracy'])
hist = model_reg5.fit_generator(
        initial_epoch=0, epochs=EPOCHS, callbacks=[cb_adaLR, cb_tb],
        generator=datagen.flow(X, y, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X)//BATCH_SIZE,
        validation_data=(X_val, y_val),
        validation_steps=len(X_val)//BATCH_SIZE,
        workers=0, use_multiprocessing=True
        )
model_reg5.save('model_'+MODEL_NAME_reg5+'.h5')
# ----------------------------MODEL-DEEPER-REG5---------------------------------

# ----------------------------MODEL-RESNET-2------------------------------------
MODEL_NAME_res2 = 'res_test_2'

# loading and splitting data to training and validation set
# (X, y), (X_val, y_val) = my.load_train_data(DATA_FILE,
#                                             validation_split=0.2)
(X_train, y_train), (X_val, y_val) = my.load_train_data(DATA_FILE,
                                            validation_split=0.1)

# input tensor for a 1-channel 48x48 image
ins = Input(shape=(48, 48, 1))

y = Conv2D(64, (3, 3), padding='same', activation='relu')(ins)
y = BatchNormalization()(y)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(y)
x = BatchNormalization()(x)
y = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
y = BatchNormalization()(y)
y = Conv2D(64, (3, 3), padding='same', activation='relu')(y)
y = BatchNormalization()(y)
x = keras.layers.add([x, y])
x = Dropout(.5)(x)
y = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
y = BatchNormalization()(y)
y = Conv2D(64, (3, 3), padding='same', activation='relu')(y)
y = BatchNormalization()(y)
x = keras.layers.add([x, y])
x = Dropout(.5)(x)

y = Conv2D(128, (3, 3), padding='same', activation='relu', strides=2)(x)
y = BatchNormalization()(y)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(y)
x = BatchNormalization()(x)
y = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
y = BatchNormalization()(y)
y = Conv2D(128, (3, 3), padding='same', activation='relu')(y)
y = BatchNormalization()(y)
x = keras.layers.add([x, y])
x = Dropout(.5)(x)
y = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
y = BatchNormalization()(y)
y = Conv2D(128, (3, 3), padding='same', activation='relu')(y)
y = BatchNormalization()(y)
x = keras.layers.add([x, y])
x = Dropout(.5)(x)

y = Conv2D(256, (3, 3), padding='same', activation='relu', strides=2)(x)
y = BatchNormalization()(y)
x = Conv2D(256, (3, 3), padding='same', activation='relu')(y)
x = BatchNormalization()(x)
y = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
y = BatchNormalization()(y)
y = Conv2D(256, (3, 3), padding='same', activation='relu')(y)
y = BatchNormalization()(y)
x = keras.layers.add([x, y])
x = Dropout(.5)(x)
y = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
y = BatchNormalization()(y)
y = Conv2D(256, (3, 3), padding='same', activation='relu')(y)
y = BatchNormalization()(y)
x = keras.layers.add([x, y])
x = Dropout(.5)(x)
y = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
y = BatchNormalization()(y)
y = Conv2D(256, (3, 3), padding='same', activation='relu')(y)
y = BatchNormalization()(y)
x = keras.layers.add([x, y])
x = Dropout(.5)(x)


x = AveragePooling2D(pool_size=(3, 3), strides=2)(x)

x = Flatten()(x)

x = Dense(units=512, activation='relu')(x)
x = BatchNormalization()(x)

outs = Dense(units=7, activation='softmax')(x)

model_res2 = Model(ins, outs)


# Image data augmentation
datagen = ImageDataGenerator(
    rotation_range=39,
    shear_range=0.27,
    width_shift_range=0.16,
    height_shift_range=0.16,
    horizontal_flip=True
    )
datagen.fit(X_train)

# Model HYPER-parameters
EPOCHS = 100
BATCH_SIZE = 32
cb_adaLR = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=4,
                             verbose=1)
cb_tb = TensorBoard(log_dir='./logs/'+MODEL_NAME_res2, batch_size=BATCH_SIZE)

# Model compilation and fitting
model_res2.summary()
model_res2.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])

hist = model_res2.fit_generator(
        initial_epoch=0, epochs=EPOCHS, callbacks=[cb_tb, cb_adaLR],
        generator=datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train)//BATCH_SIZE, validation_data=(X_val, y_val),
        validation_steps=len(X_val)//BATCH_SIZE,
        workers=0, use_multiprocessing=True
        )
model_res2.save('model_'+MODEL_NAME_res2+'.h5')
# ----------------------------MODEL-RESNET-2------------------------------------

# ----------------------------MODEL-INCEPTION-----------------------------------
MODEL_NAME_incept = 'inception_v4_1'
# loading and splitting data to training and validation set
(X, y), (X_val, y_val) = my.load_train_data(DATA_FILE, validation_split=0.2)

# input tensor for a 1-channel 48x48 image
ins = Input(shape=(48, 48, 1))

# Conv1
x = Conv2D(32, (3, 3), padding='same', activation='relu')(ins)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
# Conv2
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu', strides=(2, 2))(x)
x = BatchNormalization()(x)
# Inception 1-1
tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
tower_3 = BatchNormalization()(tower_3)
tower_4 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_4 = BatchNormalization()(tower_4)
x = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)
# Inception 1-2
tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
tower_3 = BatchNormalization()(tower_3)
tower_4 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_4 = BatchNormalization()(tower_4)
x = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)
# Inception 2-1
tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (1, 6), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (6, 1), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (1, 6), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (6, 1), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Conv2D(64, (1, 6), padding='same', activation='relu')(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Conv2D(64, (6, 1), padding='same', activation='relu')(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
tower_3 = BatchNormalization()(tower_3)
tower_4 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_4 = BatchNormalization()(tower_4)
x = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)
# Inception 2-2
tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (1, 6), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (6, 1), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (1, 6), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (6, 1), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Conv2D(64, (1, 6), padding='same', activation='relu')(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Conv2D(64, (6, 1), padding='same', activation='relu')(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
tower_3 = BatchNormalization()(tower_3)
tower_4 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_4 = BatchNormalization()(tower_4)
x = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)
# Inception 2-3
tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (1, 6), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (6, 1), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (1, 6), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (6, 1), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Conv2D(64, (1, 6), padding='same', activation='relu')(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_2 = Conv2D(64, (6, 1), padding='same', activation='relu')(tower_2)
tower_2 = BatchNormalization()(tower_2)
tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
tower_3 = BatchNormalization()(tower_3)
tower_4 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_4 = BatchNormalization()(tower_4)
x = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)
# Inception 3-1
tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1_1 = Conv2D(64, (1, 3), padding='same', activation='relu')(tower_1)
tower_1_1 = BatchNormalization()(tower_1_1)
tower_1_2 = Conv2D(64, (1, 3), padding='same', activation='relu')(tower_1)
tower_1_2 = BatchNormalization()(tower_1_2)
tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_2 = BatchNormalization()(tower_2)
tower_2_1 = Conv2D(64, (1, 3), padding='same', activation='relu')(tower_2)
tower_2_1 = BatchNormalization()(tower_2_1)
tower_2_2 = Conv2D(64, (3, 1), padding='same', activation='relu')(tower_2)
tower_2_2 = BatchNormalization()(tower_2_2)
tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
tower_3 = BatchNormalization()(tower_3)
tower_4 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_4 = BatchNormalization()(tower_4)
x = keras.layers.concatenate([tower_1_1, tower_1_2, tower_2_1, tower_2_2, tower_3, tower_4], axis=3)
# Inception 3-2
tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_1 = BatchNormalization()(tower_1)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
tower_1 = BatchNormalization()(tower_1)
tower_1_1 = Conv2D(64, (1, 3), padding='same', activation='relu')(tower_1)
tower_1_1 = BatchNormalization()(tower_1_1)
tower_1_2 = Conv2D(64, (1, 3), padding='same', activation='relu')(tower_1)
tower_1_2 = BatchNormalization()(tower_1_2)
tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_2 = BatchNormalization()(tower_2)
tower_2_1 = Conv2D(64, (1, 3), padding='same', activation='relu')(tower_2)
tower_2_1 = BatchNormalization()(tower_2_1)
tower_2_2 = Conv2D(64, (3, 1), padding='same', activation='relu')(tower_2)
tower_2_2 = BatchNormalization()(tower_2_2)
tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
tower_3 = BatchNormalization()(tower_3)
tower_4 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_4 = BatchNormalization()(tower_4)
x = keras.layers.concatenate([tower_1_1, tower_1_2, tower_2_1, tower_2_2, tower_3, tower_4], axis=3)

x = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
x = Flatten()(x)

x = Dropout(.5)(x)
x = Dense(units=512, activation='relu')(x)
x = BatchNormalization()(x)


outs = Dense(units=7, activation='softmax')(x)

model_incep = Model(ins, outs)

# Image data augmentation
datagen = ImageDataGenerator(
    rotation_range=39,
    shear_range=0.27,
    width_shift_range=0.16,
    height_shift_range=0.16,
    horizontal_flip=True
    )
datagen.fit(X)

# Model HYPER-parameters
EPOCHS = 200
BATCH_SIZE = 64
cb_adaLR = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=8, verbose=1)
cb_tb = TensorBoard(log_dir='./logs/'+MODEL_NAME_incept, batch_size=BATCH_SIZE)


# Model compilation and fitting
model_incep.summary()
model_incep.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
#model.fit(x=X, y=y, batch_size=BATCH_SIZE, epochs=EPOCHS)
hist = model_incep.fit_generator(
        initial_epoch=0, epochs=EPOCHS, callbacks=[cb_tb, cb_adaLR],
        generator=datagen.flow(X, y, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X)//BATCH_SIZE, validation_data=(X_val, y_val),
        validation_steps=len(X_val)//BATCH_SIZE,
        workers=0, use_multiprocessing=True
        )
model_incep.save('model_'+MODEL_NAME_incept+'.h5')
# ----------------------------MODEL-INCEPTION-----------------------------------


# ----------------------------ENSEMBLE-MODEL------------------------------------
MODEL_NAME_reg23 = 'deeper_reg23v2'
MODEL_NAME_reg11 = 'deeper_reg11'
MODEL_NAME_reg5 = 'deeper_reg5'
MODEL_NAME_res2 = 'res_test_2'
MODEL_NAME_incept = 'inception_v4_1'

model_names = [MODEL_NAME_reg23,
               MODEL_NAME_reg11,
               MODEL_NAME_reg5,
               MODEL_NAME_res2,
               MODEL_NAME_incept]
version_name = 'reg23v2-reg11-reg5-res2-incept1'
MODEL_PATH_Ens = './model_ensembled_'+version_name+'.h5'
models = []

for name in model_names:
    modelTemp = load_model('./model_'+name+'.h5')  # load model
    modelTemp.name = name  # change name to be unique
    models.append(modelTemp)

# h*w*c=(48, 48, 1)
model_input = Input(shape=models[0].input_shape[1:])
modelEns = ensembleModels(models, model_input)
modelEns.summary()
modelEns.compile(optimizer='adam', loss='categorical_crossentropy',
                 metrics=['accuracy'])
modelEns.save(MODEL_PATH_Ens)
# ----------------------------ENSEMBLE-MODEL------------------------------------
