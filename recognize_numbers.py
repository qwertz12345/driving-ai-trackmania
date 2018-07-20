import os
from random import shuffle

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


def get_data():
    path = r"C:\Users\mivog\PycharmProjects\driving ai trackmania\velocity as auxiliary input\sample"
    x_ = []
    y_ = []
    for k in range(10):
        cur_dir = os.path.join(path, str(k))
        files_list = os.listdir(cur_dir)
        for j in files_list:
            x_.append(cv2.imread(os.path.join(cur_dir, j)))
            y_.append(k)
    files_list = os.listdir(
        r"C:\Users\mivog\PycharmProjects\driving ai trackmania\velocity as auxiliary input\sample\empty")
    for j in files_list:
        x_.append(cv2.imread(
            os.path.join(
                r"C:\Users\mivog\PycharmProjects\driving ai trackmania\velocity as auxiliary input\sample\empty",
                j)))
        y_.append(10)
    return x_, y_


def look_at_data():
    x, y = get_data()
    for i, e in enumerate(x):
        print(y[i])
        cv2.imshow(str(y[i]), e)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def build_model(img_shape): # 'batch_input_shape': (None, 32, 18, 3)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=img_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(11))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  )
    return model


def training(name, epochs, old_model=""):
    _x, _y = get_data()
    indeces = [t for t in range(len(_x))]
    shuffle(indeces)
    x = []
    y = []
    for i in indeces:
        x.append(_x[i])
        y.append(_y[i])
    y = to_categorical(y, num_classes=11)
    assert len(x) == len(y)
    x = np.array(x).astype('float32')
    x /= 255
    y = np.array(y)
    split = int(0.8 * len(x))
    x_train = x[:split]
    x_val = x[split:]
    y_train = y[:split]
    y_val = y[split:]

    tensboard = TensorBoard(log_dir="logs/{}".format(name), histogram_freq=0, batch_size=32,
                            write_graph=False, write_grads=False, write_images=False, embeddings_freq=0,
                            embeddings_layer_names=None, embeddings_metadata=None
                            )
    starting_epoch = 0
    if old_model == "":
        model = build_model(x_train[0].shape)
    else:
        model = load_model(old_model)
        print("Old model loaded.")
    train_datagen = ImageDataGenerator(
        width_shift_range=.1, height_shift_range=.1, channel_shift_range=.1)
    checkpoint = ModelCheckpoint("weights/" + name + "_weights.{epoch:02d}-{val_loss:.2f}.hdf5", period=1)
    model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=32),
                        steps_per_epoch=x_train.shape[0] // 32,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        callbacks=[tensboard, checkpoint],
                        initial_epoch=starting_epoch)
    model.save("{}-{}.h5".format(name, epochs))
    print("Model saved")


def recognize(images, model):
    # images = np.asarray(images)
    predictions = model.predict(images)  # , batch_size=1)  # 'batch_input_shape': (None, 32, 18, 3)
    # print([a for a in zip(range(10), prediction[0])])
    result = [(str(np.argmax(e)), max(e)) if np.argmax(e) < 10 else ("0", max(e)) for e in predictions]
    return result


if __name__ == '__main__':
    training("velocity_recognition", 100)
    # print(recognize(cv2.imread(
    #     r"C:\Users\mivog\PycharmProjects\driving ai trackmania\velocity as auxiliary input\d_811.png")))
