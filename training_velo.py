import re
from random import seed
from random import shuffle

import numpy as np
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, concatenate
from keras.models import Model, load_model


def get_class_weights(y, smooth_factor=0.0):
    # based on cbaziotis
    # https://github.com/fchollet/keras/issues/5116#issuecomment-274466261
    counter = [0 for _ in range(4)]
    for e in y:
        for i in range(4):
            if e[i] == 1:
                counter[i] += 1
    print("Zähler", counter)
    if smooth_factor > 0:
        p = max(counter) * smooth_factor
        for k in range(len(counter)):
            counter[k] += p
    majority = max(counter)
    return {index: majority / counter[index] if counter[index] != 0 else 0 for index in range(4)}


name = "test loss weights"
epochs = 20
# old_model = r"C:\Users\mivog\PycharmProjects\driving ai trackmania\velocity as auxiliary input\weights\new track mnist model_weights.05-0.13.hdf5"
old_model = ""
smoothing_factor = 0.01

tensboard = TensorBoard(log_dir="velo_logs/{}".format(name), histogram_freq=0, batch_size=32,
                        write_graph=False, write_grads=False, write_images=False, embeddings_freq=0,
                        embeddings_layer_names=None, embeddings_metadata=None
                        )

data = np.load(
    r"C:\Users\mivog\PycharmProjects\driving ai trackmania\velocity as auxiliary input\with_velocity\training_data_velo.npy")
seed(3)
shuffle(data)
# data = data[:100]  # testing purpose
split = int(0.6 * len(data))
training_data = data[:split]
testing_data = data[split:]
x_train = np.asarray([k[0] for k in training_data])
aux_train = np.asarray([k[1] for k in training_data])
y_train = np.asarray([k[2] for k in training_data])
x_test = np.asarray([k[0] for k in testing_data])
aux_test = np.asarray([k[1] for k in testing_data])
y_test = np.asarray([k[2] for k in testing_data])
x_train = x_train.astype('float32')
aux_train = aux_train.astype('float32')
x_test = x_test.astype('float32')
aux_test = aux_test.astype('float32')
x_train /= 255
x_test /= 255
aux_train /= 500
aux_test /= 500
# y_train = keras.utils.to_categorical(y_train, 4)
# y_test = keras.utils.to_categorical(y_test, 4)

starting_epoch = 0
if old_model == "":
    print(x_train.shape)
    main_input = Input(shape=x_train[0].shape, dtype='float32', name='main_input')

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(main_input)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    # x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    convs_out = Dense(128, activation='relu')(x)

    auxiliary_output = Dense(4, activation='sigmoid', name='aux_output')(convs_out)

    auxiliary_input = Input(shape=(1,), name='aux_input')
    x = concatenate([convs_out, auxiliary_input])

    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)

    main_output = Dense(4, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    model.compile(optimizer='Adam', loss="binary_crossentropy",
                  loss_weights=[1.0, 1.0])  # 0.2])  # , metrics=['accuracy']
    # print(model.evaluate([x_train, aux_train1], [y_train, y_train]))
else:
    model = load_model(old_model)
    print("Old model loaded.")
    reg = re.search("(?<=weights\.)\d+", old_model).group()
    if reg.isdigit():
        starting_epoch = int(reg) + 1
        print("Starting epoch", starting_epoch)

print(model.summary())
checkpoint = ModelCheckpoint("weights/" + name + "_weights.{epoch:02d}-{val_loss:.2f}.hdf5", period=1)
weights = get_class_weights(np.concatenate((y_train, y_test)), smooth_factor=smoothing_factor)
# weights[2] *= 2
# weights[3] *= 2
print("Weights:", weights)

model.fit([x_train, aux_train], [y_train, y_train],
          epochs=epochs, validation_data=([x_test, aux_test], [y_test, y_test]),
          class_weight=weights, callbacks=[tensboard, checkpoint], batch_size=32, initial_epoch=starting_epoch)

model.save("{}-{}.h5".format(name, epochs))
scores = model.evaluate([x_test, aux_test], [y_test, y_test], verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
print(scores)
