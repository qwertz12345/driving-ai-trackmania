import os

import cv2
import numpy as np
from keras.models import load_model

from load_data import load_data
from recognize_numbers import recognize


def keys_to_ar(keys):
    key_dict = {
        "W": 0,
        "S": 1,
        "A": 2,
        "D": 3
    }
    ar = [0] * 4
    for key in keys:
        ar[key_dict[key]] = 1
    return ar


training_data_directory = os.path.abspath(r"E:\Trackmania Data\training_data_new")


def flip_images(data):
    data_flipped = []
    for d in data:
        img = d[0]
        velo = d[1]
        keys = d[2]
        img_flipped = cv2.flip(img, flipCode=1)
        keys_flipped = ['A' if e == 'D' else 'D' if e == 'A' else e for e in keys]
        data_flipped.append([img_flipped, velo, keys_flipped])
    return np.concatenate((data, data_flipped))


def get_velocities(velocity_images, velo_recognition_model=load_model(
                   r"C:\Users\mivog\PycharmProjects\driving ai trackmania\velocity_recognition-100.h5")):
    img_digits = [np.zeros((32, 18, 3), dtype='float32') for _ in range(3 * len(velocity_images))]
    for i in range(len(velocity_images)):
        temp = velocity_images[i] / 255
        img_digits[3 * i][0:temp.shape[0]][:] = temp[:, :18]
        img_digits[3 * i + 1][0:temp.shape[0]][:] = temp[:, 17:35]
        img_digits[3 * i + 2][0:temp.shape[0]][:] = temp[:, 35:53]

        # img_digits.append(img[:, :18])  # here: height, width
        # img_digits.append(img[:, 18:36])
        # img_digits.append(img[:, 36:54])
    digits_with_confidence_value = recognize(np.array(img_digits),
                                             velo_recognition_model)  # 'batch_input_shape': (None, 32, 18, 3)
    velocities = np.zeros(len(img_digits) // 3)
    i = 0
    for j in range(len(velocities)):
        velocities[j] = digits_with_confidence_value[i][0] + digits_with_confidence_value[i + 1][0] + \
                        digits_with_confidence_value[i + 2][0]
        velocities[j] = int(velocities[j])
        i += 3
    return velocities


def main():
    velo_recognition_model = load_model(
        r"C:\Users\mivog\PycharmProjects\driving ai trackmania\velocity_recognition-100.h5")
    data = load_data(training_data_directory)
    velocities = get_velocities(data[:, 1], velo_recognition_model)
    assert len(data) == len(velocities)
    # print(velocities[:30])
    data_velos = []
    for i in range(len(data)):
        if velocities[i] < 500:
            data_velos.append([data[i, 0], velocities[i], data[i, 2]])
    data = flip_images(np.array(data_velos))
    for i, d in enumerate(data):
        data[i, 2] = keys_to_ar(d[2])
    #
    print("New data length: " + str(len(data)))
    np.save(os.path.join(training_data_directory, "training_data_velo.npy"), data)


if __name__ == "__main__":
    # execute only if run as a script
    main()
