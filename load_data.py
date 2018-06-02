import os
import numpy as np


def load_data(training_data_directory):
    print("Loading data")
    starting_value = 1
    file_name = os.path.join(training_data_directory, "training_data-1_{}.npy".format(starting_value))
    training_data = np.load(file_name)
    starting_value += 1
    while True:
        file_name = os.path.join(training_data_directory, "training_data-1_{}.npy".format(starting_value))
        if os.path.isfile(file_name):
            training_data = np.concatenate((training_data, np.load(file_name)))
            starting_value += 1
        else:
            break

    print("Data with length", len(training_data), "loaded")
    return training_data


if __name__ == '__main__':
    data = load_data(r"E:\Trackmania Data\training_data_new")
    print(data[0, 2])
