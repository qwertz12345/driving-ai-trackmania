import pathlib
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

from grabscreen import grab_screen
from getkeys import key_check


# TODO: DEBUG


def pause(time_adjustment):
    print("Paused. Press 'r' to resume ...")
    pause_start_time = time.time()
    stop = False
    time.sleep(0.5)
    while True:
        keys = key_check()
        time.sleep(0.5)
        if "O" and "P" in keys:
            print("Stop recording and saving ...")
            stop = True
            break
        elif "R" in keys:
            print("Unpausing ...")
            break
    pause_stop_time = time.time()

    return time_adjustment + pause_stop_time - pause_start_time, stop


def save_image(name, image):
    im = Image.fromarray(image[..., ::-1])
    im.save(name)


def save(training_data, run_name):
    """
    Saves images as png, key and time data as npy.
    Reduce data size by resizing and compressing (lossless) the images before saving.
    :param training_data: Tuple of the image, keys and time stamps
    :param run_name: Usually the track name.
    """

    keys_and_times = np.array([(elem[1], elem[2]) for elem in training_data])
    time_diffs = np.diff(np.concatenate((np.array([0.0]), keys_and_times[:, 1])))
    training_directory = r"E:\Trackmania Data\training_data_new\\" + run_name + "_" + str(
        time.strftime("%Y%m%d-%H%M%S"))
    pathlib.Path(training_directory + r"\\screenshots").mkdir(parents=True)
    with open(training_directory + r"\\" + "keys_timings.txt", "w") as text_file:
        for k, line in enumerate(keys_and_times):
            text_file.write("{0:5}     {1:20s}     {2:f}       {3:f}\n".format(k, str(line[0]), line[1], time_diffs[k]))

    image_names = []
    for k, _ in enumerate(training_data):
        image_names.append(training_directory + r"\\screenshots\\" + str(k) + "_" + str(keys_and_times[k][0]) + ".png")

    pool = Pool()
    pool.starmap(save_image, [(image_names[k], e[0]) for k, e in enumerate(training_data)])
    pool.close()
    pool.join()


def main(file_name):
    screen = grab_screen(region=(8, 200, 800, 430), window_title="TrackMania Nations Forever")
    training_data = []
    _ = key_check()
    time_adjustment, stop_signal = pause(time.time())
    print("Recording ...")

    while True:
        keys = key_check()
        if "T" in keys:
            time_adjustment, stop_signal = pause(time_adjustment)
            if stop_signal:
                break
        screenshot = screen.getFrame()
        training_data.append((screenshot, keys, time.time() - time_adjustment))

    screen.clear()
    save(training_data, file_name)


if __name__ == '__main__':
    save_name = input("Enter run name: ")
    print("Keys: 't' to pause \n"
          "      'r' to resume \n"
          "      'o' and 'p' simultaneously while paused to save and exit.")
    main(save_name)
