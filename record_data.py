import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from getkeys import key_check
from grabscreen import grab_screen


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


# def resize_and_crop_image(image):
#     """
#     :param image:
#     :return: also convert to grayscale
#     """
#     edited = image[0].crop(box=(0, 0, 800, 320)).resize((400, 160), Image.LANCZOS)
#     return edited, image[1]
#
#
# def resizing_cropping(images):
#     pool = Pool()
#     resized = pool.map(resize_and_crop_image, images)
#     pool.close()
#     pool.join()
#     print("Images resized.")
#     return resized


def save_image(path, image):
    im = Image.fromarray(image[..., ::-1]).crop(box=(0, 0, 800, 320)).resize((400, 160))  # , Image.LANCZOS)
    im.save(path)


def save(training_data, run_name):
    """
    Saves images as png, key and time data as npy.
    Reduce data size by resizing and compressing (lossless) the images before saving.
    :param training_data: Tuple of the image, keys and time stamps
    :param run_name: Usually the track name.
    """

    keys_and_times = np.array([(elem[1], elem[2]) for elem in training_data])
    time_diffs = np.diff(np.concatenate((np.array([0.0]), keys_and_times[:, 1])))
    training_directory = Path(r"E:\Trackmania Data\data") / Path(
        "{}_{}".format(run_name, time.strftime("%Y%m%d-%H%M%S")))
    (training_directory / Path("screenshots")).mkdir(parents=True)
    with (training_directory / Path("keys_timings.txt")).open("w") as text_file:
        for k, line in enumerate(keys_and_times):
            text_file.write("{0:5}     {1:20s}     {2:f}     {3:f}\n".format(k, str(line[0]), line[1], time_diffs[k]))

    image_paths = []
    for k, _ in enumerate(training_data):
        image_paths.append(training_directory / Path("screenshots")
                           / Path("{:05d}_{}.png".format(k, str(keys_and_times[k][0]).strip("[]"))))

    pool = Pool(processes=3)
    pool.starmap(save_image, [(image_paths[k], e[0]) for k, e in enumerate(training_data)])
    pool.close()
    pool.join()

    # for k, e in enumerate(training_data):
    #     save_image(image_paths[k], e[0])


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
    # training_data = resizing_cropping(training_data)
    save(training_data, file_name)


if __name__ == '__main__':
    save_name = input("Enter run name: ")
    print("Keys: 't' to pause \n"
          "      'r' to resume \n"
          "      'o' and 'p' while paused to save and exit.")
    main(save_name)
