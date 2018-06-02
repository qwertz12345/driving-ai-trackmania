import os

import cv2
import numpy as np

from getkeys import key_check
from getFrame import *


# TODO: SAVE TIMINGS TO SAVE FILE
# TODO: CHANGE FILE NAME PATTERN AND DEBUG!!!


def create_screenshot(framethread):
    screenshot_full = framethread.returnFrame()
    velocity = screenshot_full[-29:, -60:-7]
    screenshot = screenshot_full[200:400, 25:-25]
    screenshot = cv2.resize(screenshot, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    return screenshot, velocity


def stats(loop_times):
    loop_times = [t for t in loop_times if t != 0]
    m = "mean " + str(np.mean(loop_times)) + " seconds, "
    s = "std " + str(np.std(loop_times)) + " seconds"
    return m + s


def main(file_name):
    framethread = getFrameThread(8, 30, 800, 600, "TrackMania Nations Forever").start()
    # file_index = 1
    training_directory = os.path.abspath(r"E:\Trackmania Data\training_data_new")
    # while True:
    #     file_name = 'training_data-{}.npy'.format(file_index)
    #     if os.path.isfile(os.path.join(training_directory, file_name)):
    #         print('File exists, moving along', file_index)
    #         file_index += 1
    #     else:
    #         print('File does not exist, starting fresh!', file_index)
    #         break

    loop_times = []
    training_data = []
    paused = True
    _ = key_check()
    print("Press t to start")
    driving_keys = ("W", "A", "S", "D")
    last_time = time.time()

    while True:     # main loop
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print("Unpausing")
                time.sleep(1 / 2)
                last_time = time.time()
            else:
                print("Pausing")
                paused = True
                time.sleep(1 / 2)
        elif 'O' and 'P' in keys and paused:
            print("Saving and Stopping")
            # file_name = 'training_data-{}.npy'.format(file_index)
            np.save(os.path.join(training_directory, file_name), training_data)
            print(stats(loop_times))
            break
        elif 'M' in keys and paused:
            print(stats(loop_times))
            time.sleep(1 / 2)

        correct_driving_keys = True
        if not paused:
            for key in keys:
                if key not in driving_keys:
                    correct_driving_keys = False
            if len(keys) <= 2 and correct_driving_keys:
                screenshot, velocity = create_screenshot(framethread)
                training_data.append([screenshot, velocity, keys])
                if len(training_data) % 500 == 0:
                    print(len(training_data))

            new_time = time.time()
            current_loop_time = new_time - last_time
            fixed_loop_time = 0.035
            difference = fixed_loop_time - current_loop_time
            if difference > 0:
                time.sleep(difference)
            # loop_times.append(current_loop_time)
            loop_times.append(time.time() - last_time)  # actual loop times

            last_time = new_time

    framethread.stopNow()
    with open(os.path.join(training_directory, "stats.txt"), "w") as text_file:
        text_file.write(stats(loop_times))


if __name__ == '__main__':
    main("newest")
    # framet = getFrameThread(8, 30, 800, 600, "TrackMania Nations Forever").start()
    # time.sleep(1)
    # scr, vel = create_screenshot(framet)
    # cv2.imshow('image', scr)
    # cv2.waitKey(0)
    # cv2.imshow("2", vel)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # framet.stopNow()
