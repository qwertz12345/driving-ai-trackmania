import time

import numpy as np
from PIL import Image
from keras.models import load_model

from getFrame import *
from create_training_data_new import stats, create_screenshot
from getkeys import key_check
from recognize_numbers import recognize
from directkeys import PressKey, ReleaseKey, W, A, S, D


def get_velocity_binned(velo_image, velo_recognition_model):
    img_digits = [Image.new("RGB", (18, 32)) for _ in range(3)]
    img_digits[0].paste(Image.fromarray(velo_image[:, :18]))
    img_digits[1].paste(Image.fromarray(velo_image[:, 17:35]))
    img_digits[2].paste(Image.fromarray(velo_image[:, 35:53]))
    img_digits = [np.asarray(e) for e in img_digits]
    digits_with_confidence_value = recognize(np.array(img_digits) / 255, velo_recognition_model)
    velocity = int('{}{}{}'.format(digits_with_confidence_value[0][0], digits_with_confidence_value[1][0],
                                   digits_with_confidence_value[2][0]))

    bin_edges = np.array(np.mat(
        '1. 28.6875 56.375 84.0625 111.75 139.4375 167.125 194.8125 222.5 250.1875 277.875 305.5625 333.25 360.9375 388.625 416.3125 444')).flatten()
    number_bins = len(bin_edges) - 1
    bin_ind = np.digitize([velocity], bin_edges) - 1
    assert len(bin_ind) == 1
    for k, ind in enumerate(bin_ind):
        if ind == number_bins:
            bin_ind[k] -= 1
    velocity_binned_one_hot = []
    for ind in bin_ind:
        one_hot_vector = np.zeros(number_bins)
        one_hot_vector[ind] = 1
        velocity_binned_one_hot.append(one_hot_vector)

    return np.array(velocity_binned_one_hot)


def straight(suggestion):
    if not suggestion:
        PressKey(W)
    else:
        print("straight")


def left(suggestion):
    if not suggestion:
        PressKey(A)
    else:
        print("left")


def right(suggestion):
    if not suggestion:
        PressKey(D)
    else:
        print("right")


def reverse(suggestion):
    if not suggestion:
        PressKey(S)
    else:
        print("reverse")


def release_all_keys():
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)


def steer(n, suggestion):
    # [W, S, A, D]
    {0: straight, 1: reverse, 2: left, 3: right}[n](suggestion)


def main():
    model = load_model("all_data_changed model")
    velo_recognition_model = load_model(
        r"C:\Users\mivog\PycharmProjects\driving ai trackmania\velocity_recognition-100.h5")
    suggestion_mode = True
    print("Press t to pause, press r to resume, press o and p while paused to get stats")

    framethread = getFrameThread(8, 30, 800, 600, "TrackMania Nations Forever").start()
    time.sleep(0.5)

    _ = key_check()
    loop_times = []
    last_time = time.time()

    while True:
        keys = key_check()
        if 'T' in keys:
            print("Pausing")
            release_all_keys()
            time.sleep(1)
            b = False
            while 'R' not in keys:
                keys = key_check()
                if "O" in keys and "P" in keys and len(loop_times) > 1 and not b:
                    stats(loop_times)
                    b = True
                if "Z" in keys:
                    if suggestion_mode:
                        suggestion_mode = False
                        print("Suggestion mode deactivated")
                        time.sleep(1)
                    else:
                        release_all_keys()
                        suggestion_mode = True
                        print("Suggestion mode activated")
                        time.sleep(1)
            print("Unpausing")
            time.sleep(1)
            last_time = time.time()
        if "Z" in keys:
            if suggestion_mode:
                suggestion_mode = False
                print("Suggestion mode deactivated")
                time.sleep(1)
                last_time = time.time()
            else:
                release_all_keys()
                suggestion_mode = True
                print("Suggestion mode activated")
                time.sleep(1)

        screenshot, velocity = create_screenshot(framethread)
        screenshot = np.array(screenshot) / 255
        velo_bin_one_hot = get_velocity_binned(velocity, velo_recognition_model)

        prediction = model.predict([np.array([screenshot]), velo_bin_one_hot], batch_size=1)
        prediction = prediction[0]
        prediction = prediction[0]
        # print([int(round(p * 100)) for p in prediction])

        # prediction[prediction <= 0.5] = 0
        # prediction[prediction > 0.5] = 1
        if prediction[0] > 0.5 and prediction[0] > prediction[1]:
            prediction[0] = 1
            prediction[1] = 0
        elif prediction[1] > prediction[0] and prediction[1] > 0.25:
            prediction[0] = 0
            prediction[1] = 1
        else:
            prediction[0] = 0
            prediction[1] = 0

        if prediction[2] > 0.25 and prediction[2] > prediction[3]:
            prediction[2] = 1
            prediction[3] = 0
        elif prediction[3] > prediction[2] and prediction[3] > 0.25:
            prediction[3] = 1
            prediction[2] = 0
        else:
            prediction[2] = 0
            prediction[3] = 0

        if not suggestion_mode:
            release_all_keys()
        for i, pred in enumerate(prediction):
            if pred == 1:
                steer(i, suggestion_mode)

        if len(loop_times) > 50000:
            del loop_times[:-4000]
        new_time = time.time()
        loop_times.append(new_time - last_time)
        last_time = new_time


if __name__ == '__main__':
    main()
