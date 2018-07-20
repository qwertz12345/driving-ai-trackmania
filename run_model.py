from ast import literal_eval
from time import sleep
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects

from directkeys import PressKey, ReleaseKey, W, A, S, D
from getkeys import key_check
from grabscreen import grab_screen
from training import custom_accs


def main():
    screen = grab_screen(region=(8, 200, 800, 320), window_title="TrackMania Nations Forever")
    with open("class_indices.txt", "r") as f:
        line = f.readline()
    class_indices = literal_eval(line)
    # class_indices = {'A': 0, 'AW': 1, 'D': 2, 'DW': 3, 'W': 4, 'no_key': 5}
    get_custom_objects().update(
        {
         "A_class_accuracy": custom_accs(class_indices)["A_class_accuracy"],
         "AW_class_accuracy": custom_accs(class_indices)["AW_class_accuracy"]
         }
    )
    model = load_model(r"E:\Trackmania Data\checkpoints\checkpoint.18-0.13.hdf5")

    # diffs = []
    # _ = key_check()
    print("Press 'R' to start.")
    while "R" not in key_check():
        sleep(0.5)

    while True:
        screenshot = screen.getFrame()
        screenshot = np.array(
            Image.fromarray(screenshot[..., ::-1]).convert("L").resize((160, 64))
        )

        prediction = model.predict(np.expand_dims(np.expand_dims(screenshot, axis=2), axis=0))

        keys_dict = {'A': 0, 'AW': 1, 'D': 2, 'DW': 3, 'W': 4, 'no_key': 5}
        inv_keys_dict = {v: k for k, v in keys_dict.items()}
        key_pred_dict = {inv_keys_dict[k]: e for k, e in enumerate(prediction[0])}

        # to_print = ""
        # for key, val in key_pred_dict.items():
        #     to_print += "{}: {:3.1f} | ".format(key, val)
        # print(to_print)

        ReleaseKey(W)
        ReleaseKey(S)
        ReleaseKey(A)
        ReleaseKey(D)

        max_pred = max(key_pred_dict.values())
        for keys, pred in key_pred_dict.items():
            if pred == max_pred:
                if "W" in keys:
                    PressKey(W)
                if "A" in keys:
                    PressKey(A)
                if "D" in keys:
                    PressKey(D)
                if "S" in keys:
                    PressKey(S)
                print(keys)
                break

        ctrl_keys = key_check()
        if "T" in ctrl_keys:
            ReleaseKey(W)
            ReleaseKey(S)
            ReleaseKey(A)
            ReleaseKey(D)
            screen.clear()
            print("Paused")

            while "R" not in ctrl_keys:
                if "O" and "P" in ctrl_keys:
                    ReleaseKey(W)
                    ReleaseKey(S)
                    ReleaseKey(A)
                    ReleaseKey(D)
                    screen.clear()
                    quit(2)
                sleep(0.5)
                ctrl_keys = key_check()
            screen = grab_screen(region=(8, 200, 800, 320), window_title="TrackMania Nations Forever")


if __name__ == '__main__':
    main()
