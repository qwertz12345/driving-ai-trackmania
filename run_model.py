from ast import literal_eval
from time import sleep, time
from pathlib import Path
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from keras_applications.mobilenet_v2 import relu6
from re import findall
import cv2


from directkeys import PressKey, ReleaseKey, W, A, S, D
from getkeys import key_check
from grabscreen import grab_screen
from training import custom_accs


def main(checkpoint_path, show_timings=False):
    # base_data_dir = r"E:\Trackmania Data"
    screen = grab_screen(region=(8, 200, 800, 430), window_title="TrackMania Nations Forever")
    checkpoint_path = Path(checkpoint_path)
    run_path = checkpoint_path.parent.parent
    # batch_size = int(findall(r"\d\d", run_path.name.split(".")[1])[0])

    with open(run_path / "class_indices.txt", "r") as f:
        line = f.readline()
    class_indices = literal_eval(line)
    get_custom_objects().update(
        {
         "A_class_accuracy": custom_accs(class_indices)["A_class_accuracy"],
         "AW_class_accuracy": custom_accs(class_indices)["AW_class_accuracy"],
         'relu6': relu6
         }
    )
    model = load_model(checkpoint_path)

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 240, 100)

    active_mode = False
    while True:
        if show_timings:
            start_time = time()

        screenshot = screen.getFrame()

        screenshot = np.array(
            Image.fromarray(screenshot[..., ::-1]).crop(box=(0, 0, 800, 320)).convert("L").resize((50, 50), Image.LANCZOS)
        )

        # cv2.imshow("image", screenshot)
        # cv2.waitKey(1)

        prediction = model.predict(np.expand_dims(np.expand_dims(screenshot / 255, axis=2), axis=0))
        # print(prediction)

        keys_dict = {'A': 0, 'AW': 1, 'D': 2, 'DW': 3, 'W': 4}
        inv_keys_dict = {v: k for k, v in keys_dict.items()}
        key_pred_dict = {inv_keys_dict[k]: e for k, e in enumerate(prediction[0])}

        to_print = ""
        for key, val in key_pred_dict.items():
            to_print += "{}: {:3.1f} | ".format(key, val)
        print(to_print, end="\r", flush=True)

        if active_mode:
            ReleaseKey(W)
            ReleaseKey(S)
            ReleaseKey(A)
            ReleaseKey(D)

        max_pred = max(key_pred_dict.values())
        for keys, pred in key_pred_dict.items():
            if pred == max_pred:
                if active_mode:
                    if "W" in keys:
                        PressKey(W)
                    if "A" in keys:
                        PressKey(A)
                    if "D" in keys:
                        PressKey(D)
                    if "S" in keys:
                        PressKey(S)
                # print(keys)
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
                if "Z" in ctrl_keys:
                    if active_mode:
                        active_mode = False
                        print("Passive mode")
                    else:
                        active_mode = True
                        print("Active mode")
                if "O" and "P" in ctrl_keys:
                    ReleaseKey(W)
                    ReleaseKey(S)
                    ReleaseKey(A)
                    ReleaseKey(D)
                    screen.clear()
                    quit(2)
                sleep(0.5)
                ctrl_keys = key_check()
            screen = grab_screen(region=(8, 200, 800, 430), window_title="TrackMania Nations Forever")

        if show_timings:
            print(time() - start_time)


if __name__ == '__main__':
    main(
        r"E:\Trackmania Data\runs\run_mobilenetV2augaug-True_batch_size32-start_time20180730-032337\checkpoints\checkpoint.14-0.74.hdf5"
    )
