import os

import cv2
import numpy as np

# for unprocessed data only!

training_directory = os.path.abspath(r"E:\Trackmania Data\training_data_new")
directory = os.path.abspath(r"E:\trackmania_imgs\new")

f = np.load(os.path.join(training_directory, "training_data-1_1.npy"))
print(len(f))

start = 1
h1, w1 = f[0, 0].shape[:2]
h2, w2 = f[0, 1].shape[:2]
# f = f[:1000]
for e in f:
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1, :3] = e[0]
    vis[:h2, w1:w1 + w2, :3] = e[1]
    cv2.imwrite(os.path.join(directory, str(start) + "_" + str(e[2]) + ".jpg"), vis)
    start += 1
