import time

import cv2

from grabscreen import grab_screen

# import numpy as np

# final_image = np.zeros((80, 60, 3), dtype=np.int)

time.sleep(3)
for i in range(10):
    # screen = grab_screen(region=(590, 485, 635, 480 + 30))
    screen = grab_screen(region=(0, 0, 640, 480 + 40))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    speed_image = screen[485:510, 590:635]
    screen_cropped = screen[75:465, 10:630]
    screen_cropped_resized = cv2.resize(screen_cropped, (62, 39), 0, 0, 0, cv2.INTER_AREA)
    speed_image_resized = cv2.resize(speed_image, (18, 10), 0, 0, 0, cv2.INTER_AREA)
    combined = screen_cropped_resized
    combined[-len(speed_image_resized):, int(len(screen_cropped_resized[1]) / 2 - len(speed_image_resized[1]) / 2):int(
        len(screen_cropped_resized[1]) / 2 + len(speed_image_resized[1]) / 2)] = speed_image_resized
    cv2.imwrite("whole_screen-{}.png".format(i), screen)
    cv2.imwrite("speed-{}.png".format(i), speed_image)
    cv2.imwrite("cropped-{}.png".format(i), screen_cropped)
    cv2.imwrite("resized-{}.png".format(i), screen_cropped_resized)
    cv2.imwrite("speed_resized-{}.png".format(i), speed_image_resized)
    cv2.imwrite("combined-{}.png".format(i), combined)
    time.sleep(0.1)
