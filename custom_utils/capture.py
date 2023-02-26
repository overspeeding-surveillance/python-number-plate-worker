import os
import cv2
import uuid


def capture_plate(frame, x1, y1, x2, y2, filename):
    if not os.path.exists("../number_plates"):
        os.makedirs("../number_plates")

    roi = frame[y1: y2, x1:x2]
    path = "../number_plates/" + filename
    print("captured " + filename)
    cv2.imwrite(path, roi)
