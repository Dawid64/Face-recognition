import cv2
def gray_scale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)