import cv2
import numpy as np


def gray_scale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def find_countours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.drawContours(frame, contours, -1, (50, 50, 200), 3)
    return frame


def find_motion(frame):
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    if find_motion.last_frame is None:
        find_motion.last_frame = frame_blurred
        return np.zeros_like(frame)

    diff_frame = cv2.absdiff(frame_blurred, find_motion.last_frame)
    find_motion.last_frame = frame_blurred
    _, motion_mask = cv2.threshold(diff_frame, 50, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    return motion_mask


find_motion.last_frame = None


def find_light(frame):

    frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    _, motion_mask = cv2.threshold(frame_blurred, 220, 255, cv2.THRESH_BINARY)
    return motion_mask
