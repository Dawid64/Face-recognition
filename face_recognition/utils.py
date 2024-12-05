from typing import Tuple
import numpy as np
import cv2


def draw_bounding_box(frame:np.ndarray, bbox:Tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = bbox
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return frame
