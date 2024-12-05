from typing import Callable
import cv2
import numpy as np
import datetime
from time import time


def nothing(frame: np.ndarray) -> np.ndarray:
    return frame


def run_camera_app(filter: Callable[[np.ndarray], np.ndarray] = nothing, camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        raise SystemExit
    
    total_frames = 0
    starting_time = time()
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to read from the camera.")
                break
            
            total_frames += 1
            
            frame = filter(frame)
            cv2.imshow('Live Camera Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                now = datetime.datetime.now()
                cv2.imwrite(f'captured_images/camera_capture{now}.png', frame)
                print(f"Image saved as 'captured_images/camera_capture{now}.png'")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print('Average number of fps:', round(total_frames / (time() - starting_time), 1))
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_app()
