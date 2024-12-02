from typing import Callable
import cv2
import numpy as np

def nothing(frame:np.ndarray) -> np.ndarray:
    return frame

def run_camera_app(filter: Callable[[np.ndarray], np.ndarray] = nothing, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        raise SystemExit
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to read from the camera.")
                break

            cv2.imshow('Live Camera Feed', filter(frame))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_app()