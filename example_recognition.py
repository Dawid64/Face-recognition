import cv2
from face_recognition.model.model import Model
from face_recognition import run_camera_app

image = cv2.imread('samples/0.jpg')
model = Model()
model.train('samples/samples.csv', 100)
bbox = model.detect_faces(image)

run_camera_app(model.get_filter())