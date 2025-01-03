import cv2
from face_recognition import run_camera_app, Model

image = cv2.imread('samples/0.jpg')
model = Model()
model.train('samples/samples.csv', 5)
bbox = model.detect_faces(image)

run_camera_app(model.get_filter())