import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from face_recognition.utils import draw_bounding_box
import random

def random_transform(image, x, y, width, height, original_width, original_height):
    if random.choice([True, False]):
        image = cv2.flip(image, 1)
        x = original_width - x - width 

    angle = random.uniform(-15, 15)
    matrix = cv2.getRotationMatrix2D((original_width / 2, original_height / 2), angle, 1)
    image = cv2.warpAffine(image, matrix, (original_width, original_height))

    if random.choice([True, False]):
        ksize = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    return image, x, y, width, height

def data_generator(csv_path, batch_size=32, image_size=(128, 128)):
    data = pd.read_csv(csv_path)
    while True:
        for start in range(0, len(data), batch_size):
            batch_data = data.iloc[start:start + batch_size]
            images = []
            labels = []
            for _, row in batch_data.iterrows():
                image_path = row['image_path']
                image = cv2.imread(image_path)
                if image is None:
                    continue
                original_height, original_width = image.shape[:2]

                x = float(row['x'])
                y = float(row['y'])
                width = float(row['w'])
                height = float(row['h'])
                image, x, y, width, height = random_transform(
                    image, x, y, width, height, original_width, original_height
                )
                image = cv2.resize(image, image_size) / 255.0
                normalized_x = x / original_width
                normalized_y = y / original_height
                normalized_width = width / original_width
                normalized_height = height / original_height

                images.append(image)
                labels.append([normalized_x, normalized_y, normalized_width, normalized_height])
            yield np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)


class Model:
    def __init__(self):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(256, (3, 3), activation='relu'),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(4)
        ])
        self.model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['accuracy'])
        self.history = None

    def train(self, csv_path='samples/samples.csv', num_epochs=10, batch_size=32):
        image_size = (128, 128)
        train_generator = data_generator(csv_path, batch_size, image_size)
        steps_per_epoch = len(pd.read_csv(csv_path)) // batch_size
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs
        )
        self.history = history
        return history
    
    def _preprocess_image(self, image:np.ndarray, target_size=(128, 128)):
        original_height, original_width = image.shape[:2]
        resized_image = cv2.resize(image, target_size)
        normalized_image = resized_image / 255.0
        return normalized_image, original_width, original_height
    
    def detect_faces(self, image:np.ndarray):
        image, original_width, original_height = self._preprocess_image(image)
        input_image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(input_image, verbose=0)[0]
        x = int(prediction[0] * original_width)
        y = int(prediction[1] * original_height)
        width = int(prediction[2] * original_width)
        height = int(prediction[3] * original_height)
        return x, y, width, height
    
    def load(self, model_path='face_recognition/model/face_detection_model.h5'):
        self.model = load_model(model_path)
        
    def save(self, model_path='face_recognition/model/face_detection_model.h5'):
        self.model.save(model_path)
    
    def get_filter(self):
        def face_rect(frame):
            bbox = self.detect_faces(frame)
            return draw_bounding_box(frame, bbox)
        return face_rect
    
    def __str__(self):
        return f"Model: {self.model}, Model Type: {self.model_type}, Model Data: {self.model_data}"
    