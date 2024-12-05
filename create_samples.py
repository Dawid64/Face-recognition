import sys
import os
import cv2
import pandas as pd


def create_samples(num_samples, input_dir, output_dir='samples'):
    dataframe = []
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    iterator = 0
    for path in os.listdir(input_dir):
        if iterator == num_samples:
            break
        image = cv2.imread(f'{input_dir}/{path}')
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            continue
        x, y, w, h = faces[0]
        ending = path.split('.')[-1]
        cv2.imwrite(f'{output_dir}/{iterator}.{ending}', image)
        dataframe.append([f'{output_dir}/{iterator}.{ending}', x, y, w, h])
        iterator += 1
    dataframe = pd.DataFrame(dataframe, columns=['image_path', 'x', 'y', 'w', 'h'])
    dataframe.to_csv(f'{output_dir}/samples.csv', index=False)


if __name__ == '__main__':
    num_samples = int(sys.argv[1])
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]
    create_samples(num_samples, input_dir, output_dir)
