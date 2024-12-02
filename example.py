from face_recognition import run_camera_app, gray_scale
if __name__ == '__main__':
    run_camera_app(filter=gray_scale)