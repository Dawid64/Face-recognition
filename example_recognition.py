from face_recognition import run_camera_app, Model

model = Model()
model.train('samples/samples.csv', 200)
model.save()

run_camera_app(model.get_filter())