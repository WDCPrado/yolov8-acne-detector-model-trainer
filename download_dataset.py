from roboflow import Roboflow

rf = Roboflow(api_key="9lYa5F0qcJFfv6J6063z")
project = rf.workspace("dermatologiaestoril").project("yolov8-acne-detection")
version = project.version(4)
dataset = version.download("yolov8")
