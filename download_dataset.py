from roboflow import Roboflow
rf = Roboflow(api_key="9lYa5F0qcJFfv6J6063z")
project = rf.workspace("zahra-oy1dk").project("acne-2bwmy")
version = project.version(1)
dataset = version.download("yolov8")
