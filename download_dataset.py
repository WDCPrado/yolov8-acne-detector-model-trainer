from roboflow import Roboflow
rf = Roboflow(api_key="9lYa5F0qcJFfv6J6063z")
project = rf.workspace("zahra-oy1dk").project("jerawat-cn06q")
version = project.version(4)
dataset = version.download("yolov8")
