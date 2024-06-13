
from roboflow import Roboflow
rf = Roboflow(api_key="9lYa5F0qcJFfv6J6063z")
project = rf.workspace("chulalongkorn-university-vjyly").project("acne-jvorn")
version = project.version(23)
dataset = version.download("yolov8")
