import shutil

source_path = '/root/projects/yolov8-acne-detector-model/logs/models/bestv1.pt'
destination_path = '/root/projects/api-yolov8-acne-detection/models/bestv1.pt'

shutil.copyfile(source_path, destination_path)