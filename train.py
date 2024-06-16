from ultralytics import YOLO

# Cargar el modelo desde el archivo de configuración
v = 4
data_folder = f"yolov8-acne-detection-{v}"
data = f"{data_folder}/data.yaml"  # carpeta del dataset


# Entrenar el modelo
model = YOLO("yolov8l.pt")  # Cargar un modelo preentrenado
results = model.train(data=data, epochs=20, imgsz=640, device=0)
