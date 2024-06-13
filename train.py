from ultralytics import YOLO

# Cargar el modelo desde el archivo de configuración
v = 3
data_folder=f"yolov8-acne-detection-{v}"
data = f"{data_folder}/data.yaml" #carpeta del dataset


 # Entrenar el modelo
model = YOLO("yolov8n.pt")  # Cargar un modelo preentrenado 
results = model.train(data=data, 
                        epochs=10, 
                        imgsz=640, 
                        device=0)