from ultralytics import YOLO
import cv2
import os

# Función para encontrar la última versión disponible en la carpeta de modelos
def get_latest_model(models_dir):
    versions = []
    for filename in os.listdir(models_dir):
        if filename.startswith("bestv") and filename[5:-3].isdigit() and filename.endswith(".pt"):
            versions.append(int(filename[5:-3]))
    if versions:
        latest_version = max(versions)
        return f"bestv{latest_version}.pt", latest_version
    else:
        raise FileNotFoundError("No se encontraron versiones en la carpeta de modelos.")

# Definir la ruta del directorio de modelos
models_dir = "models"

# Obtener la última versión
latest_model, latest_version = get_latest_model(models_dir)

# Ruta completa del último modelo
latest_model_path = os.path.join(models_dir, latest_model)

# Cargar el modelo YOLO preentrenado desde la última versión
model = YOLO(latest_model_path)

# Ruta de la imagen
image_path = "images_test/acne.png"

# Leer la imagen
frame = cv2.imread(image_path)

if frame is None:
    print("Error: No se puede abrir la imagen")
    exit()

# Realizar la predicción
results = model.predict(source=frame, save=False, conf=0.5)

# Dibujar las predicciones en la imagen
for result in results:
    boxes = result.boxes  # Obtener las cajas detectadas
    for box in boxes:
        # Convertir coordenadas de la caja a enteros
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]  # Confianza de la predicción
        cls = box.cls[0]  # Clase de la predicción

        # Dibujar la caja y la etiqueta en la imagen
        label = f"{model.names[int(cls)]}: {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Guardar la imagen con las predicciones
output_path = f"images_test/acne_with_predictions_v{latest_version}.png"
cv2.imwrite(output_path, frame)
print(f"Imagen guardada en {output_path}")
