from ultralytics import YOLO
import cv2
import os


# Cargar el modelo YOLO preentrenado desde la última versión
model = YOLO("runs/detect/train3/weights/best.pt")

# Ruta de la imagen
image_path = "acne.png"

# Leer la imagen
frame = cv2.imread(image_path)

if frame is None:
    print("Error: No se puede abrir la imagen")
    exit()

# Realizar la predicción
results = model(frame)

print(results)
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
output_path = f"pred.png"
cv2.imwrite(output_path, frame)
print(f"Imagen guardada en {output_path}")
