import os
import shutil
from ultralytics import YOLO

# Función para encontrar la última versión y generar la nueva versión
def get_next_version(logs_dir):
    versions = []
    for dirname in os.listdir(logs_dir):
        if dirname.startswith("v") and dirname[1:].isdigit():
            versions.append(int(dirname[1:]))
    if versions:
        latest_version = max(versions)
        return f"v{latest_version + 1}"
    else:
        return "v1"

# Definir la ruta del directorio de logs
logs_dir = "logs"
models_dir = "models"

# Crear el directorio de modelos si no existe
os.makedirs(models_dir, exist_ok=True)

# Obtener la siguiente versión
version = get_next_version(logs_dir)

# Ruta donde se guardará la nueva versión
ruta = os.path.join(logs_dir, version)
best_path = os.path.join(logs_dir, version, "weights", "best.pt")

# Cargar el modelo desde el archivo de configuración
data = "acne-1/data.yaml"
model = YOLO("yolov8l.pt")  # Cargar un modelo preentrenado

# Entrenar el modelo
results = model.train(data=data, 
                      epochs=20, 
                      imgsz=640, 
                      project=logs_dir,
                      name=version,
                      exist_ok=True, 
                      save=True, 
                      save_txt=True, 
                      save_conf=True,
                      device=[0])

# Guardar el mejor modelo entrenado en la carpeta models con la versión en el nombre
new_best_path = os.path.join(models_dir, f"best{version}.pt")
shutil.copy(best_path, new_best_path)

print(f"Modelo entrenado y guardado en la ruta: {ruta}")
print(f"El mejor modelo se ha guardado en: {new_best_path}")
