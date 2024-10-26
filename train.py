from ultralytics import YOLO
import torch
import torchvision
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()  # Añadir esto para manejar la inicialización de procesos

    print(f"PyTorch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    data = "dataset/data.yaml"
    model = YOLO("yolo11n.pt")  # Cargar un modelo preentrenado

    # Verificar si CUDA está disponible
    cuda_available = torch.cuda.is_available()
    print(f"CUDA disponible: {cuda_available}")

    if not cuda_available:
        print(
            "ADVERTENCIA: No se detectó GPU. El entrenamiento en CPU será significativamente más lento."
        )
        print(
            "Se recomienda usar una GPU NVIDIA compatible con CUDA para el entrenamiento."
        )
        # Opcional: permitir al usuario decidir si continuar
        respuesta = input("¿Desea continuar con CPU? (s/n): ")
        if respuesta.lower() != "s":
            exit()

    # Modificar la selección del dispositivo
    device = torch.device("cuda" if cuda_available else "cpu")

    # Ajustar los parámetros de entrenamiento para CPU
    batch_size = 2 if not cuda_available else 8  # Reducir el batch size para CPU
    results = model.train(
        data=data, epochs=20, imgsz=640, device=device, batch=batch_size
    )
