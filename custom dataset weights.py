import multiprocessing
from ultralytics import YOLO

if __name__ == "__main__":
    multiprocessing.freeze_support()
    model = YOLO(r"C:\Users\Aya\Downloads\DATASET\best.pt")
    results = model(source=0, show=True, save=True)
