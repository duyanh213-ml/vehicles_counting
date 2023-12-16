from ultralytics import YOLO
from config import MODEL_PATH, VIDEO1_PATH, VIDEO2_PATH, DEVICE
from utils import implement_system

if __name__ == "__main__":
    torch_model = YOLO(MODEL_PATH).to(device=DEVICE)
    
    video_path = VIDEO1_PATH
    
    implement_system(torch_model=torch_model, video_path=video_path)