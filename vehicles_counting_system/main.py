"""
===========================================================================================================
Latest update: 12-10-23 
===========================================================================================================

This is the main file that implement our system

"""


# Import libraries
from ultralytics import YOLO
from config import MODEL_PATH, VIDEO1_PATH, VIDEO2_PATH, DEVICE
from utils import implement_system



if __name__ == "__main__":
    # Initalize the model
    torch_model = YOLO(MODEL_PATH).to(device=DEVICE)
    
    # You can choose video2 if you want
    video_path = VIDEO1_PATH
    
    # Implementing our vehicles couting system
    implement_system(torch_model=torch_model, video_path=video_path)