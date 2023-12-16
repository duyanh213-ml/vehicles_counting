"""
===========================================================================================================
Latest update: 12-07-23 
@author: anhnd
===========================================================================================================


we will use YOLOv8 from a pretrained model which is available in Ultralytics
(https://www.ultralytics.com/) to train our custom dataset. Our dataset consists 
of 1000 images taken from a camera mounted on a highway, of which we use 768 images 
for model training. The remaining amount is for testing.


"""


from ultralytics import YOLO
import os
from config import *



# This function will create a yml file which would be need for training our model 
def create_yml_config(train_path_imgs, val_path_imgs, yml_path):
    content = f"train: {train_path_imgs}\nval: {val_path_imgs}\nnc: 1\nnames: ['vehicles']"
    
    with open(yml_path, encoding="utf-8", mode="w") as f:
        f.write(content)
    print("config.yaml has been created!!")
    

if __name__ == "__main__":
    
    # Training pipeline
    create_yml_config(train_path_imgs=TRAIN_PATH_IMGS,
                      val_path_imgs=VAL_PATH_IMGS,
                      yml_path=YML_PATH)
    
    # The yolov8n is the pretrained-model
    model = YOLO('yolov8n.pt')
    
    # After running it, a folder which named "runs" will be created automatically
    # to save our training, evaluation results and our best model too. 
    model.train(data=YML_PATH, epochs=EPOCHS, batch=BATCH)
    
    