"""
===========================================================================================================
Latest update: 12-07-23 
===========================================================================================================
"""


# Import libraries
import os

# This code will get the absolute path of the /vehicles_counting, you don't need
# to change the directory mannually.
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The dir of training and validation images respectively, use for create config.yaml
TRAIN_PATH_IMGS = f"{ROOT_PATH}/dataset/train/images"
VAL_PATH_IMGS = f"{ROOT_PATH}/dataset/val/images"

# Directory of the config.yaml
YML_PATH = f"{ROOT_PATH}/train/config.yaml"

# Hyper-params for training our model
EPOCHS = 80
BATCH = 32



    
    
    