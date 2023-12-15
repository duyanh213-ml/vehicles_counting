"""
===========================================================================================================
Latest update: 11-30-23 
@author: anhnd
===========================================================================================================
"""

import os
import cv2

# Config for video path:
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = f"{ROOT_DIR}/video"

YOLO_PART1_PATH = f"{VIDEO_DIR}/for_yolo_part1.mp4"
YOLO_PART2_PATH = f"{VIDEO_DIR}/for_yolo_part2.mp4"
YOLO_PART3_PATH = f"{VIDEO_DIR}/for_yolo_part3.mp4"

VIDEO1_PATH = f"{VIDEO_DIR}/video1.mp4"
VIDEO2_PATH = f"{VIDEO_DIR}/video2.mp4"

# Config for cv2.putText()
FONT = cv2.FONT_HERSHEY_SIMPLEX
ORG = (50, 50)
FONT_SCALE = 1
COLOR = (255, 0, 0)
THICKNESS = 2

# Config for raw_image
RAW_IMAGE_PATH = f"{ROOT_DIR}/raw_images"

IMAGE_PART1_PATH = f"{RAW_IMAGE_PATH}/part1"
IMAGE_PART2_PATH = f"{RAW_IMAGE_PATH}/part2"
IMAGE_PART3_PATH = f"{RAW_IMAGE_PATH}/part3"

FILE_NAME_FORMAT = "image_"
TYPE_IMAGE = ".jpg"

# Config for the numbers of images that we will get in each video
LIMIT_IMAGE_NUMBER_LIST = [324, 600, 100]

# Ex: in yolo part 1: each 13 frames, we will get one and put into raw_images
STEP_SIZE_LIST = [13, 10, 15]

# The ratio of train:val = 75:25
RATIO_CONST = 4

# config for train-val path
DATASET_PATH = f"{ROOT_DIR}/dataset"

TRAIN_IMAGE_PATH = f"{DATASET_PATH}/train/images"
VAL_IMAGE_PATH = f"{DATASET_PATH}/val/images"

