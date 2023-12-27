"""
===========================================================================================================
Latest update: 12-10-23 
===========================================================================================================
"""

# Import libraries
import os
import cv2

# Our vehicles_counting path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Best model path
MODEL_PATH = f"{ROOT_DIR}/runs/detect/train3/weights/best.pt"
DEVICE = "cuda"

# Path for our test video
VIDEO1_PATH = f"{ROOT_DIR}/video/video1.mp4"
VIDEO2_PATH = f"{ROOT_DIR}/video/video2.mp4"


# Config for params
# ====================================================

# This is for cv2.putText()
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX 
TEXT_ORG = (50, 50) 
TEXT_FONTSCALE = 2
TEXT_COLOR = (150, 200, 0) 
TEXT_THICKNESS = 2




# This is for cv2.line()
LINE_X0 = 100
LINE_X1 = 1200
LINE_Y0 = 500

LINE_THICKNESS = 3

ORIGIN_LINE_COLOR = (0, 0, 200)
MODIFIED_LINE_COLOR = (0, 150, 0)

# This is for cv2.circle() 
CIRCLE_RADIUS = 3
CIRCLE_COLOR = (0, 0, 255)
CIRCLE_THICKNESS = -1


# This is for cv2.rectangle()
REC_COLOR = (0, 255, 0)
REC_THICKNESS = 2 



