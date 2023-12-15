"""
===========================================================================================================
Latest update: 11-30-23 
@author: anhnd
===========================================================================================================
"""


# Import libraries
import cv2
from config import *
from random import shuffle
import shutil

# This function will generate images which is cut from 3 "yolo-part" videos
# The total images will be 1024
def generate_images(yolo_part1_path, yolo_part2_path,
                    yolo_part3_path, limit_image_number_list, step_size_list,
                    image_part1_path, image_part2_path, image_part3_path, 
                    file_name_format, type_image):
    
    path_list = (yolo_part1_path, yolo_part2_path, yolo_part3_path)
    destination_list = (image_part1_path, image_part2_path, image_part3_path)

    # This var stores the total image number
    file_name_id = 1    

    for i, path in enumerate(path_list):

        vid = cv2.VideoCapture(path)

        # This counter uses to count each frame in video
        count = 0
        # This counter uses to count the "accepted" frame - which will be stored in raw_images folder
        # and it also stop the while loop when the image numbers is enough
        limit = 1

        while limit <= limit_image_number_list[i]:
            ret, frame = vid.read()
            
            if ret:     
                if count % step_size_list[i] == 0:
                    cv2.imwrite(f"{destination_list[i]}/{file_name_format}{file_name_id}{type_image}", frame)
                    file_name_id += 1
                    limit += 1
                
                count += 1
                
        print(f"...Create {limit - 1} images of part {i + 1} successfully!!!")
               
        vid.release()
        
    print(f"...Complete!!! the number of images is {file_name_id - 1}")


# This function uses to move files from a folder (source) to another one (destination)
# and it will serve for train_val_split
def move_file(files_to_move, source_folder, destination_folder):
    for file_name in files_to_move:
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        shutil.move(source_file, destination_file)
    
        
# This function will split dataset which is created in raw_images folder
# The ratio we will choose is stored with RATIO_CONST in config.py
def train_val_split(image_part1_path, image_part2_path, image_part3_path, 
                    ratio_const, train_image_path, val_image_path):
    
    destination_list = (image_part1_path, image_part2_path, image_part3_path)
    
    for part in destination_list:
        list_filename = os.listdir(part)
        shuffle(list_filename)
        
        split_index = len(list_filename) // ratio_const
        
        train_files = list_filename[split_index:]
        val_files = list_filename[:split_index]
        
        move_file(files_to_move=train_files, source_folder=part, destination_folder=train_image_path)
        move_file(files_to_move=val_files, source_folder=part, destination_folder=val_image_path)
        
    print("...Spliting train - validation successfully!!!")
        
        
        
        
    
        
if __name__ == "__main__":

    # For more detail about the const values, please see the config.py
    generate_images(yolo_part1_path=YOLO_PART1_PATH, yolo_part2_path=YOLO_PART2_PATH,
                    yolo_part3_path=YOLO_PART3_PATH, limit_image_number_list=LIMIT_IMAGE_NUMBER_LIST,
                    step_size_list=STEP_SIZE_LIST, image_part1_path=IMAGE_PART1_PATH,
                    image_part2_path=IMAGE_PART2_PATH, image_part3_path=IMAGE_PART3_PATH, 
                    file_name_format=FILE_NAME_FORMAT, type_image=TYPE_IMAGE)
    
    train_val_split(image_part1_path=IMAGE_PART1_PATH, image_part2_path=IMAGE_PART2_PATH,
                    image_part3_path=IMAGE_PART3_PATH, ratio_const=RATIO_CONST,
                    train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH)
