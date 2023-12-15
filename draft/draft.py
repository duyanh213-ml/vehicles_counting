import os
import cv2
from ultralytics import YOLO


MODEL_PATH = "/home/anhnd/Workspace/vehicles_counting/draft/tool.pt"

VIDEO = "/home/anhnd/Workspace/vehicles_counting/video/video1.mp4"


def get_box_info(box):
    (x, y, w, h) = [int(v) for v in box]
    center_X = int((x + (x + w)) / 2.0)
    center_Y = int((y + (y + h)) / 2.0)
    return x, y, w, h, center_X, center_Y


def is_touch_line(center_y, line_y, epsilon = 3):
    return abs(center_y - line_y) < epsilon
    




# Run the model
torch_model = YOLO(MODEL_PATH).to(device="cuda")




vide = cv2.VideoCapture(VIDEO)

ret, frame = vide.read()
count = 0

car_counting = 0

trackers_list = []
flag = True


# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
# org 
org = (50, 50) 
# fontScale 
fontScale = 1 
# Blue color in BGR 
color = (255, 255, 0) 
# Line thickness of 2 px 
thickness = 2

while ret:
    laser_line_color = (0, 0, 200)
    
    if count % 10 == 0:
        if len(trackers_list) != 0:
            trackers_list.clear()
            
        results = torch_model.predict(frame)
        for r in results:

            boxes = r.boxes.cpu().numpy()

            xyxys = boxes.xyxy

            for xyxy in xyxys:                
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), 
                              (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), thickness=2)
                center_X = int((xyxy[0] + xyxy[2]) / 2)
                center_Y = int((xyxy[1] + xyxy[3]) / 2)
                
                width = int(xyxy[2] - xyxy[0])
                height = int(xyxy[3] - xyxy[1])
                
                cv2.circle(frame, (center_X, center_Y), 3, (0, 0, 255), -1)
                
                if is_touch_line(center_Y, 500):
                    laser_line_color = (0, 150, 0)
                    car_counting += 1
                
                tracker = cv2.TrackerKCF.create()
                tracker.init(frame, (int(xyxy[0]), int(xyxy[1]), width, height))
                
                trackers_list.append(tracker)
        flag = False
                
    if (flag): 
        for tracker in trackers_list:
            _, select_box = tracker.update(frame)
                
            x0 = int(select_box[0])
            y0 = int(select_box[1])
            w = int(select_box[2])
            h = int(select_box[3])
                
            center_X = int(x0 + w / 2)
            center_Y = int(y0 + h / 2)
            
                
            cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), thickness=2)
            cv2.circle(frame, (center_X, center_Y), 3, (0, 0, 255), -1)
            
            if is_touch_line(center_Y, 500):    
                car_counting += 1
                laser_line_color = (0, 150, 0)
                
                
    cv2.line(frame, (100, 500), (1200, 500), laser_line_color, 3)   
    # Using cv2.putText() method 
    frame = cv2.putText(frame, f'Number of vehicles: {car_counting}', org, font, fontScale, color, thickness, cv2.LINE_AA)     
    cv2.imshow("Roadmap", frame)
    

   


    if (cv2.waitKey(1) & 0xff == ord("q")):
        break

    count += 1

    ret, frame = vide.read()
    
    flag = True
    
    
    
cv2.destroyAllWindows()
vide.release()
