"""
===========================================================================================================
Latest update: 12-10-23 
===========================================================================================================

"""

# Import libraries
import cv2
from config import *


def is_touch_line(center_y, line_y, epsilon = 10):
    return abs(center_y - line_y) < epsilon
    

def implement_system(torch_model, video_path, origin_line_color=ORIGIN_LINE_COLOR, 
                     rec_color=REC_COLOR, rec_thickness=REC_THICKNESS, circle_radius=CIRCLE_RADIUS, 
                     circle_color=CIRCLE_COLOR, circle_thickness=CIRCLE_THICKNESS, 
                     modified_line_color=MODIFIED_LINE_COLOR, line_x0=LINE_X0, line_x1=LINE_X1, 
                     line_y0=LINE_Y0, line_thickness=LINE_THICKNESS, text_org=TEXT_ORG, 
                     text_font=TEXT_FONT, text_fontScale=TEXT_FONTSCALE, 
                     text_color=TEXT_COLOR, text_thickness=TEXT_THICKNESS):

    vide = cv2.VideoCapture(video_path)

    ret, frame = vide.read()
    count = 0

    car_counting = 0

    trackers_list = []
    flag = True

    while ret:
        laser_line_color = origin_line_color
        
        if count % 8 == 0:
            if len(trackers_list) != 0:
                trackers_list.clear()
                
            results = torch_model.predict(frame)
            for r in results:

                boxes = r.boxes.cpu().numpy()

                xyxys = boxes.xyxy

                for xyxy in xyxys:                
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), 
                                (int(xyxy[2]), int(xyxy[3])), 
                                color=rec_color, thickness=rec_thickness)
                    
                    center_X = int((xyxy[0] + xyxy[2]) / 2)
                    center_Y = int((xyxy[1] + xyxy[3]) / 2)
                    
                    width = int(xyxy[2] - xyxy[0])
                    height = int(xyxy[3] - xyxy[1])
                    
                    cv2.circle(frame, (center_X, center_Y), radius=circle_radius, 
                        color=circle_color, thickness=circle_thickness)
                    
                    
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
                
                    
                cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), 
                            color=rec_color, thickness=rec_thickness)
                
                cv2.circle(frame, (center_X, center_Y), radius=circle_radius, 
                        color=circle_color, thickness=circle_thickness)
                
                if is_touch_line(center_Y, 500):    
                    car_counting += 1
                    laser_line_color = modified_line_color
                    trackers_list.remove(tracker)
                    
                    
        cv2.line(frame, (line_x0, line_y0), (line_x1, line_y0), 
                color=laser_line_color, thickness=line_thickness)   

        frame = cv2.putText(frame, f'Number of vehicles: {car_counting}', org=text_org, 
                fontScale=text_fontScale, color=text_color, 
                thickness=text_thickness, fontFace=text_font)     
        cv2.imshow("Roadmap", frame)

        if (cv2.waitKey(1) & 0xff == ord("q")):
            break

        count += 1

        ret, frame = vide.read()
        
        flag = True
        
        
        
    cv2.destroyAllWindows()
    vide.release()
