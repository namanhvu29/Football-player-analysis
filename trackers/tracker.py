from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd  
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox', []) for x in ball_positions ]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:{"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_track(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
                return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players": [],
            "ball": [],
            "referee": []
        }

        for frame_num, detection in enumerate(detections):
            cls_name = detection.names
            cls_name_inv = {v:k for k,v in cls_name.items()}

            #Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Convert Goalkeeper to Player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_name[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_name_inv['player']
                    detection_supervision.data['class_name'][object_ind] = "player"

            #Track Object
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['ball'].append({})
            tracks['referee'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_name_inv['player']:
                    tracks['players'][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_name_inv['referee']:
                    tracks['referee'][frame_num][track_id] = {"bbox": bbox}
                
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_name_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox": bbox}
                    
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame, 
            center=(x_center, y2), 
            axes=(int(width), int(0.35*width)), 
            angle=0, 
            startAngle=-45, 
            endAngle=235, 
            color=color, 
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        y1_rect = y2 - rectangle_height//2 + 15
        x2_rect = x_center + rectangle_width//2
        y2_rect = y2 + rectangle_height//2 + 15
        

        if track_id is not None:
            cv2.rectangle(
                frame,
                (x1_rect, y1_rect),
                (x2_rect, y2_rect),
                color=color,
                thickness=2,
                lineType=cv2.LINE_4,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_4,
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)
        
        triangle_point = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        cv2.drawContours(frame, [triangle_point], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_point], 0, (0,0,0), 2)
        return frame
        


    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            if frame_num < len(tracks["players"]):
                player_dict = tracks['players'][frame_num]
                ball_dict = tracks['ball'][frame_num]
                referee_dict = tracks['referee'][frame_num]
                

                #Draw Player
                for track_id, player in player_dict.items():
                    color = player.get("team_color", (0, 0, 255))
                    frame = self.draw_ellipse(frame, player['bbox'], color=color, track_id=track_id)

                    if player.get("has ball", False):
                        frame = self.draw_triangle(frame, player['bbox'], color=(0, 255, 0))

                #Draw Ball
                for track_id, ball in ball_dict.items():
                    frame = self.draw_ellipse(frame, ball['bbox'], color=(0, 0, 255), track_id=None)

                #Draw Referee
                for track_id, referee in referee_dict.items():
                    frame = self.draw_ellipse(frame, referee['bbox'], color=(255, 0, 0), track_id=None)

                #Draw ball
                for track_id, ball in ball_dict.items():
                    frame = self.draw_triangle(frame, ball['bbox'], color=(0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames