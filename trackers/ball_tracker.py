from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
    
    def interpolate_ball_pos(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        
        df_ball_positions = pd.DataFrame(ball_positions, columns =['x1','y1','x2','y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions


    def detect_frame(self,frame):
        results = self.model.predict(frame,conf=0.2)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0] #bounding box
            ball_dict[1] = result

        return ball_dict
    
    def detect_frames(self,frames,read_from_stub=False, stub_path=None):
        ball_detections = []
        
        if read_from_stub is True and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections,f)

        return ball_detections
    
    def draw_bbox(self,video_frames,ball_detections):
        output_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw bbox
            for track_id, bbox in ball_dict.items():
                x1,y1,x2,y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), (int(bbox[1] -10))),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                frame = cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,255,0), 2)
            output_frames.append(frame)

        return output_frames
    
    def get_ball_hits(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns =['x1','y1','x2','y2'])
        df_ball_positions['mid_y'] = (df_ball_positions['y1']+df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        df_ball_positions['ball_hit'] = 0
        hit_frames_threshold = 25
        for i in range(1,len(df_ball_positions)-int(hit_frames_threshold*1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1]<0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1]>0

            if negative_position_change or positive_position_change:
                count = 0
                for change_frame in range(i+1, i+int(hit_frames_threshold*1.2)):
                    negative_position_change_next_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame+1]<0
                    positive_position_change_next_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame+1]>0

                    if negative_position_change and negative_position_change_next_frame:
                        count +=1
                    if positive_position_change and positive_position_change_next_frame:
                        count +=1
                if count > hit_frames_threshold-1:
                    df_ball_positions['ball_hit'].iloc[i]=1
        frames_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()
        return frames_ball_hits
