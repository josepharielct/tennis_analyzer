from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import get_bbox_center, measure_distance
class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
    
    def detect_frame(self,frame):
        results = self.model.track(frame,persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0]) #entity id
            result = box.xyxy.tolist()[0] #bounding box
            object_class_id = box.cls.tolist()[0] #Entity classification 
            object_class_name = id_name_dict[object_class_id]
            if object_class_name == 'person':
                player_dict[track_id] = result

        return player_dict
    
    def detect_frames(self,frames,read_from_stub=False, stub_path=None):
        player_detections = []
        
        if read_from_stub is True and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections,f)

        return player_detections
    
    def draw_bbox(self,video_frames,player_detections):
        output_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw bbox
            for track_id, bbox in player_dict.items():
                x1,y1,x2,y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), (int(bbox[1] -10))),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
                frame = cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 2)
            output_frames.append(frame)

        return output_frames

    def filter_players(self, court_keypoints, player_detections):
        player_detections_frame_one = player_detections[0]
        chosen_players = self.choose_players(court_keypoints,player_detections_frame_one)
        filtered_player_detections =[]
        for player_dict in player_detections:
            filtered_playered_dict = {track_id:bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_playered_dict)
        return filtered_player_detections


    def choose_players(self, court_keypoints, player_dict):
        distances =[]
        for track_id, bbox in player_dict.items():
            player_center = get_bbox_center(bbox)

            #Calculate court vs player distance
            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i],court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id,min_distance))
            
        # Sort distance
        distances.sort(key=lambda x: x[1])
        #Get top 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players
