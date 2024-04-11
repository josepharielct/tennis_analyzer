import cv2 
import numpy as np
import sys
sys.path.append('../')
from utils import (
    convert_pixels_to_meters,
    convert_meters_to_pixels,
    get_foot_pos,
    get_closest_keypoint_idx,
    get_bbox_height,
    measure_xy_distance,
    get_bbox_center,
    measure_distance
)
import constants

class MiniCourt:
    def __init__(self, frame):
        self.rectangle_width = 250
        self.rectangle_height = 500
        self.buffer = 50
        self.court_padding = 20

        self.set_canvas_bg(frame)
        self.set_minicourt_drawing()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def set_canvas_bg(self,frame):
        frame = frame.copy()

        self.end_x = frame.shape[1]-self.buffer
        self.end_y = self.rectangle_height+self.buffer
        self.start_x = self.end_x - self.rectangle_width
        self.start_y =self.end_y - self.rectangle_height

    def set_minicourt_drawing(self):
        self.court_start_x = self.start_x + self.court_padding
        self.court_start_y = self.start_y + self.court_padding
        self.court_end_x = self.end_x - self.court_padding
        self.court_end_y = self.end_y - self.court_padding
        self.court_width = self.court_end_x - self.court_start_x
    def METERS2PIXELS(self, meters):
        return convert_meters_to_pixels(meters,
                                        constants.DOUBLE_LINE_WIDTH,
                                        self.court_width
                                        )
    def set_court_drawing_key_points(self):
        kps_drawing = [0]*28

        kps_drawing[0] , kps_drawing[1] = int(self.court_start_x), int(self.court_start_y)
        kps_drawing[2] , kps_drawing[3] = int(self.court_end_x), int(self.court_start_y)
        kps_drawing[4] = int(self.court_start_x)
        kps_drawing[5] = self.court_start_y + self.METERS2PIXELS(constants.HALF_COURT_LINE_HEIGHT*2)
        kps_drawing[6] = kps_drawing[0] + self.court_width
        kps_drawing[7] = kps_drawing[5] 
        kps_drawing[8] = kps_drawing[0] +  self.METERS2PIXELS(constants.DOUBLE_ALLY_DIFFERENCE)
        kps_drawing[9] = kps_drawing[1] 
        kps_drawing[10] = kps_drawing[4] + self.METERS2PIXELS(constants.DOUBLE_ALLY_DIFFERENCE)
        kps_drawing[11] = kps_drawing[5] 
        kps_drawing[12] = kps_drawing[2] - self.METERS2PIXELS(constants.DOUBLE_ALLY_DIFFERENCE)
        kps_drawing[13] = kps_drawing[3] 
        kps_drawing[14] = kps_drawing[6] - self.METERS2PIXELS(constants.DOUBLE_ALLY_DIFFERENCE)
        kps_drawing[15] = kps_drawing[7] 
        kps_drawing[16] = kps_drawing[8] 
        kps_drawing[17] = kps_drawing[9] + self.METERS2PIXELS(constants.NO_MANS_LAND_HEIGHT)
        kps_drawing[18] = kps_drawing[16] + self.METERS2PIXELS(constants.SINGLE_LINE_WIDTH)
        kps_drawing[19] = kps_drawing[17] 
        kps_drawing[20] = kps_drawing[10] 
        kps_drawing[21] = kps_drawing[11] - self.METERS2PIXELS(constants.NO_MANS_LAND_HEIGHT)
        kps_drawing[22] = kps_drawing[20] +  self.METERS2PIXELS(constants.SINGLE_LINE_WIDTH)
        kps_drawing[23] = kps_drawing[21] 
        kps_drawing[24] = int((kps_drawing[16] + kps_drawing[18])/2)
        kps_drawing[25] = kps_drawing[17] 
        kps_drawing[26] = int((kps_drawing[20] + kps_drawing[22])/2)
        kps_drawing[27] = kps_drawing[21] 
        self.kps_drawing=kps_drawing

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]
    def draw_rect_bg(self,frame):
        shapes = np.zeros_like(frame, np.uint8)
        # Draw Rectangle
        cv2.rectangle(shapes, (self.start_x,self.start_y), (self.end_x,self.end_y), (255,255,255), -1)
        out = frame.copy()
        alpha=0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame,alpha,shapes,1-alpha,0)[mask]
        return out
    def draw_court(self,frame):
        for i in range(0, len(self.kps_drawing),2):
            x = int(self.kps_drawing[i])
            y = int(self.kps_drawing[i+1])
            cv2.circle(frame, (x,y),5,(0,0,255),-1)

        for line in self.lines:
            start_point = (int(self.kps_drawing[line[0]*2]), int(self.kps_drawing[line[0]*2+1]))
            end_point = (int(self.kps_drawing[line[1]*2]), int(self.kps_drawing[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0,0,0), 2)

        net_start_point = (self.kps_drawing[0], int((self.kps_drawing[1] + self.kps_drawing[5])/2))
        net_end_point = (self.kps_drawing[2], int((self.kps_drawing[1] + self.kps_drawing[5])/2))
        cv2.line(frame, net_start_point,net_end_point, (255,0,0), 2)
        return frame
    
    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_rect_bg(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point(self):
        return (self.court_start_x,self.court_start_y)
    def get_width(self):
        return self.court_width
    def get_kps(self):
        return self.kps_drawing
    def get_minicourt_coor(self, object_pos, closest_kps, closest_kps_idx, player_height_pixels,player_height_meters):
        #get distance between closes keypoints
        distance_from_kps_x_pixels, distance_from_kps_y_pixels = measure_xy_distance(object_pos, closest_kps)
        #convert pixel to meters
        distance_kps_x_meters = convert_pixels_to_meters(distance_from_kps_x_pixels,player_height_meters,player_height_pixels)
        distance_kps_y_meters = convert_pixels_to_meters(distance_from_kps_y_pixels,player_height_meters,player_height_pixels)
        #convert to mini court coor
        minicourt_x_distance = self.METERS2PIXELS(distance_kps_x_meters)
        minicourt_y_distance = self.METERS2PIXELS(distance_kps_y_meters)

        closest_minicourt_kps = (self.kps_drawing[closest_kps_idx*2],
                                 self.kps_drawing[closest_kps_idx*2+1]
                                )
        minicourt_player_pos = (
            closest_minicourt_kps[0] + minicourt_x_distance,
            closest_minicourt_kps[1] + minicourt_y_distance
        )
        return minicourt_player_pos

    def BBOX2MINICOURT(self, player_boxes, ball_boxes, ori_kps):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS
        }

        output_player_box = []
        output_ball_box = []
        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_bbox_center(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key = lambda x: measure_distance(ball_position, get_bbox_center(player_bbox[x])))
            


            output_player_bbox_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_pos = get_foot_pos(bbox)

                #Get nearest kps in pixels
                closest_kps_idx = get_closest_keypoint_idx(foot_pos, ori_kps, [0,2,12,13])
                closest_kps = (ori_kps[closest_kps_idx*2], ori_kps[closest_kps_idx*2+1])
                #Player height in pixels
                frame_index_min = max(0,frame_num-20)
                frame_index_max = min(len(player_boxes), frame_num+50)
                bbox_height_pixels = [get_bbox_height(player_boxes[i][player_id]) for i in range(frame_index_min, frame_index_max)]
                max_player_height_pixels = max(bbox_height_pixels)

                #Convert closes kps to meters
                minicourt_player_pos = self.get_minicourt_coor(foot_pos,
                                                               closest_kps,
                                                               closest_kps_idx,
                                                               max_player_height_pixels,
                                                               player_heights[player_id]
                                                               )
                output_player_bbox_dict[player_id] = minicourt_player_pos

                if closest_player_id_to_ball == player_id:
                    #Get nearest kps in pixels
                    closest_kps_idx = get_closest_keypoint_idx(ball_position, ori_kps, [0,2,12,13])
                    closest_kps = (ori_kps[closest_kps_idx*2], ori_kps[closest_kps_idx*2+1])
                    minicourt_player_pos = self.get_minicourt_coor(ball_position,
                                                               closest_kps,
                                                               closest_kps_idx,
                                                               max_player_height_pixels,
                                                               player_heights[player_id]
                                                               )
                    output_ball_box.append({1: minicourt_player_pos})
            output_player_box.append(output_player_bbox_dict)
        return output_player_box, output_ball_box
    
    def draw_points_on_mini_court(self,frames,positions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames
