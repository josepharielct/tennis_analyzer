from utils import (read_video,
                   save_video,
                   measure_distance,
                   convert_pixels_to_meters,
                   draw_player_stats)
import constants
from trackers import PlayerTracker, BallTracker
from courtline_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
from copy import deepcopy
def main():
    # Read video as frames
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/player_detections.pkl')
    # Detect Balls
    ball_tracker = BallTracker(model_path='models/yolov5_best.pt')
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/ball_detections.pkl')    
    
    ball_detections = ball_tracker.interpolate_ball_pos(ball_detections)

    #Detect courtline
    court_model_path = 'models/keypoints_model.pth'
    courtline_detector = CourtLineDetector(court_model_path)
    court_keypoints = courtline_detector.predict(video_frames[0])

    # choose players
    player_detections = player_tracker.filter_players(court_keypoints,player_detections)
    #MiniCourt
    mini_court = MiniCourt(video_frames[0])

    #Detect ball hits
    ball_hits = ball_tracker.get_ball_hits(ball_detections)

    #Convert positions to minicourt positions
    player_minicourt_detections, ball_minicourt_detections = mini_court.BBOX2MINICOURT(player_detections,ball_detections, court_keypoints)
    
    player_stats = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0, #can get average shot speed
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0, 
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0
    }]

    for ball_hit_ind in range(len(ball_hits)-1):
        start_frame = ball_hits[ball_hit_ind]
        end_frame = ball_hits[ball_hit_ind+1]
        ball_hit_time_sec = (end_frame-start_frame)/24 #Since we use 24fps

        #Distance covered
        distance_covered_by_ball_pixels = measure_distance(ball_minicourt_detections[start_frame][1],
                                            ball_minicourt_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixels_to_meters(distance_covered_by_ball_pixels,
                                                                       constants.DOUBLE_LINE_WIDTH,
                                                                       mini_court.get_width())
        ball_speed = (distance_covered_by_ball_meters/ball_hit_time_sec) *3.6

        #PLayer who hit the ball
        player_positions = player_minicourt_detections[start_frame]
        player_who_hit_ball = min(player_positions.keys(), key = lambda player_id: measure_distance(player_positions[player_id],
                                                                                               ball_minicourt_detections[start_frame][1]))
        
        #opponent player speed
        opponent_player_id = 1 if player_who_hit_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_minicourt_detections[start_frame][opponent_player_id],
                                                               player_minicourt_detections[end_frame][opponent_player_id])

        distance_covered_by_opponent_meters = convert_pixels_to_meters(distance_covered_by_opponent_pixels,
                                                                       constants.DOUBLE_LINE_WIDTH,
                                                                       mini_court.get_width())

        speed_of_opponent = (distance_covered_by_opponent_meters/ball_hit_time_sec) *3.6
        
        current_player_stats = deepcopy(player_stats[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_who_hit_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_who_hit_ball}_total_shot_speed'] += ball_speed
        current_player_stats[f'player_{player_who_hit_ball}_last_shot_speed'] = ball_speed

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats.append(current_player_stats)
    
    player_stats_df = pd.DataFrame(player_stats)
    frames_df = pd.DataFrame({'frame_num':list(range(len(video_frames)))})
    player_stats_df = pd.merge(frames_df, player_stats_df, on = 'frame_num', how='left')
    player_stats_df = player_stats_df.ffill()

    player_stats_df['player_1_average_shot_speed'] = player_stats_df['player_1_total_shot_speed']/player_stats_df['player_1_number_of_shots']
    player_stats_df['player_2_average_shot_speed'] = player_stats_df['player_2_total_shot_speed']/player_stats_df['player_2_number_of_shots']
    player_stats_df['player_1_average_player_speed'] = player_stats_df['player_1_total_player_speed']/player_stats_df['player_2_number_of_shots']
    player_stats_df['player_2_average_player_speed'] = player_stats_df['player_2_total_player_speed']/player_stats_df['player_1_number_of_shots']



    
    #Draw output
    ## Draw player,ball,court bbox
    output_video_frames = player_tracker.draw_bbox(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bbox(output_video_frames, ball_detections)
    output_video_frames = courtline_detector.draw_kps_video(output_video_frames,court_keypoints)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_minicourt_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_minicourt_detections, (0,255,255))
    output_video_frames = draw_player_stats(output_video_frames, player_stats_df)
    ## Draw frame number in top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    #Output Video
    save_video(output_video_frames,"output_videos/output_video.avi")

if __name__ == "__main__":
    main()