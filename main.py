from utils import read_video, save_video
from trackers import Tracker
from team_assigment.team_assign import TeamAssign
import cv2
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator


def main():
    #Read Video
    video_frames = read_video('./input_videos/test.mp4')


    #Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_track(video_frames, 
                                    read_from_stub=False, 
                                    stub_path='stubs/track_stubs.pkl')

    #get object position
    tracker.add_position_to_tracks(tracks)


    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, 
                                                                    read_from_stub=False, 
                                                                    stub_path='stubs/camera_movement.pkl')
    camera_movement_estimator._adjust_position_to_track(tracks, camera_movement_per_frame)

    #view transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    #Interpolate ball position
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    #speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    #Team player assign
    team_assign = TeamAssign()
    team_assign.assign_team_color(video_frames[0], 
                                tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assign.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assign.team_color[team]

    #Assign ball to player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assignned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        if assignned_player != -1:
            tracks['players'][frame_num][assignned_player]['has ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assignned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)


    #Draw output
    ##Draw object track
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ##Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    ##Draw speed and distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    
    #Save Video
    save_video(output_video_frames, './output_videos/output_video1.avi')

if __name__ == '__main__':
    main()