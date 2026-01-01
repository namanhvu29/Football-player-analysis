from utils import read_video, save_video
from trackers import Tracker
from team_assigment.team_assign import TeamAssign
import cv2
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner


def main():
    #Read Video
    video_frames = read_video('./input_videos/short_video.mp4')


    #Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_track(video_frames, 
                                    read_from_stub=False, 
                                    stub_path='stubs/track_stubs.pkl')
    
    #Interpolate ball position
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

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
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assignned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        if assignned_player != -1:
            tracks['players'][frame_num][assignned_player]['has ball'] = True


    #Draw output
    ##Draw object track
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    #Save Video
    save_video(output_video_frames, './output_videos/output_video1.avi')

if __name__ == '__main__':
    main()