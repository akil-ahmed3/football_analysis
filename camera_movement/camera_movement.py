import cv2
import pickle
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovement():
    def __init__(self, frame):
        first_frame_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_greyscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.min_dist = 5

        self.features = dict(
            maxCorners = 100,
            qualityLevel =0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )

        self.lk_params = dict(
            winSize = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_cache = False, cache_path = None):
        if read_from_cache and cache_path is not None and os.path.exists(cache_path):
            with open(cache_path,'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)

        old_grey = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_grey, **self.features)

        for frame_num in range(1, len(frames)):
            frame_grey = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_grey, frame_grey, old_features, None, **self.lk_params)

            max_dist = 0

            camera_movement_x, camera_movement_Y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_pt = new.ravel()
                old_features_pt = old.ravel()

                distance = measure_distance(new_features_pt, old_features_pt)

                print(frame_num, distance )
                if distance > max_dist:
                    max_dist = distance
                    camera_movement_x, camera_movement_Y = measure_xy_distance(old_features_pt, new_features_pt)

                if max_dist > self.min_dist:
                    camera_movement[frame_num] = [camera_movement_x, camera_movement_Y]
                    old_features = cv2.goodFeaturesToTrack(frame_grey, **self.features)
                
            old_grey = frame_grey.copy()

        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self,frames, camera_movement_per_frame):
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame) 

        return output_frames


                


        
    