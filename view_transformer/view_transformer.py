import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        court_width = 68
        court_length = 23.92

        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 160],
            [1640, 915]
        ])

        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None
        
        reshaped_pt = point.reshape(-1, 1, 2).astype(np.float32)
        transform_pt = cv2.perspectiveTransform(reshaped_pt, self.perspective_transformer)
 
        return transform_pt.reshape(-1, 2)


    def add_tranformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)

                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed