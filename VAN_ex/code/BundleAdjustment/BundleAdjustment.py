import gtsam
import numpy as np
from VAN_ex.code.Ex4.ex4 import TracksDB
from VAN_ex.code.BundleAdjustment import BundleWindow


class BundleAdjustment:

    def __init__(self, tracks_db: TracksDB, T_arr: np.ndarray):
        self.keyframes = []
        self.bundle_windows = []
        self.cameras_rel_pose = []
        self.points_rel_pose = []
        self.init_camera_rel_pose = []
        self.tracks_db = tracks_db
        self.T_arr = T_arr

    def choose_keyframes(self, type):
        if type == 'length':
            INTERVAL = 10
            key_frames = []
            for frame_id in range(len(self.tracks_db.frame_ids)):
                if frame_id % INTERVAL == 0:
                    key_frames.append(frame_id)
            self.keyframes = key_frames

        elif type == 'end_frame':
            FRAC = 0.85
            self.keyframes.append(0)
            while self.keyframes[-1] < len(self.tracks_db.frame_ids) - 1:
                tracks_in_keyframe = self.tracks_db.get_track_ids(self.keyframes[-1])
                end_frames = sorted([self.tracks_db.tracks[track].frame_ids[-1] for track in tracks_in_keyframe])
                self.keyframes.append(end_frames[int(len(end_frames) * FRAC)])
                if len(self.tracks_db.frame_ids) - 1 - self.keyframes[-1] < 10:
                    self.keyframes.append(len(self.tracks_db.frame_ids) - 1)
                    break
        print('First 10 Keyframes: ', self.keyframes[:10])

    def solve(self):
        self.bundle_windows = self.create_bundle_windows(self.keyframes)
        for bundle_window in self.bundle_windows:
            bundle_window.create_graph(self.T_arr, self.tracks_db)
            result = bundle_window.optimize()

            # Between each keyframe and its predecessor
            if not self.cameras_rel_pose:
                self.cameras_rel_pose.append(result.atPose3(gtsam.symbol('c', 0)))
                self.init_camera_rel_pose.append(bundle_window.initial_estimates.atPose3(gtsam.symbol('c', 0)))
            self.cameras_rel_pose.append(result.atPose3(gtsam.symbol('c', bundle_window.frames_idxs[-1])))
            self.points_rel_pose.append([result.atPoint3(point) for point in bundle_window.points])
            self.init_camera_rel_pose.append(bundle_window.initial_estimates.atPose3(gtsam.symbol('c', bundle_window.frames_idxs[-1])))

    def get_relative_poses(self):
        return self.cameras_rel_pose, self.points_rel_pose, self.init_camera_rel_pose

    @staticmethod
    def create_bundle_windows(keyframes):
        bundle_windows = []
        for i in range(len(keyframes) - 1):
            bundle_windows.append(BundleWindow.Bundle(keyframes[i], keyframes[i + 1]))
        return bundle_windows
