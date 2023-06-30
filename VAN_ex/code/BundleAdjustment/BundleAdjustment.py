import os
import pickle

import gtsam
import numpy as np
from tqdm import tqdm

from VAN_ex.code.DataBase.TracksDB import TracksDB
from VAN_ex.code.BundleAdjustment import BundleWindow
from VAN_ex.code.utils import utils, projection_utils

FRAC = 0.6


def save_ba(ba, path):
    """
    Saves the BundleAdjustment object to the serialized file.
    """
    with open(path, 'wb') as file:
        pickle.dump(ba, file)


def load_ba(path):
    """
    Loads the BundleAdjustment object from the serialized file.
    """
    with open(path, 'rb') as file:
        ba = pickle.load(file)
    return ba


class BundleAdjustment:

    def __init__(self, tracks_db: TracksDB, T_arr: np.ndarray):
        self.keyframes = [0]
        self.bundle_windows = []
        self.cameras_rel_pose = []
        self.points_rel_pose = []
        self.init_camera_rel_pose = []
        self.tracks_db = tracks_db
        self.T_arr = T_arr

    @utils.measure_time
    def choose_keyframes(self, choosing_method=None, **kwargs):
        print("choosing keyframes...")
        if choosing_method is None:
            self.choose_keyframes_median()
        else:
            choosing_method(**kwargs)
        print("finished choosing keyframes, number of keyframes: ", len(self.keyframes))

    def choose_keyframes_median(self, median=FRAC):
        while self.keyframes[-1] < len(self.tracks_db.frame_ids) - 1:
            tracks_in_keyframe = self.tracks_db.get_track_ids(self.keyframes[-1])
            end_frames = sorted([self.tracks_db.tracks[track].frame_ids[-1] for track in tracks_in_keyframe])
            self.keyframes.append(end_frames[int(len(end_frames) * median)])
            if len(self.tracks_db.frame_ids) - 1 - self.keyframes[-1] < 10:
                self.keyframes.append(len(self.tracks_db.frame_ids) - 1)
                break
        # print('First 10 Keyframes: ', self.keyframes[:10])

    def choose_keyframes_every_n(self, n=5):
        self.keyframes = list(range(0, len(self.tracks_db.frame_ids), n))[:15]
        # self.keyframes.append(len(self.tracks_db.frame_ids) - 1)
        # print('First 10 Keyframes: ', self.keyframes[:10])

    @utils.measure_time
    def solve(self):
        print("solving...")
        self.bundle_windows = self.create_bundle_windows(self.keyframes)
        cameras = [gtsam.Pose3()]
        points = []
        for bundle_window in tqdm(self.bundle_windows):
            bundle_window.create_graph_v2(self.T_arr, self.tracks_db)
            bundle_window.optimize()
            cameras.append(bundle_window.get_from_optimized(obj='camera_p3d'))
            points.append(bundle_window.get_from_optimized(obj='landmarks'))

        cameras = np.array(cameras)
        self.cameras_rel_pose = cameras
        self.points_rel_pose = points
        print("finish solving")
        return self.cameras_rel_pose, self.points_rel_pose

    def get_relative_poses(self):
        cameras = projection_utils.convert_rel_gtsam_trans_to_global(self.cameras_rel_pose)
        landmarks = projection_utils.convert_rel_landmarks_to_global(cameras, self.points_rel_pose)
        return cameras, landmarks

    @staticmethod
    def create_bundle_windows(keyframes):
        bundle_windows = []
        for i in range(len(keyframes) - 1):
            bundle_windows.append(BundleWindow.Bundle(keyframes[i], keyframes[i + 1]))
        return bundle_windows

    def serialize(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def deserialize(file_name):
        with open(file_name, 'rb') as file:
            return pickle.load(file)
