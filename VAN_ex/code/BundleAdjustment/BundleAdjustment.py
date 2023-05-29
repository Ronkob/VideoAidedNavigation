import gtsam
import numpy as np
from VAN_ex.code import utils
from VAN_ex.code.Ex3 import ex3 as ex3_utils
from VAN_ex.code.Ex4.ex4 import TracksDB
from VAN_ex.code.BundleAdjustment import BundleWindow


class BundleAdjustment:

    def __init__(self, tracks_db: TracksDB, T_arr: np.ndarray):
        self.keyframes = []
        self.bundle_windows = []
        self.cameras_rel_pose = []
        self.points_rel_pose = []
        self.tracks_db = tracks_db
        self.T_arr = T_arr

    def decide_on_keyframes_by_time(self, keyframe_time_interval: float = 5):
        key_frames = []
        for frame_id in range(len(self.tracks_db.frame_ids))[:50]:
            if frame_id % keyframe_time_interval == 0:
                key_frames.append(frame_id)

        self.keyframes = key_frames

    def solve(self):
        self.bundle_windows = create_bundle_windows(self.keyframes)
        for bundle_window in self.bundle_windows:
            bundle_window.create_graph(self.T_arr, self.tracks_db)
            result = bundle_window.optimize()

            # Between each keyframe and its predecessor
            self.cameras_rel_pose.append(result.atPose3(gtsam.symbol('c', bundle_window.frames_idxs[-1])))
            self.points_rel_pose.append([result.atPoint3(point) for point in bundle_window.points])

    def get_relative_poses(self):
        return self.cameras_rel_pose, self.points_rel_pose


def create_bundle_windows(keyframes):
    bundle_windows = []
    for i in range(len(keyframes) - 1):
        bundle_windows.append(BundleWindow.Bundle(keyframes[i], keyframes[i + 1]))
    return bundle_windows
