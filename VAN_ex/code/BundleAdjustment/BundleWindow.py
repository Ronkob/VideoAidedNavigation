import gtsam
import numpy as np
from VAN_ex.code import utils
from VAN_ex.code.Ex3 import ex3 as ex3_utils

MAX_Z = 350


class Bundle:

    def __init__(self, first_frame, last_frame):
        self.frames_idxs = np.arange(first_frame, last_frame+1)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        self.optimizer = None
        self.result = None
        self.points = []
        self.cameras = []

    def get_marginals(self):
        marginals = gtsam.Marginals(self.graph, self.result)
        return marginals

    def get_factor_error(self, initial: bool = False):
        if initial:
            return self.graph.error(self.initial_estimates)
        else:
            return self.graph.error(self.result)

    def optimize(self):
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimates)
        # print a detailed debug info on self.graph and self.initial_estimates
        # print("optimizing...")
        # print("Size of graph: ", self.graph.size())
        # print("Number of initial estimates: ", self.initial_estimates.size())

        self.result = self.optimizer.optimize()
        return self.result

    def create_graph(self, T_arr, tracks_db):
        K = utils.create_gtsam_K()
        first_frame_ext_mat = T_arr[self.frames_idxs[0]]
        world_base_camera = utils.fix_ext_mat(first_frame_ext_mat)  # World coordinates for transformations

        # Create a pose for each camera in the bundle window
        for frame_id in self.frames_idxs:
            ext_mat = T_arr[frame_id]
            cur_ext_mat = ex3_utils.composite_transformations(world_base_camera, ext_mat)
            cur_cam_in_world = utils.fix_ext_mat(cur_ext_mat)

            pose = gtsam.Pose3(cur_cam_in_world)
            cam_symbol = gtsam.symbol('c', frame_id)
            self.cameras.append(cam_symbol)
            self.initial_estimates.insert(cam_symbol, pose)

            # Add a prior factor for first camera pose
            if frame_id == self.frames_idxs[0]:  # Constraints for first frame
                factor = gtsam.PriorFactorPose3(cam_symbol, pose, gtsam.noiseModel.Diagonal.Sigmas(
                    np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])))
                self.graph.add(factor)

        for track_id in tracks_db.get_track_ids(self.frames_idxs[0]):
            track = tracks_db.tracks[track_id]
            last_frame_id = track.frame_ids[-1]
            track_frames = track.get_frame_ids()

            if track.frame_ids[-1] < self.frames_idxs[-1]:
                continue

            # Create a point for each track in the first keypoint frame
            base_stereo_frame = gtsam.StereoCamera(pose, K)  # Pose of last frame in bundle window
            xl, xr, y = tracks_db.feature_location(last_frame_id, track_id)
            point = gtsam.StereoPoint2(xl, xr, y)
            p3d = base_stereo_frame.backproject(point)

            if p3d[2] < 0 or p3d[2] > MAX_Z:  # Threshold for far points
                continue

            point_symbol = gtsam.symbol('q', track.get_track_id())
            self.points.append(point_symbol)
            self.initial_estimates.insert(point_symbol, p3d)

            # Create a factor for each frame of track
            for frame_id in self.frames_idxs:
                if frame_id not in track_frames:
                    continue
                cam_symbol = gtsam.symbol('c', frame_id)
                xl, xr, y = tracks_db.feature_location(frame_id, track_id)
                point = gtsam.StereoPoint2(xl, xr, y)

                factor = gtsam.GenericStereoFactor3D(point, gtsam.noiseModel.Isotropic.Sigma(3, 1.0), cam_symbol,
                                                     point_symbol, K)
                self.graph.add(factor)
