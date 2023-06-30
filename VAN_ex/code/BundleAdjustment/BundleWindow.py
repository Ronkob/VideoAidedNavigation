import gtsam
import numpy as np

from VAN_ex.code.Ex3 import ex3 as ex3_utils
from VAN_ex.code.DataBase.TracksDB import TracksDB
from VAN_ex.code.DataBase.Track import Track
from VAN_ex.code.Ex5 import ex5 as ex5_utils
from VAN_ex.code.utils import projection_utils, utils

MAX_Z = 200
MIN_Y = -10


class Bundle:

    def __init__(self, first_frame, last_frame, loop_tracks=None):
        if loop_tracks:
            self.frames_idxs = [first_frame, last_frame]
        else:
            self.frames_idxs = np.arange(first_frame, last_frame + 1)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        self.points = []
        self.cameras = []
        self.optimizer = None
        self.prior_factor = None
        self.result = None
        self.loop_tracks = loop_tracks

    def get_marginals(self):
        # print("Getting marginals...")
        # print(f"Graph: {self.graph}")
        # print(f"Result: {self.result}")

        if self.result:
            # print("Computing marginals from result...")
            marginals = gtsam.Marginals(self.graph, self.result)
        else:
            # print("Computing marginals from initial estimates...")
            marginals = gtsam.Marginals(self.graph, self.initial_estimates)
        return marginals

    def get_factor_error(self, initial: bool = False):
        if initial:
            return self.graph.error(self.initial_estimates)
        else:
            return self.graph.error(self.result)

    def optimize(self):
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimates)
        self.result = self.optimizer.optimize()
        return self.result

    def create_graph_v2(self, T_arr, tracks_db: TracksDB):
        K = utils.create_gtsam_K()
        base_camera = projection_utils.convert_ext_mat_to_world(T_arr[self.frames_idxs[0]])

        tracks_in_frames = set()

        # Create a pose for each camera in the bundle window
        for frame_id in self.frames_idxs:
            tracks_in_frames.update(tracks_db.get_track_ids(frame_id))

            symbol = gtsam.symbol('c', frame_id)
            self.cameras.append(symbol)

            ext_mat = T_arr[frame_id]
            cur_ext_mat = projection_utils.composite_transformations(base_camera, ext_mat)
            pose = gtsam.Pose3(projection_utils.convert_ext_mat_to_world(cur_ext_mat))
            self.initial_estimates.insert(symbol, pose)

            # Add a prior factor just for first camera pose
            if frame_id == self.frames_idxs[0]:  # Constraints for first frame
                sigmas = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) ** 7
                cov = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)
                factor = gtsam.PriorFactorPose3(symbol, pose, cov)
                self.prior_factor = factor
                self.graph.add(factor)

        tracks_in_frames = list(tracks_in_frames)
        for track_id in tracks_in_frames:
            # create a gtsam camera for the last frame of the bundle window
            last_stereo = gtsam.StereoCamera(pose, K)
            first_frame = max(self.frames_idxs[0], tracks_db.tracks[track_id].get_frame_ids()[0])
            last_frame = min(self.frames_idxs[-1], tracks_db.tracks[track_id].get_frame_ids()[-1])
            if first_frame > last_frame:
                continue
            if self.loop_tracks:
                tracks = self.loop_tracks
            else:
                tracks = tracks_db.tracks[track_id]
            self.extract_factors_to_gtsam(track=tracks, first_frame=first_frame, last_frame=last_frame,
                                          gtsam_frame_to_triangulate_from=last_stereo, K=K)

    def extract_factors_to_gtsam(self, track: Track, first_frame, last_frame, gtsam_frame_to_triangulate_from, K):
        left_kp_all, right_kp_all = track.get_left_kp(), track.get_right_kp()

        # get the locations of the track only inside the bundle window
        left_locations = {frame_id: kp for frame_id, kp in left_kp_all.items() if first_frame <= frame_id <= last_frame}
        right_locations = {frame_id: kp for frame_id, kp in right_kp_all.items() if
                           first_frame <= frame_id <= last_frame}

        last_left_kp = left_locations[last_frame]
        last_right_kp = right_locations[last_frame]

        # coords for triangulation
        xl, xr, y = last_left_kp[0], last_right_kp[0], last_left_kp[1]
        point = gtsam.StereoPoint2(xl, xr, y)
        p3d = gtsam_frame_to_triangulate_from.backproject(point)

        # Add the point to the graph and to the list of points only if it's z is not too big
        if p3d[-1] >= MAX_Z or p3d[2] <= 0:
            return

        # watch for the point's y coordinate
        if p3d[1] > 1 or p3d[1] < MIN_Y:
            return

        # print(f"Adding point {track.get_track_id()} to graph, p3d: {p3d}")

        symbol = gtsam.symbol('q', track.get_track_id())
        self.points.append(symbol)
        self.initial_estimates.insert(symbol, p3d)

        for frame_id in range(first_frame, last_frame + 1):
            xl, xr, y = left_locations[frame_id][0], right_locations[frame_id][0], left_locations[frame_id][1]
            inner_point = gtsam.StereoPoint2(xl, xr, y)

            # Create a stereo factor
            projection_cov = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
            factor = gtsam.GenericStereoFactor3D(inner_point, projection_cov, gtsam.symbol('c', frame_id),
                                                 symbol, K)
            # Add the factor to the graph
            self.graph.add(factor)

    def get_from_optimized(self, obj: str):
        if obj == 'values':
            return self.result

        elif obj == 'landmarks':
            landmarks = []
            for landmark_sym in self.points:
                landmark = self.result.atPoint3(landmark_sym)
                landmarks.append(landmark)
            return landmarks

        elif obj == 'camera_poses':
            cameras_poses = []
            for camera_sym in self.cameras:
                cam_pose = self.result.atPose3(camera_sym)
                cameras_poses.append(cam_pose)
            return cameras_poses

        elif obj == 'camera_p3d':
            cam_pose = self.result.atPose3(gtsam.symbol('c', self.frames_idxs[-1]))
            return cam_pose
        else:
            raise Exception('Invalid object name')

    def get_from_initial(self, obj):
        if obj == 'values':
            return self.initial_estimates

        elif obj == 'landmarks':
            landmarks = []
            for landmark_sym in self.points:
                landmark = self.initial_estimates.atPoint3(landmark_sym)
                landmarks.append(landmark)
            return landmarks

        elif obj == 'camera_poses':
            cameras_poses = []
            for camera_sym in self.cameras:
                cam_pose = self.initial_estimates.atPose3(camera_sym)
                cameras_poses.append(cam_pose)
            return cameras_poses

        elif obj == 'camera_p3d':
            cam_pose = self.initial_estimates.atPose3(gtsam.symbol('c', self.frames_idxs)).inverse()
            return cam_pose
        else:
            raise Exception('Invalid object name')
