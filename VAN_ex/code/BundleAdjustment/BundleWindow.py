import gtsam
import numpy as np

from VAN_ex.code.Ex4.ex4 import Track, TracksDB
from VAN_ex.code.Ex5 import ex5 as ex5_utils
from VAN_ex.code.Ex3 import ex3 as ex3_utils
from VAN_ex.code.utils import projection_utils, utils

MAX_Z = 350


class Bundle:

    def __init__(self, first_frame, last_frame):
        self.frames_idxs = np.arange(first_frame, last_frame + 1)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        self.optimizer = None
        self.points = []
        self.cameras = []
        self.result = None

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

    def alternate_ver_create_graph(self, T_arr, tracks_db: TracksDB):
        K = utils.create_gtsam_K()
        base_camera = projection_utils.convert_ext_mat_to_world(T_arr[self.frames_idxs[0]])

        pose = None
        cam_symbol = None

        tracks_in_frames = set()
        # Create a pose for each camera in the bundle window
        for frame_id in self.frames_idxs:
            tracks_in_frames.update(tracks_db.get_track_ids(frame_id))

            cam_symbol = gtsam.symbol('c', frame_id)
            self.cameras.append(cam_symbol)

            ext_mat = T_arr[frame_id]
            cur_ext_mat = projection_utils.composite_transformations(base_camera, ext_mat)

            pose = gtsam.Pose3(projection_utils.convert_ext_mat_to_world(cur_ext_mat))
            self.initial_estimates.insert(cam_symbol, pose)

            # Add a prior factor just for first camera pose
            if frame_id == self.frames_idxs[0]:  # Constraints for first frame
                # sigmas array: first 3 for angles second 3 for location
                # I chose those values by assuming that theirs 1 angles uncertainty at the angles,
                # about 30cm at the x axes, 10cm at the y axes and 1 meter at the z axes which is the moving direction
                sigmas = np.array([(1 * np.pi / 180) ** 2] * 3 + [1e-1, 1e-2, 1.0])
                pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)
                factor = gtsam.PriorFactorPose3(cam_symbol, pose, pose_uncertainty)
                self.graph.add(factor)

        tracks_in_frames = list(tracks_in_frames)
        for track_id in tracks_in_frames:
            # Create a gtsam object for the last frame for making the projection at the function "add_factors"
            gtsam_last_cam = gtsam.StereoCamera(pose, K)
            first_frame = max(self.frames_idxs[0], tracks_db.tracks[track_id].get_frame_ids()[0])
            last_frame = min(self.frames_idxs[-1], tracks_db.tracks[track_id].get_frame_ids()[-1])
            if first_frame > last_frame:
                continue
            self.add_factors_to_graph(track=tracks_db.tracks[track_id], first_frame=first_frame, last_frame=last_frame,
                                      gtsam_frame_to_triangulate_from=gtsam_last_cam, K=K)

    def add_factors_to_graph(self, track: Track, first_frame, last_frame, gtsam_frame_to_triangulate_from, K):
        left_kp_all, right_kp_all = track.get_left_kp(), track.get_right_kp()
        # strip the locations to the bundle window
        left_locations = {frame_id: kp for frame_id, kp in left_kp_all.items() if first_frame <= frame_id <= last_frame}
        right_locations = {frame_id: kp for frame_id, kp in right_kp_all.items() if
                           first_frame <= frame_id <= last_frame}

        last_left_kp = left_locations[last_frame]
        last_right_kp = right_locations[last_frame]

        # measures for triangulation
        measure_xl, measure_xr, measure_y = last_left_kp[0], last_right_kp[0], last_left_kp[1]
        gtsam_stereo_point2_for_triangulation = gtsam.StereoPoint2(measure_xl, measure_xr, measure_y)
        gtsam_p3d = gtsam_frame_to_triangulate_from.backproject(gtsam_stereo_point2_for_triangulation)

        # Add landmark symbol to "values" dictionary
        p3d_sym = gtsam.symbol('q', track.get_track_id())
        self.points.append(p3d_sym)
        self.initial_estimates.insert(p3d_sym, gtsam_p3d)

        for frame_id in range(first_frame, last_frame + 1):
            # Measurement values
            measure_xl, measure_xr, measure_y = left_locations[frame_id][0], right_locations[frame_id][0], \
                left_locations[frame_id][1]
            gtsam_measurement_pt2 = gtsam.StereoPoint2(measure_xl, measure_xr, measure_y)

            # Factor creation
            projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
            factor = gtsam.GenericStereoFactor3D(gtsam_measurement_pt2, projection_uncertainty,
                                                 gtsam.symbol('c', frame_id), p3d_sym, K)
            # Add factor to the graph
            self.graph.add(factor)

    # def create_graph(self, T_arr, tracks_db):
    #     K = ex5_utils.compute_K()
    #     base_camera = T_arr[self.frames_idxs[0]]
    #
    #     # Create a pose for each camera in the bundle window
    #     for frame_id in self.frames_idxs:
    #         ext_mat = T_arr[frame_id]
    #         cur_ext_mat = ex3_utils.composite_transformations(base_camera, ext_mat)
    #
    #         pose = gtsam.Pose3(ex5_utils.fix_ext_mat(cur_ext_mat))
    #         cam_symbol = gtsam.symbol('c', frame_id)
    #         self.cameras.append(cam_symbol)
    #         self.initial_estimates.insert(cam_symbol, pose)
    #
    #         # Add a prior factor just for first camera pose
    #         if frame_id == self.frames_idxs[0]:  # Constraints for first frame
    #             factor = gtsam.PriorFactorPose3(cam_symbol, pose, gtsam.noiseModel.Diagonal.Sigmas(
    #                 np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])))
    #             self.graph.add(factor)
    #
    #     for track_id in tracks_db.get_track_ids(self.frames_idxs[0]):
    #         track = tracks_db.tracks[track_id]
    #         last_frame_id = track.frame_ids[-1]
    #         track_frames = track.get_frame_ids()
    #
    #         # if track.frame_ids[-1] < self.frames_idxs[-1]:
    #         #     continue
    #
    #         # Create a point for each track in the first keypoint frame
    #         base_stereo_frame = gtsam.StereoCamera(pose, K)  # Pose of last frame in bundle window
    #         xl, xr, y = tracks_db.feature_location(last_frame_id, track_id)
    #         point = gtsam.StereoPoint2(xl, xr, y)
    #         p3d = base_stereo_frame.backproject(point)
    #
    #         if p3d[2] < 0 or p3d[2] > MAX_Z:  # Threshold for far points
    #             continue
    #
    #         point_symbol = gtsam.symbol('q', track.get_track_id())
    #         self.points.append(point_symbol)
    #         self.initial_estimates.insert(point_symbol, p3d)
    #
    #         # Create a factor for each frame of track
    #         for frame_id in self.frames_idxs:
    #             if frame_id not in track_frames:
    #                 continue
    #             cam_symbol = gtsam.symbol('c', frame_id)
    #             xl, xr, y = tracks_db.feature_location(frame_id, track_id)
    #             point = gtsam.StereoPoint2(xl, xr, y)
    #
    #             factor = gtsam.GenericStereoFactor3D(point, gtsam.noiseModel.Isotropic.Sigma(3, 1.0), cam_symbol,
    #                                                  point_symbol, K)
    #             self.graph.add(factor)

    def get_from_optimized(self, obj:str):
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