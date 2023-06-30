import os

import gtsam
import pickle
import numpy as np
from tqdm import tqdm

from VAN_ex.code.BundleAdjustment import BundleWindow
from VAN_ex.code.BundleAdjustment.BundleAdjustment import BundleAdjustment, FRAC
from VAN_ex.code.BundleAdjustment.BundleWindow import Bundle
from VAN_ex.code.PreCalcData.paths_to_data import RELATIVES_PATH
from VAN_ex.code.utils import utils
# from VAN_ex.code.PoseGraph.VertexGraph import VertexGraph
from dijkstar import Graph


def save_pg(pose_graph, path):
    """
    Save the pose graph to a file.
    """
    with open(path, 'wb') as file:
        pickle.dump(pose_graph, file)


def load_pg(path):
    """
    Load the pose graph from a file.
    if the path not exist return None
    """
    with open(path, 'rb') as file:
        pose_graph = pickle.load(file)

    return pose_graph


class PoseGraph:
    """
    Class represents the factor graph which is build from the keyframes.
    """

    def __init__(self, tracks_db=None, T_arr=None, ba: BundleAdjustment = None, relative_poses_path=None,
                 choosing_method=None, **kwargs):

        self.keyframes = [0]
        self.graph = gtsam.NonlinearFactorGraph()
        self.rel_poses, self.rel_covs = [], []
        self.tracks_db = tracks_db
        self.T_arr = T_arr
        self.initial_estimates = gtsam.Values()
        self.result = None
        self.bundle_windows = []

        if ba is not None:
            self.init_from_ba(ba)

        else:
            self.init_without_ba(choosing_method, **kwargs)

        if relative_poses_path is not None:
            with open(relative_poses_path, 'rb') as file:
                self.rel_poses, self.rel_covs = pickle.load(file)
        else:
            self.calculate_rel_cov_and_poses(self.bundle_windows)

        self.create_initial_pose_graph()

        self.vertex_graph = Graph()
        self.init_vertex_graph()

    def init_without_ba(self, choosing_method, **kwargs):
        self.choose_keyframes(choosing_method, **kwargs)
        self.bundle_windows = self.create_bundle_windows(self.keyframes)
        self.optimize_bundles(self.bundle_windows)

    def init_from_ba(self, ba: BundleAdjustment):
        self.keyframes = ba.keyframes
        self.bundle_windows = ba.bundle_windows
        self.tracks_db = ba.tracks_db
        self.T_arr = ba.T_arr

    def optimize_bundles(self, bundle_windows):
        """
        Optimize the bundle windows.
        """
        for bundle in tqdm(bundle_windows):
            bundle.create_graph_v2(self.T_arr, self.tracks_db)
            bundle.optimize()

    @utils.measure_time
    def calculate_rel_cov_and_poses(self, bundle_windows):
        """
        Calculate relative poses and covariances between keyframes.
        """
        for bundle in tqdm(bundle_windows):
            rel_cov, rel_pos = self.rel_cov_and_pos_for_bundle(bundle)
            self.rel_covs.append(rel_cov)
            self.rel_poses.append(rel_pos)

        with open(RELATIVES_PATH, 'wb') as file:
            pickle.dump((self.rel_covs, self.rel_poses), file)

    def get_graph_error(self, initial: bool = False):
        """
        Calculate the error of the graph.
        """
        if initial:
            return self.graph.error(self.initial_estimates)
        else:
            return self.graph.error(self.result)

    def solve(self):
        """
        Solve the pose graph.
        """
        print('Solving the pose graph..')
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimates)
        self.result = optimizer.optimize()
        print('finished solving the pose graph.')
        return self.result

    def get_marginals(self):
        """
        Get the marginals of the graph after optimization.
        """
        marginals = gtsam.Marginals(self.graph, self.result)
        return marginals

    def create_initial_pose_graph(self):
        """
        Create the pose graph with initial estimates.
        """
        # Init first camera pose
        init_pose = gtsam.Pose3()
        init_cam = gtsam.symbol('c', 0)
        self.initial_estimates.insert(init_cam, init_pose)

        # Add a prior factor to graph
        sigmas = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) ** 7
        pose_cov = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)
        factor = gtsam.PriorFactorPose3(init_cam, init_pose, pose_cov)
        self.graph.add(factor)

        prev_cam = init_cam
        cur_world_pose = init_pose

        for i in range(len(self.rel_covs)):
            # Add relative pose factor
            camera = gtsam.symbol('c', i + 1)
            cov = gtsam.noiseModel.Gaussian.Covariance(self.rel_covs[i])
            factor = gtsam.BetweenFactorPose3(prev_cam, camera, self.rel_poses[i], cov)
            self.graph.add(factor)

            # Add initial estimate
            cur_world_pose = cur_world_pose.compose(self.rel_poses[i])
            self.initial_estimates.insert(camera, cur_world_pose)

            prev_cam = camera

    def choose_keyframes(self, choosing_method=None, **kwargs):
        if choosing_method is None:
            self.choose_keyframes_median()
        else:
            choosing_method(self, **kwargs)

    def choose_keyframes_median(self, median=FRAC):
        while self.keyframes[-1] < len(self.tracks_db.frame_ids) - 1:
            tracks_in_keyframe = self.tracks_db.get_track_ids(self.keyframes[-1])
            end_frames = sorted([self.tracks_db.tracks[track].frame_ids[-1] for track in tracks_in_keyframe])
            self.keyframes.append(end_frames[int(len(end_frames) * median)])
            if len(self.tracks_db.frame_ids) - 1 - self.keyframes[-1] < 10:
                self.keyframes.append(len(self.tracks_db.frame_ids) - 1)
                break
        print('First 10 Keyframes: ', self.keyframes[:10])

    def choose_keyframes_every_n(self, n=5):
        self.keyframes = list(range(0, len(self.tracks_db.frame_ids), n))
        self.keyframes.append(len(self.tracks_db.frame_ids) - 1)
        print('First 10 Keyframes: ', self.keyframes[:10])

    def rel_cov_and_pos_for_bundle(self, bundle: BundleWindow):
        first_kf, second_kf = bundle.frames_idxs[0], bundle.frames_idxs[-1]
        keys = gtsam.KeyVector()
        keys.append(gtsam.symbol('c', first_kf))
        keys.append(gtsam.symbol('c', second_kf))
        try:
            marginals = bundle.get_marginals()
        except:
            print("Failed to get marginals for bundle window: ", bundle.frames_idxs)
            # print the exception traceback
            import traceback
            traceback.print_exc()
            # exit(1)
            return
        # rel_cov = marginals.jointMarginalCovariance(keys).at(keys[1], keys[1])
        joint_information_mat = marginals.jointMarginalInformation(keys).at(keys[1], keys[1])
        rel_cov = np.linalg.inv(joint_information_mat)

        first_pose = bundle.result.atPose3(gtsam.symbol('c', first_kf))
        second_pose = bundle.result.atPose3(gtsam.symbol('c', second_kf))
        rel_pose = first_pose.between(second_pose)
        return rel_cov, rel_pose

    def init_vertex_graph(self):
        """
        Initialize the vertex graph with keyframes and relative covariances as weights.
        """
        for i in range(len(self.keyframes) - 1):
            self.vertex_graph.add_edge(i, i + 1, utils.weight_func(self.rel_covs[i]))

    def calc_cov_along_path(self, path):
        """
        Calculate the covariance along the shortest path.
        """
        rel_cov = np.zeros((6, 6))
        path = path.nodes
        for j in range(path[0], path[-1]):
            rel_cov += self.rel_covs[j]
        return rel_cov

    @staticmethod
    def create_bundle_windows(keyframes):
        bundle_windows = []
        for i in range(len(keyframes) - 1):
            bundle_windows.append(Bundle(keyframes[i], keyframes[i + 1]))
        return bundle_windows
