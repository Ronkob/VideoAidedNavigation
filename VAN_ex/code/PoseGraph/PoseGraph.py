import gtsam
import pickle
import numpy as np

from VAN_ex.code.BundleAdjustment.BundleWindow import Bundle
from VAN_ex.code.utils import utils
from VAN_ex.code.PoseGraph.VertexGraph import VertexGraph
from dijkstar import Graph


class PoseGraph:
    """
    Class represents the factor graph which is build from the keyframes.
    """

    def __init__(self, tracks_db, T_arr):
        self.keyframes = [0]
        self.graph = gtsam.NonlinearFactorGraph()
        self.rel_poses, self.rel_covs = [], []
        self.tracks_db = tracks_db
        self.T_arr = T_arr
        self.initial_estimates = gtsam.Values()
        self.result = None

        self.choose_keyframes()
        self.bundle_windows = self.create_bundle_windows(self.keyframes)

        # self.calculate_rel_cov_and_poses()
        with open('relatives.pkl', 'rb') as file:  # When already saved
            self.rel_covs, self.rel_poses = pickle.load(file)

        self.create_pose_graph()

        # self.vertex_graph = VertexGraph(len(self.keyframes), self.rel_covs)
        self.vertex_graph = Graph()
        self.init_vertex_graph()


    @utils.measure_time
    def calculate_rel_cov_and_poses(self):
        """
        Calculate relative poses and covariances between keyframes.
        """
        for bundle in self.bundle_windows:
            rel_cov, rel_pos = self.rel_cov_and_pos_for_bundle(bundle)
            self.rel_covs.append(rel_cov)
            self.rel_poses.append(rel_pos)

        with open('relatives.pkl', 'wb') as file:
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
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimates)
        self.result = optimizer.optimize()

    def get_marginals(self):
        """
        Get the marginals of the graph after optimization.
        """
        marginals = gtsam.Marginals(self.graph, self.result)
        return marginals

    def create_pose_graph(self):
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

    def choose_keyframes(self):
        FRAC = 0.6
        while self.keyframes[-1] < len(self.tracks_db.frame_ids) - 1:
            tracks_in_keyframe = self.tracks_db.get_track_ids(self.keyframes[-1])
            end_frames = sorted([self.tracks_db.tracks[track].frame_ids[-1] for track in tracks_in_keyframe])
            self.keyframes.append(end_frames[int(len(end_frames) * FRAC)])
            if len(self.tracks_db.frame_ids) - 1 - self.keyframes[-1] < 10:
                self.keyframes.append(len(self.tracks_db.frame_ids) - 1)
                break
        print('First 10 Keyframes: ', self.keyframes[:10])

    def rel_cov_and_pos_for_bundle(self, bundle):
        first_kf, second_kf = bundle.frames_idxs[0], bundle.frames_idxs[-1]
        keys = gtsam.KeyVector()
        keys.append(gtsam.symbol('c', first_kf))
        keys.append(gtsam.symbol('c', second_kf))
        bundle.create_graph_v2(self.T_arr, self.tracks_db)
        bundle.optimize()
        try:
            marginals = bundle.get_marginals()
        except:
            print("Failed to get marginals for bundle window: ",
                  bundle.frames_idxs)
            # print the exception traceback
            import traceback
            traceback.print_exc()
            # exit(1)
            return
        # rel_cov = marginals.jointMarginalCovariance(keys).at(keys[1], keys[1])
        joint_information_mat = marginals.jointMarginalInformation(keys).at(
            keys[1], keys[1])
        rel_cov = np.linalg.inv(joint_information_mat)

        first_pose = bundle.result.atPose3(gtsam.symbol('c', first_kf))
        second_pose = bundle.result.atPose3(gtsam.symbol('c', second_kf))
        rel_pose = first_pose.between(second_pose)
        return rel_cov, rel_pose

    def init_vertex_graph(self):
        """
        Initialize the vertex graph with keyframes and relative covariances as weights.
        """
        for i in range(len(self.keyframes)-1):
            self.vertex_graph.add_edge(i, i+1, utils.weight_func(self.rel_covs[i]))

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
