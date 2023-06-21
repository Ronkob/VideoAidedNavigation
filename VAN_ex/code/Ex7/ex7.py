import os
import gtsam
import pickle
import numpy as np

from gtsam.utils import plot
import matplotlib.pyplot as plt

from VAN_ex.code.Ex3.ex3 import calculate_relative_transformations, track_movement_successive
from VAN_ex.code.Ex4.ex4 import TracksDB, Track
from VAN_ex.code.Ex5.ex5 import plot_scene_3d, plot_scene_from_above
from VAN_ex.code.BundleAdjustment import BundleWindow
from VAN_ex.code.PoseGraph.PoseGraph import PoseGraph
from VAN_ex.code.utils import utils

DB_PATH = os.path.join('..', 'Ex4', 'tracks_db.pkl')
T_ARR_PATH = os.path.join('..', 'Ex3', 'T_arr.npy')
MAHAL_THRESH = 0.5
MAX_CANDIDATES = 5


def q7_1(pose_graph: PoseGraph, n_idx: int):
    """
    Detect Loop Closure Candidates.
    a. Relative Covariance - find the shortest path from c_n to c_i and sum the covariances along the path to get an
     estimate of the relative covariance.
    b. Detect Possible Candidates - choose the most likely candidate to be close to the pose c_n by applying a
    Mahalanobis distance test with c_n,i - the relative pose between c_n and c_i.
    Choose a threshold to determine if the candidate advances to the next (expensive) stage.
    """
    candidate_frames = dict()
    cn_symbol = gtsam.symbol('c', n_idx)
    cn_pose = pose_graph.result.atPose3(cn_symbol)

    for i in range(n_idx):
        ci_symbol = gtsam.symbol('c', i)
        ci_pose = pose_graph.result.atPose3(ci_symbol)

        # Find the shortest path from c_n to c_i using dijkstra algorithm
        shortest_path = pose_graph.vertex_graph.find_shortest_path(n_idx, i)

        # Sum the covariances along the path to get an estimate of the relative covariance
        rel_cov = pose_graph.vertex_graph.calc_cov_along_path(shortest_path, pose_graph.rel_covs)

        # Calculate Mahalanobis distance
        mahalanobis_dist = np.sqrt(np.dot(np.dot((cn_pose.between(ci_pose).matrix() - np.eye(4)).T, rel_cov),
                                            cn_pose.between(ci_pose).matrix() - np.eye(4)))

        # Choose a threshold to determine if the candidate advances to the next (expensive) stage
        if mahalanobis_dist < MAHAL_THRESH:
            candidate_frames[i] = mahalanobis_dist

    if not candidate_frames:
        return []
    sorted_candidates = sorted(candidate_frames.items(), key=lambda x: x[1])
    return [item for item in sorted_candidates[:MAX_CANDIDATES]]


def q7_2(candidates, n_idx):
    """
    Consensus Matching - perform consensus match between the two candidate frames. (See exercise 3)
    Set a threshold for the number of inliers that indicates a successful match. Note that this is
    typically a more difficult match than that of two consecutive frames
    """
    INLIERS_THRESH = 70
    inliers = []
    for candidate in candidates:
        left_ext_mat, cur_inliers, inliers_precent = track_movement_successive([n_idx, candidate])
        if inliers_precent > INLIERS_THRESH:
            inliers.append(cur_inliers)
    return inliers


def q7_3():
    """
    Relative Pose Estimation - using the inlier matches perform a small Bundle optimization to extract the relative
     pose of the two frames as well as its covariance.
    """
    pass


def q7_4():
    """
    Update the Pose Graph - add the resulting synthetic measurement to the pose graph and optimize it to update the
    trajectory estimate.
    """
    pass


def q7_5():
    """
    Display plots.
    """
    pass


def run_ex7():
    """
    Runs all exercise 7 sections.
    """
    np.random.seed(1)
    # Load tracks DB
    tracks_db = TracksDB.deserialize(DB_PATH)
    T_arr = np.load(T_ARR_PATH)
    rel_t_arr = calculate_relative_transformations(T_arr)

    pose_graph = PoseGraph(tracks_db, T_arr)

    # For each key frame cn in the pose graph loop over previous keyframes ci, i < n, and perform
    # steps 7.1-7.4:
    for i in range(1, len(pose_graph.keyframes)):
        # Detect Loop Closure Candidates
        candidates = q7_1(pose_graph, i)

        # 7.2 Consensus Matching
        # inliers = q7_2(candidates, i)

        # 7.3 Relative Pose Estimation
        # q7_3(inliers)

        # 7.4 Update the Pose Graph
        # q7_4()

    # q7_5()


def main():
    run_ex7()


if __name__ == '__main__':
    main()
