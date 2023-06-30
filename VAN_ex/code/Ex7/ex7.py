import os
import gtsam
import pickle
import numpy as np

from gtsam.utils import plot
import matplotlib.pyplot as plt

from VAN_ex.code.utils import utils, projection_utils, auxilery_plot_utils
from VAN_ex.code.Ex3.ex3 import calculate_relative_transformations, track_movement_successive
from VAN_ex.code.DataBase.TracksDB import TracksDB, create_loop_tracks
from VAN_ex.code.DataBase.Track import Track
from VAN_ex.code.utils.auxilery_plot_utils import plot_scene_from_above, plot_scene_3d
from VAN_ex.code.BundleAdjustment.BundleWindow import Bundle
from VAN_ex.code.PoseGraph.PoseGraph import PoseGraph, save_pos_graph, load_pos_graph
from VAN_ex.code.PreCalcData.paths_to_data import RELATIVES_PATH

from dijkstar import find_path

DB_PATH = os.path.join('..', 'Ex4', 'tracks_db.pkl')
T_ARR_PATH = os.path.join('..', 'Ex3', 'T_arr.npy')

MAHAL_THRESH = 0.5
MAX_CANDIDATES = 3
INLIERS_PREC_THRESH = 70

loops_arr = []


# @utils.measure_time
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
        shortest_path = find_path(pose_graph.vertex_graph, i, n_idx)

        # Sum the covariances along the path to get an estimate of the relative covariance
        rel_cov = pose_graph.calc_cov_along_path(shortest_path)

        # Calculate Mahalanobis distance
        mahalanobis_dist = utils.calc_mahalanobis_dist(cn_pose, ci_pose, rel_cov)
        # print the mahalanobis distance with no new line
        # Choose a threshold to determine if the candidate advances to the next (expensive) stage
        if mahalanobis_dist < MAHAL_THRESH:
            print(i, ":", mahalanobis_dist)
            candidate_frames[i] = mahalanobis_dist

    print(n_idx, ":", candidate_frames)
    if not candidate_frames:
        return []
    sorted_candidates = sorted(candidate_frames.items(), key=lambda x: x[1])
    return [item for item in sorted_candidates[:MAX_CANDIDATES]]  # only best 3 candidates


@utils.measure_time
def q7_2(candidates, n_idx):
    """
    Consensus Matching - perform consensus match between the two candidate frames. (See exercise 3)
    Set a threshold for the number of inliers that indicates a successful match.
    Note that this is typically a more difficult match than that of two consecutive frames.
    """
    fitted_candidates = []
    for candidate in candidates:
        left_ext_mat, cur_inliers, inliers_precent = track_movement_successive([n_idx, candidate])
        if inliers_precent > INLIERS_PREC_THRESH:
            left0_inliers, _, left1_inliers, _ = cur_inliers
            fitted_candidates.append([left0_inliers, left1_inliers, candidate])
            loops_arr.append([n_idx, candidate])

    return fitted_candidates


@utils.measure_time
def q7_3(fitters, n_idx, pose_graph):
    """
    Relative Pose Estimation - using the inlier matches, perform a small Bundle
    optimization to extract the relative pose of the two frames as well as
    its covariance.
    """
    relatives = []

    for data in fitters:
        left0_inliers, left1_inliers, candidate = data
        loop_tracks = create_loop_tracks(left0_inliers, left1_inliers, candidate, n_idx)
        bundle = Bundle(candidate, n_idx, loop_tracks)
        rel_pose, rel_cov = pose_graph.rel_cov_and_pos_for_bundle(bundle)
        relatives.append((rel_pose, rel_cov, candidate))

    return relatives


@utils.measure_time
def q7_4(relatives, pose_graph, n_idx):
    """
    Update the Pose Graph - add the resulting synthetic measurement to the pose
    graph and optimize it to update the trajectory estimate.
    """
    for rel_pose, rel_cov, i in relatives:
        pose_graph.vertex_graph.add_edge(i, n_idx, utils.weight_func(rel_cov))
        cur_symbol = gtsam.symbol('c', n_idx)
        prev_symbol = gtsam.symbol('c', i)
        cov = gtsam.noiseModel.Gaussian.Covariance(rel_cov)
        factor = gtsam.BetweenFactorPose3(prev_symbol, cur_symbol, rel_pose, cov)
        pose_graph.graph.add(factor)

    pose_graph.solve()


@utils.measure_time
def q7_5():
    """
    Display plots.
    """
    print(len(loops_arr))


def run_ex7():
    """
    Runs all exercise 7 sections.
    """
    np.random.seed(1)
    # Load tracks DB
    tracks_db = TracksDB.deserialize(DB_PATH)
    T_arr = np.load(T_ARR_PATH)
    rel_t_arr = calculate_relative_transformations(T_arr)

    pose_graph = load_pos_graph('pose_graph.pkl')
    if pose_graph is None:
        pose_graph = PoseGraph(tracks_db, rel_t_arr, None, PoseGraph.choose_keyframes_median, **{'median': 0.6})
        pose_graph.solve()
        PoseGraph.save(pose_graph, 'pose_graph.pkl')

    print(f'pose graph has {len(pose_graph.keyframes)} keyframes and {len(pose_graph.tracks)} tracks')

    auxilery_plot_utils.plot_scene_from_above(pose_graph.result, marginals=pose_graph.get_marginals(), question='q7')

    # For each key frame cn in the pose graph loop over previous keyframes ci, i < n, and perform
    # steps 7.1-7.4:
    for i in range(1, len(pose_graph.keyframes)):
        # Detect Loop Closure Candidates
        candidates = q7_1(pose_graph, i)

        if not candidates:
            continue

        else:
            print(f'candidates for {i}: {candidates}')
            # Consensus Matching
            fitters = q7_2(candidates, i)

            # Relative Pose Estimation
            relatives = q7_3(fitters, i, pose_graph)

            # Update the Pose Graph
            q7_4(relatives, pose_graph, i)

    # Display Plots
    q7_5()


def main():
    run_ex7()


if __name__ == '__main__':
    main()
