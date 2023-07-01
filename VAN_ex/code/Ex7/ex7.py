import os
import gtsam
import pickle
import numpy as np

from gtsam.utils import plot
import matplotlib.pyplot as plt
from tqdm import tqdm

from VAN_ex.code.Ex3 import ex3
from VAN_ex.code.PreCalcData.PreCalced import Data
from VAN_ex.code.utils import utils, projection_utils, auxilery_plot_utils
from VAN_ex.code.Ex3.ex3 import calculate_relative_transformations, track_movement_successive
from VAN_ex.code.DataBase.TracksDB import TracksDB, create_loop_tracks
from VAN_ex.code.DataBase.Track import Track
from VAN_ex.code.utils.auxilery_plot_utils import plot_scene_from_above, plot_scene_3d
from VAN_ex.code.BundleAdjustment.BundleWindow import Bundle
from VAN_ex.code.PoseGraph.PoseGraph import PoseGraph, save_pg, load_pg
from VAN_ex.code.PreCalcData import paths_to_data

from dijkstar import find_path

# Constants
MAHAL_THRESH = 50
MAX_CANDIDATES = 3
INLIERS_PREC_THRESH = 75

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

    if not candidate_frames:
        return []
    sorted_candidates = sorted(candidate_frames.items(), key=lambda x: x[1])
    return [item for item in sorted_candidates[:MAX_CANDIDATES]]  # only best 3 candidates


@utils.measure_time
def q7_2(candidates, n_idx, key_frames):
    """
    Consensus Matching - perform consensus match between the two candidate frames. (See exercise 3)
    Set a threshold for the number of inliers that indicates a successful match.
    Note that this is typically a more difficult match than that of two consecutive frames.
    """
    fitted_candidates = []
    for candidate in candidates:
        key_frame_num, mahalanobis_dist = candidate
        curr_frame = key_frames[n_idx]
        candidate_frame = key_frames[key_frame_num]
        left_ext_mat, cur_inliers, inliers_precent = track_movement_successive([candidate_frame, curr_frame],
                                                                               plot=False)
        if inliers_precent > INLIERS_PREC_THRESH:
            left0_inliers, right0_inliers, left1_inliers, right1_inliers = cur_inliers
            fitted_candidates.append([(left0_inliers, right0_inliers), (left1_inliers, right1_inliers), candidate])
            loops_arr.append([candidate_frame, curr_frame])

    return fitted_candidates


@utils.measure_time
def q7_3(fitters, n_idx, pose_graph: PoseGraph):
    """
    Relative Pose Estimation - using the inlier matches, perform a small Bundle
    optimization to extract the relative pose of the two frames as well as
    its covariance.
    """
    relatives = []
    key_frames = pose_graph.keyframes

    for data in fitters:
        first_inliers, second_inliers, candidate = data
        key_frame, mahalanobis_dist = candidate
        curr_frame, candidate_frame = key_frames[n_idx], key_frames[key_frame]
        loop_tracks_db = create_loop_tracks(first_inliers, second_inliers, candidate_frame, curr_frame)
        bundle = Bundle(candidate_frame, curr_frame, True)
        bundle.create_graph_v2(T_arr=pose_graph.T_arr, tracks_db=loop_tracks_db)
        bundle.optimize()
        rel_cov, rel_pose = pose_graph.rel_cov_and_pos_for_bundle(bundle)
        relatives.append((rel_cov, rel_pose, candidate))

    return relatives


@utils.measure_time
def q7_4(relatives, pose_graph, n_idx):
    """
    Update the Pose Graph - add the resulting synthetic measurement to the pose
    graph and optimize it to update the trajectory estimate.
    """
    for rel_cov, rel_pose, candidate in relatives:
        frame, mahalanobis_dist = candidate
        first_symbol = gtsam.symbol('c', frame)
        second_symbol = gtsam.symbol('c', n_idx)
        cov = gtsam.noiseModel.Gaussian.Covariance(rel_cov)
        factor = gtsam.BetweenFactorPose3(first_symbol, second_symbol, rel_pose, cov)
        pose_graph.graph.add(factor)
        pose_graph.vertex_graph.add_edge(frame, n_idx, utils.weight_func(rel_cov))

    pose_graph.solve()


@utils.measure_time
def q7_5():
    """
    Display plots.
    """

    # How many successful loop closures were detected?
    loops_array = np.load('loops_arr.npy', allow_pickle=True)
    print(f'Number of successful loop closures: {len(loops_array)}')
    print(loops_array)

    # Plot the match results of a single successful consensus match of your choice.
    # (For the left images, inliers and outliers in different colors)
    ex3.track_movement_successive([463, 1221], plot=True)

    # Choose 5 versions of the pose graph along the process and plot them (including location covariance).
    #     TODO: implement

    # Plot a graph of the absolute location error for the whole pose graph both with and without loop closures.
    #     TODO: implement

    # Plot a graph of the location uncertainty size for the whole pose graph both with and without loop closures.
    # (What measure of uncertainty size did you choose?)
    #     TODO: implement

    # Plot the pose graph locations along with the ground truth both with and without loop closures.
    # no_loop_closure = Data().get_pose_graph()
    # with_loop_closure = load_pg('pg_loop_closure.pkl')
    #
    # graphs_lst = [no_loop_closure, with_loop_closure]
    # titles = ['Bundle Adjustment', 'Loop Closure']
    # auxilery_plot_utils.plot_pose_graphs(graphs_lst, titles)


@utils.measure_time
def run_ex7():
    """
    Runs all exercise 7 sections.
    """
    np.random.seed(1)
    # Load tracks DB
    data = Data()
    pose_graph = data.get_pose_graph()

    print(f'pose graph has {len(pose_graph.keyframes)} keyframes and {len(pose_graph.tracks_db.tracks)} tracks')

    auxilery_plot_utils.plot_scene_from_above(pose_graph.result, question='q7 no marginals')

    # For each key frame cn in the pose graph loop over previous keyframes ci, i < n, and perform
    key_frames = pose_graph.keyframes
    # steps 7.1-7.4:
    for i in tqdm(range(1, len(key_frames))):
        # Detect Loop Closure Candidates
        candidates = q7_1(pose_graph, i)

        if not candidates:
            continue

        else:
            print(f'candidates for {i}: {candidates}')
            # Consensus Matching
            fitters = q7_2(candidates, i, key_frames)

            # Relative Pose Estimation
            relatives = q7_3(fitters, i, pose_graph)

            # Update the Pose Graph
            q7_4(relatives, pose_graph, i)

    # save the loop-closure pose graph to file
    save_pg(pose_graph, 'pg_loop_closure.pkl')

    # save loops_arr to file
    np.save('loops_arr.npy', np.array(loops_arr))

    print(f'pose graph has {len(pose_graph.keyframes)} keyframes and {len(pose_graph.tracks_db.tracks)} tracks')
    print('We found {} loop closures on frames'.format(len(loops_arr), loops_arr))


def main():
    # run_ex7()
    q7_5()


if __name__ == '__main__':
    main()
