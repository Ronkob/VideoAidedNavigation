import os
import gtsam
import pickle
import numpy as np

from gtsam.utils import plot
import matplotlib.pyplot as plt

from VAN_ex.code.BundleAdjustment.BundleAdjustment import BundleAdjustment
from VAN_ex.code.Ex3.ex3 import calculate_relative_transformations
from VAN_ex.code.DataBase.TracksDB import TracksDB
from VAN_ex.code.DataBase.Track import Track
from VAN_ex.code.PreCalcData.paths_to_data import BA_PATH, DB_PATH, T_ARR_PATH
from VAN_ex.code.utils.auxilery_plot_utils import plot_scene_from_above, plot_scene_3d
from VAN_ex.code.BundleAdjustment import BundleWindow
from VAN_ex.code.PoseGraph.PoseGraph import PoseGraph
from VAN_ex.code.utils import projection_utils, auxilery_plot_utils

def q6_1(T_arr, tracks_db):
    """
    Extract relative pose constraint from Bundle optimization.
    Calculate the correct covariance matrix of the relative motion (synthetic) measurement.
    The covariance should be the relative covariance between the frame poses.
     i.e. the covariance associated with the distribution P(ck|c0).
    """
    c0, ck = 0, 7  # First two keyframes
    first_window = BundleWindow.Bundle(c0, ck)
    first_window.create_graph_v2(T_arr, tracks_db)
    result = first_window.optimize()  # Optimize Locations and Landmarks

    # Extract the marginal covariances of the solution
    marginals = first_window.get_marginals()

    # Plot the resulting frame locations as a 3D graph including the covariance
    # of the locations (all the frames in the bundle window).
    plot_scene_3d(result, marginals=marginals, scale=1, question='q6_1')

    # Calculate the relative covariance between the first two keyframes
    keys = gtsam.KeyVector()
    keys.append(gtsam.symbol('c', c0))
    keys.append(gtsam.symbol('c', ck))
    # rel_cov = marginals.jointMarginalCovariance(keys).at(keys[1], keys[1])  # Cov of the relative motion
    marg_cov_mat = marginals.jointMarginalInformation(keys).at(keys[1], keys[1])  # Conditioning on the second keyframe
    rel_cov = np.linalg.inv(marg_cov_mat)  # Cov of the relative motion, as seen in lecture

    # Calculate the relative pose between the first two keyframes
    pose_c0 = result.atPose3(gtsam.symbol('c', c0))
    pose_ck = result.atPose3(gtsam.symbol('c', ck))
    rel_pose = pose_c0.between(pose_ck)

    # Print the resulting relative pose and the covariance associated.
    print("Relative pose between the first two keyframes:", rel_pose)
    print("Relative covariance between the first two keyframes:", rel_cov)


def q6_2(T_arr, tracks_db, ba):
    """
    Build a Pose Graph of the keyframes. We add the relative motion
    estimated previously as constraints to that graph with the correct uncertainty
    (covariance matrix) for each constraint.
    Construct poses for the initial guess of the pose graph and optimize it.
    """
    # Build the Pose Graph of the keyframes
    keyframes_pose_graph = PoseGraph(tracks_db, T_arr)
    keyframes_pose_graph.solve()

    # Plot the initial poses you supplied the optimization.
    plot_scene_from_above(keyframes_pose_graph.initial_estimates, question='q6_2 initial poses')

    # Plot the locations without covariances for the keyframe locations resulting from the optimization.
    plot_scene_from_above(keyframes_pose_graph.result, question='q6_2 optimized poses')

    # Print the error of the factor graph before and after optimization
    print('Initial Error =', keyframes_pose_graph.get_graph_error(True))
    print('Final Error =', keyframes_pose_graph.get_graph_error(False))

    # Plot the locations with the marginal covariances
    marginals = keyframes_pose_graph.get_marginals()
    plot_scene_from_above(keyframes_pose_graph.result, marginals=marginals, question='q6_2 optimized poses with cov')


def run_ex6():
    """
    Runs all exercise 6 sections.
    """
    np.random.seed(1)
    # Load tracks DB
    tracks_db = TracksDB.deserialize(DB_PATH)
    T_arr = np.load(T_ARR_PATH)
    rel_t_arr = calculate_relative_transformations(T_arr)
    ba = None
    # ba = BundleAdjustment.deserialize(BA_PATH)

    q6_1(rel_t_arr, tracks_db)
    # q6_2(rel_t_arr, tracks_db, ba)


def main():
    run_ex6()


if __name__ == '__main__':
    main()