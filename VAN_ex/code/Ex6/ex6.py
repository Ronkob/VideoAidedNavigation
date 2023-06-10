import os
import gtsam
import numpy as np

from gtsam.utils import plot
import matplotlib.pyplot as plt

from VAN_ex.code.Ex3.ex3 import calculate_relative_transformations
from VAN_ex.code.Ex4.ex4 import TracksDB, Track
from VAN_ex.code.Ex5.ex5 import plot_scene_3d
from VAN_ex.code.BundleAdjustment import BundleWindow
from VAN_ex.code.PoseGraph.PoseGraph import PoseGraph
from VAN_ex.code.utils import utils, projection_utils, auxilery_plot_utils, gtsam_plot_utils

DB_PATH = os.path.join('..', 'Ex4', 'tracks_db.pkl')
T_ARR_PATH = os.path.join('..', 'Ex3', 'T_arr.npy')


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
    plot_scene_3d(result, marginals=marginals, scale=10, question='q6_1')

    # Calculate the relative covariance between the first two keyframes
    keys = gtsam.KeyVector()
    keys.append(gtsam.symbol('c', c0))
    keys.append(gtsam.symbol('c', ck))
    marg_cov_mat = marginals.jointMarginalInformation(keys).at(keys[1], keys[1])  # Conditioning on the second keyframe
    rel_cov = np.linalg.inv(marg_cov_mat)  # Cov of the relative motion

    # Calculate the relative pose between the first two keyframes
    pose_c0 = result.atPose3(gtsam.symbol('c', c0))
    pose_ck = result.atPose3(gtsam.symbol('c', ck))
    rel_pose = pose_c0.between(pose_ck)

    # Print the resulting relative pose and the covariance associated.
    print("Relative pose between the first two keyframes:", rel_pose)
    print("Relative covariance between the first two keyframes:", rel_cov)


def q6_2(tracks_db, T_arr):
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
    cameras_trajectory = projection_utils.get_trajectory_from_gtsam_poses(
        keyframes_pose_graph.rel_poses)
    fig, axes = plt.subplots(figsize=(6, 6))
    fig = auxilery_plot_utils.plot_camera_trajectory(
        camera_pos=cameras_trajectory, fig=fig, label="Init Poses Supplied", color='blue')
    legend_element = plt.legend(loc='upper left', fontsize=12)
    fig.gca().add_artist(legend_element)
    fig.savefig('q6_2_Init_poses.png')
    fig.show()
    plt.clf()

    # Plot the locations without covariances for the keyframe locations
    # resulting from the optimization.
    opt_poses = gtsam.utilities.extractPose3(keyframes_pose_graph.result).reshape(-1, 4, 3).transpose(0, 2, 1)
    # TODO - Build 2d graph

    # Print the error of the factor graph before and after optimization
    print('Initial Error =', keyframes_pose_graph.get_graph_error(True))
    print('Final Error =', keyframes_pose_graph.get_graph_error(False))

    # Plot the locations with the marginal covariances
    marginals = keyframes_pose_graph.get_marginals()
    # TODO - Build 2d graph

def run_ex6():
    """
    Runs all exercise 6 sections.
    """
    np.random.seed(1)
    # Load tracks DB
    tracks_db = TracksDB.deserialize(DB_PATH)
    T_arr = np.load(T_ARR_PATH)
    rel_t_arr = calculate_relative_transformations(T_arr)

    # q6_1(rel_t_arr, tracks_db)
    q6_2(tracks_db, rel_t_arr)


def main():
    run_ex6()


if __name__ == '__main__':
    main()
