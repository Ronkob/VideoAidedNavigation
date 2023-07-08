import math
import os
import gtsam
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from VAN_ex.code.BundleAdjustment.BundleWindow import Bundle
from VAN_ex.code.BundleAdjustment.BundleAdjustment import BundleAdjustment
from VAN_ex.code.DataBase.TracksDB import TracksDB
from VAN_ex.code.DataBase.Track import Track
from VAN_ex.code.PoseGraph.PoseGraph import PoseGraph, load_pg
from VAN_ex.code.PreCalcData.PreCalced import Data
from VAN_ex.code.utils import utils, projection_utils

import VAN_ex.code.Ex3.ex3 as ex3_utils

LOOPS_ARR_PATH = os.path.join('..', 'Ex7', 'loops_arr.npy')
PG_LOOP_PATH = os.path.join('..', 'Ex7', 'pg_loop_closure.pkl')
PG_PATH = os.path.join('..', 'Ex6', 'pg.pkl')


def rel_pose_move(pos):
    """
    Calculate the relative movement each two consecutive poses.
    """
    return np.abs(np.diff(pos, axis=0))


def proj_error_pnp():
    """
    Median (or any other meaningful statistic) projection error of the different track links as
    a function of distance from the reference frame (triangulation frame)
    """
    pass


def proj_error_ba():
    """
    Median (or any other meaningful statistic) projection error of the different track links as
    a function of distance from the reference frame (1st frame for Bundle)
    """
    pass


def factor_error_pnp():
    """
    Median (or any other meaningful statistic) factor error of the different track links as a
    function of distance from the reference frame.
    """
    pass


def factor_error_ba():
    """
    Same as above for Bundle Adjustment.
    """
    pass


def abs_pnp_est_error():
    """
    Absolute PnP estimation error: X axis error, Y axis error, Z axis error,
    Total error norm, Angle error.
    """
    data = Data()
    rel_pnp_matrices = data.get_rel_T_arr()
    ground_truth = np.array(ex3_utils.get_ground_truth_transformations())
    gt_locations = utils.calculate_camera_trajectory(ground_truth)
    pnp_locations = utils.calculate_camera_trajectory(rel_pnp_matrices)
    diff_x = pnp_locations[:, 0] - gt_locations[:, 0]
    diff_y = pnp_locations[:, 1] - gt_locations[:, 1]
    diff_z = pnp_locations[:, 2] - gt_locations[:, 2]
    total_error_norm = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

    # Plot location error
    fig = plt.figure(figsize=(12, 5))
    plt.ylabel('Absolute location Error')
    plt.xlabel('Frame')
    plt.title('Absolute PnP location estimation error')
    plt.plot(range(len(diff_x)), abs(diff_x), label='X axis error')
    plt.plot(range(len(diff_y)), abs(diff_y), label='Y axis error')
    plt.plot(range(len(diff_z)), abs(diff_z), label='Z axis error')
    plt.plot(range(len(total_error_norm)), total_error_norm, label='Total error norm')
    plt.legend()
    fig.savefig('Absolute PnP estimation error.png')
    plt.close(fig)

    # Plot angle error
    fig = plt.figure(figsize=(12, 5))
    plt.ylabel('Absolute angle Error')
    plt.xlabel('Frame')
    plt.title('Absolute PnP angle estimation error')
    pnp_angles = utils.calculate_camera_angles(rel_pnp_matrices)
    gt_angles = utils.calculate_camera_angles(ground_truth)
    yz_error = abs(pnp_angles[:, 0] - gt_angles[:, 0]) * 180 / np.pi
    xz_error = abs(pnp_angles[:, 1] - gt_angles[:, 1]) * 180 / np.pi
    xy_error = abs(pnp_angles[:, 2] - gt_angles[:, 2]) * 180 / np.pi
    plt.plot(range(len(yz_error)), yz_error, label='YZ error')
    plt.plot(range(len(xz_error)), xz_error, label='XZ error')
    plt.plot(range(len(xy_error)), xy_error, label='XY error')
    plt.legend()
    fig.savefig('Absolute PnP angle estimation error.png')
    plt.close(fig)


def abs_pose_est_error(angle=False):
    """
    Absolute Pose Graph estimation error: X axis error, Y axis error, Z axis error,
    Total error norm, Angle error (With and Without loop closure).
    """
    word = 'Angle' if angle else 'Norm'
    print('Plotting Absolute Pose Graph {} estimation error With and Without Loop Closure...'.format(word))
    pg_with_loop = load_pg(PG_LOOP_PATH)
    keyframes = pg_with_loop.keyframes
    ground_truth_keyframes = np.array(ex3_utils.get_ground_truth_transformations())[keyframes]
    loop_rel_cameras = pg_with_loop.get_opt_cameras()

    fig = plt.figure(figsize=(12, 5))
    plt.title('Absolute Pose Graph {} estimation error With and Without Loop Closure'.format(word))
    plt.ylabel('Absolute {} Error'.format(angle))
    plt.xlabel('Keyframe')
    if angle:
        loop_angles = np.array([utils.rotation_matrix_to_euler_angles(cam.rotation().matrix()) for cam in loop_rel_cameras])
        gt_angles = utils.calculate_camera_angles(ground_truth_keyframes)
        yz_error = abs(loop_angles[:, 0] - gt_angles[:, 0]) * 180 / np.pi
        xz_error = abs(loop_angles[:, 1] - gt_angles[:, 1]) * 180 / np.pi
        xy_error = abs(loop_angles[:, 2] - gt_angles[:, 2]) * 180 / np.pi
        plt.plot(range(len(yz_error)), yz_error, label='YZ error With')
        plt.plot(range(len(xz_error)), xz_error, label='XZ error With')
        plt.plot(range(len(xy_error)), xy_error, label='XY error With')
    else:
        gt_locations = utils.calculate_camera_trajectory(ground_truth_keyframes)
        loop_trajectory = projection_utils.get_trajectory_from_gtsam_poses(loop_rel_cameras)
        x_diff = np.abs(loop_trajectory[:, 0]-gt_locations[:, 0])
        y_diff = np.abs(loop_trajectory[:, 1]-gt_locations[:, 1])
        z_diff = np.abs(loop_trajectory[:, 2]-gt_locations[:, 2])
        total_error = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
        plt.plot(range(len(x_diff)), x_diff, label='X axis error With')
        plt.plot(range(len(y_diff)), y_diff, label='Y axis error With')
        plt.plot(range(len(z_diff)), z_diff, label='Z axis error With')
        plt.plot(range(len(total_error)), total_error, label='Total norm error With')
    print('Finished first PG')
    del pg_with_loop

    pg_no_loop = load_pg(PG_PATH)
    init_rel_cameras = pg_no_loop.get_opt_cameras()
    if angle:
        no_loop_angles = np.array([utils.rotation_matrix_to_euler_angles(cam.rotation().matrix()) for cam in init_rel_cameras])
        yz_error = abs(no_loop_angles[:, 0] - gt_angles[:, 0]) * 180 / np.pi
        xz_error = abs(no_loop_angles[:, 1] - gt_angles[:, 1]) * 180 / np.pi
        xy_error = abs(no_loop_angles[:, 2] - gt_angles[:, 2]) * 180 / np.pi
        plt.plot(range(len(yz_error)), yz_error, label='YZ error Without')
        plt.plot(range(len(xz_error)), xz_error, label='XZ error Without')
        plt.plot(range(len(xy_error)), xy_error, label='XY error Without')
    else:
        no_loop_trajectory = projection_utils.get_trajectory_from_gtsam_poses(init_rel_cameras)
        x_diff = np.abs(no_loop_trajectory[:, 0]-gt_locations[:, 0])
        y_diff = np.abs(no_loop_trajectory[:, 1]-gt_locations[:, 1])
        z_diff = np.abs(no_loop_trajectory[:, 2]-gt_locations[:, 2])
        total_error = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
        plt.plot(range(len(x_diff)), x_diff, label='X axis error Without')
        plt.plot(range(len(y_diff)), y_diff, label='Y axis error Without')
        plt.plot(range(len(z_diff)), z_diff, label='Z axis error Without')
        plt.plot(range(len(total_error)), total_error, label='Total norm error Without')

    plt.legend()
    fig.savefig('Absolute Pose Graph {} estimation error With and Without Loop Closure'.format(word))
    plt.close(fig)
    print('Plot saved')


def rel_pnp_est_error():
    """
    The error of the relative pose estimation compared to the ground truth relative pose,
    evaluated on sequence lengths of (100, 300, 800).
    o X axis, Y axis, Z axis, Total error norm (measure as error%: m/m)
    o Angle error (measure as deg/m)
    o For each graph calculate the average error of all the sequences for total norm
    and angle error (a single number for each).
    """
    seq_len = [100, 300, 800]
    rel_pnp_matrices = Data().get_rel_T_arr()
    ground_truth = np.array(ex3_utils.get_ground_truth_transformations())
    gt_locations = utils.calculate_camera_trajectory(ground_truth)
    pnp_locations = utils.calculate_camera_trajectory(rel_pnp_matrices)
    x_error, y_error, z_error, total_error = [], [], [], []

    for length in seq_len:
        # Take the last frames for better observation of the error
        for i in range(len(gt_locations)-length):
            relative_poses = rel_pose_move(pnp_locations[i:i+length])
            gt_relative_poses = rel_pose_move(gt_locations[i:i+length])
            x_error.append(np.sum(np.abs(relative_poses[:, 0]-gt_relative_poses[:, 0])) / np.sum(gt_relative_poses[:, 0]))
            y_error.append(np.sum(np.abs(relative_poses[:, 1]-gt_relative_poses[:, 1])) / np.sum(gt_relative_poses[:, 1]))
            z_error.append(np.sum(np.abs(relative_poses[:, 2]-gt_relative_poses[:, 2])) / np.sum(gt_relative_poses[:, 2]))
            total_error.append(np.sqrt(x_error[-1]**2 + y_error[-1]**2 + z_error[-1]**2))

        fig = plt.figure()
        plt.title('Relative PnP pose estimation on sequence length {}'.format(length))
        plt.plot(range(len(x_error)), x_error, label='X relative error %')
        plt.plot(range(len(y_error)), y_error, label='Y relative error %')
        plt.plot(range(len(z_error)), z_error, label='Z relative error %')
        plt.plot(range(len(total_error)), total_error, label="Total error norm %")
        plt.ylabel("Error Percentage")
        plt.xlabel("Frame")
        plt.legend()
        fig.savefig('Relative PnP pose estimation on sequence length {}.png'.format(length))
        plt.close(fig)

    # TODO - Add angle error graph + Average


def rel_bundle_est_error():
    """
    Same as above for Bundle Adjustment.
    """
    seq_len = [100, 300, 800]
    ba = Data().get_ba()
    print('Finished loading BA')
    ba_cameras, _ = ba.get_relative_poses()
    ground_truth = np.array(ex3_utils.get_ground_truth_transformations())
    gt_locations = utils.calculate_camera_trajectory(ground_truth)
    x_error, y_error, z_error, total_error = [], [], [], []

    for length in seq_len:
        # Take the last frames for better observation of the error
        for i in range(len(gt_locations)-length):
            relative_poses = rel_pose_move(ba_cameras[i:i+length])
            gt_relative_poses = rel_pose_move(gt_locations[i:i+length])
            x_error.append(np.sum(np.abs(relative_poses[:, 0]-gt_relative_poses[:, 0])) / np.sum(gt_relative_poses[:, 0]))
            y_error.append(np.sum(np.abs(relative_poses[:, 1]-gt_relative_poses[:, 1])) / np.sum(gt_relative_poses[:, 1]))
            z_error.append(np.sum(np.abs(relative_poses[:, 2]-gt_relative_poses[:, 2])) / np.sum(gt_relative_poses[:, 2]))
            total_error.append(np.sqrt(x_error[-1]**2 + y_error[-1]**2 + z_error[-1]**2))

        fig = plt.figure()
        plt.title('Relative BA pose estimation on sequence length {}'.format(length))
        plt.plot(range(len(x_error)), x_error, label='X relative error %')
        plt.plot(range(len(y_error)), y_error, label='Y relative error %')
        plt.plot(range(len(z_error)), z_error, label='Z relative error %')
        plt.plot(range(len(total_error)), total_error,
                 label="Total error norm %")
        plt.ylabel("Error Percentage")
        plt.xlabel("Frame")
        plt.legend()
        fig.savefig('Relative BA pose estimation on sequence length {}.png'.format(length))
        plt.close(fig)

    # TODO - Add angle error graph + Average


def uncertainty_vs_kf():
    """
    Uncertainty size vs keyframe – pose graph with or without loop closure:
    o Location Uncertainty
    o Angle Uncertainty
    """
    # How did you measure uncertainty size?  Determinant of cov matrix.
    # How did you isolate the different parts of the uncertainty? Idk.
    no_loop_closure = load_pg(PG_PATH)
    keyframes = no_loop_closure.keyframes
    init_covs = [no_loop_closure.get_marginals().marginalCovariance(gtsam.symbol('c', i)) for i in range(len(keyframes))]
    weight_init_covs = [utils.weight_func(cov) for cov in init_covs]
    del no_loop_closure

    print('getting pose graph after loop closure...')
    with_loop_closure = load_pg(PG_LOOP_PATH)
    after_loop_covs = [with_loop_closure.get_marginals().marginalCovariance(gtsam.symbol('c', i)) for i in range(len(keyframes))]
    weight_loop_covs = [utils.weight_func(cov) for cov in after_loop_covs]

    fig = plt.figure()
    plt.title('Location uncertainty size with and without Loop Closure')
    plt.plot(range(len(weight_init_covs)), weight_init_covs, label='Without Loop', color='orange')
    plt.plot(range(len(weight_loop_covs)), weight_loop_covs, label='With Loop')
    plt.ylabel('Covariance Sqrt Determinant')
    plt.xlabel('Keyframe')
    plt.yscale('log')
    plt.legend()
    fig.savefig('Location uncertainty size.png')

    # TODO - Add angles


def plot_loops_matches():
    """
    Plot number of matches per successful loop closure frame, and inlier
    percentage per successful loop closure frame.
    """
    print('Plotting loops matches and inliers percentage')
    loops_array = np.load(LOOPS_ARR_PATH, allow_pickle=True)
    loops = [i for i in range(len(loops_array))]
    num_matches, inliers_prec = [], []
    for loop in tqdm(loops_array):
        frame1, frame2 = loop
        _, inliers, inliers_precent = ex3_utils.track_movement_successive([frame1, frame2])
        num_matches.append(len(inliers[0]))
        inliers_prec.append(inliers_precent)

    fig = plt.figure()
    plt.title('Inlier percentage per successful loop closure frame')
    plt.xlabel('Loop number')
    plt.ylabel('Inliers Percentage')
    plt.plot(loops, inliers_prec)
    plt.savefig('inliers_prec_per_loop.png')
    plt.close(fig)

    fig = plt.figure()
    plt.title('Matcehs per successful loop closure frame')
    plt.xlabel('Loop number')
    plt.ylabel('Matches')
    loops = [i for i in range(len(loops_array))]
    plt.plot(loops, num_matches)
    plt.savefig('matches_per_loop.png')
    plt.close(fig)
    print('Done plotting loops matches and inliers percentage')


def make_plots():
    """
    Make plots needed for final project submission.
    """
    # proj_error_pnp()
    # proj_error_ba()
    # factor_error_pnp()
    # factor_error_ba()
    # abs_pnp_est_error()  # No Angles
    # abs_pose_est_error(True)  # No Angles
    # abs_pose_est_error(False)  # V
    # rel_pnp_est_error()  # No Angles
    # rel_bundle_est_error()  # No Angles
    # plot_loops_matches()  # V
    # uncertainty_vs_kf()  # No Angles


def main():
    make_plots()


if __name__ == '__main__':
    main()