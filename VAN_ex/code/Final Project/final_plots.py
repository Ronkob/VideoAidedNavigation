import math
import os
import statistics

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
from VAN_ex.code.utils import utils, projection_utils, auxilery_plot_utils
from VAN_ex.code.Ex5 import ex5 as ex5_utils

import VAN_ex.code.Ex3.ex3 as ex3_utils

LOOPS_ARR_PATH = os.path.join('..', 'Ex7', 'loops_arr.npy')
PG_LOOP_PATH = os.path.join('..', 'Ex7', 'pg_loop_closure.pkl')
PG_PATH = os.path.join('..', 'Ex6', 'pg.pkl')

""" Absolute Estimation Error Plots """


def abs_pose_est_error():
    # plot graph for each mode of the system
    # result_pnp,  yz_error_pnp, xz_error_pnp, xy_error_pnp = abs_pnp_est_error()
    result_no_loop_pose = abs_no_loop_closure_pose_est_error(angle=False)
    yz_error_ba, xz_error_ba, xy_error_ba = abs_no_loop_closure_pose_est_error(angle=True)
    result_loop_pose = abs_with_loop_closure_pose_est_error(angle=False)
    yz_error_loop, xz_error_loop, xy_error_loop = abs_with_loop_closure_pose_est_error(angle=True)

    # plot a comparison graph for each mode of the system with the total errors
    # the graph should have 2 y axis, one for the total error and one for the angle error
    # the x axis should be the frame number
    # the title should be "Absolute Pose Estimation Error"
    # the legend should be "Total Error" and "Angle Error"
    # make sure that each y axis has a different color

    fig, ax1 = plt.subplots(figsize=(12, 5))
    # ax1.plot(range(len(yz_error_pnp)), xz_error_pnp, label='PnP')
    ax1.plot(range(len(yz_error_ba)), xz_error_ba, label='No Loop Closure Angle Error', color='red')
    ax1.plot(range(len(yz_error_loop)), xz_error_loop, label='With Loop Closure Angle Error', color='blue')
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Frame [#]')
    ax1.set_ylabel('Total Error [m]')
    # ax1.plot(range(len(result_pnp)), result_pnp, label='PnP')
    ax2 = ax1.twinx()
    ax2.plot(range(len(result_no_loop_pose)), result_no_loop_pose, label='No Loop Closure Total Error', color='orange')
    ax2.plot(range(len(result_loop_pose)), result_loop_pose, label='With Loop Closure Total Error', color='green')
    ax2.legend(loc='upper left')
    ax2.set_ylabel('Angle Error [deg]')
    fig.savefig('Absolute Pose Estimation Error - Comparative.png')
    plt.close(fig)


def abs_pnp_est_error():
    """
    Absolute PnP estimation error: X axis error, Y axis error, Z axis error,
    Total error norm, Angle error.
    """
    print('Calculating absolute PnP estimation error...')
    data = Data()
    rel_pnp_matrices = data.get_rel_T_arr()
    ground_truth = np.array(ex3_utils.get_ground_truth_transformations())
    gt_locations = utils.calculate_camera_trajectory(ground_truth)
    pnp_locations = utils.calculate_camera_trajectory(rel_pnp_matrices)
    diff_x = pnp_locations[:, 0] - gt_locations[:, 0]
    diff_y = pnp_locations[:, 1] - gt_locations[:, 1]
    diff_z = pnp_locations[:, 2] - gt_locations[:, 2]
    total_error_norm = np.sqrt(diff_x ** 2 + diff_y ** 2 + diff_z ** 2)

    # Plot location error
    fig = plt.figure(figsize=(12, 5))
    plt.ylabel('Absolute location Error [m]')
    plt.xlabel('Frame [#]')
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
    plt.ylabel('Absolute angle Error [deg]')
    plt.xlabel('Frame [#]')
    plt.title('Absolute PnP angle estimation error')
    pnp_angles = utils.calculate_camera_angles(rel_pnp_matrices) * 180 / np.pi
    gt_angles = utils.calculate_camera_angles(ground_truth) * 180 / np.pi

    yz_error, xz_error, xy_error = calc_angle_diffs(pnp_angles, gt_angles)

    plt.plot(range(len(yz_error)), yz_error, label='YZ error')
    plt.plot(range(len(xz_error)), xz_error, label='XZ error')
    plt.plot(range(len(xy_error)), xy_error, label='XY error')
    plt.legend()
    fig.savefig('Absolute PnP angle estimation error.png')
    plt.close(fig)
    return total_error_norm, yz_error, xz_error, xy_error


def abs_with_loop_closure_pose_est_error(angle=False):
    """
    Absolute Pose Graph estimation error: X axis error, Y axis error, Z axis error,
    Total error norm, Angle error (With and Without loop closure).
    """
    word = 'Angle' if angle else 'Norm'
    print('Plotting Absolute Pose Graph {} estimation error With Loop Closure...'.format(word))
    pg_with_loop = load_pg(PG_LOOP_PATH)
    keyframes = pg_with_loop.keyframes
    ground_truth_keyframes = np.array(ex3_utils.get_ground_truth_transformations())[keyframes]
    gt_angles = utils.calculate_camera_angles(ground_truth_keyframes) * 180 / np.pi
    gt_locations = utils.calculate_camera_trajectory(ground_truth_keyframes)
    loop_rel_cameras = pg_with_loop.get_opt_cameras()

    fig = plt.figure(figsize=(12, 5))
    plt.title('Absolute Pose Graph {} estimation error With Loop Closure'.format(word))
    plt.ylabel(f'Absolute {angle} Error [deg]' if angle else f'Absolute {angle} Error [m]')
    plt.xlabel('Keyframe [#]')
    if angle:
        loop_angles = np.array(
            [utils.rotation_matrix_to_euler_angles(cam.rotation().matrix()) for cam in loop_rel_cameras]) * 180 / np.pi
        yz_error, xz_error, xy_error = calc_angle_diffs(loop_angles, gt_angles)
        plt.plot(range(len(yz_error)), yz_error, label='YZ error')
        plt.plot(range(len(xz_error)), xz_error, label='XZ error')
        plt.plot(range(len(xy_error)), xy_error, label='XY error')
    else:
        loop_trajectory = projection_utils.get_trajectory_from_gtsam_poses(loop_rel_cameras)
        x_diff = np.abs(loop_trajectory[:, 0] - gt_locations[:, 0])
        y_diff = np.abs(loop_trajectory[:, 1] - gt_locations[:, 1])
        z_diff = np.abs(loop_trajectory[:, 2] - gt_locations[:, 2])
        total_error = np.sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)
        plt.plot(range(len(x_diff)), x_diff, label='X axis error')
        plt.plot(range(len(y_diff)), y_diff, label='Y axis error')
        plt.plot(range(len(z_diff)), z_diff, label='Z axis error')
        plt.plot(range(len(total_error)), total_error, label='Total norm error With')
    print('Finished first PG')
    del pg_with_loop
    plt.legend()
    fig.savefig('Absolute Pose Graph {} estimation error With Loop Closure'.format(word))
    plt.close(fig)
    print('Plot saved')
    if angle:
        return yz_error, xz_error, xy_error
    else:
        return total_error


def abs_no_loop_closure_pose_est_error(angle=False):
    """
    Absolute Pose Graph estimation error: X axis error, Y axis error, Z axis error,
    Total error norm, Angle error (With and Without loop closure).
    """
    word = 'Angle' if angle else 'Norm'
    print('Plotting Absolute Pose Graph {} estimation error Without Loop Closure...'.format(word))
    pg_no_loop = load_pg(PG_PATH)
    opt_cameras = pg_no_loop.get_opt_cameras()
    keyframes = pg_no_loop.keyframes
    ground_truth_by_keyframes = np.array(ex3_utils.get_ground_truth_transformations())[keyframes]
    gt_angles = utils.calculate_camera_angles(ground_truth_by_keyframes) * 180 / np.pi
    gt_locations = utils.calculate_camera_trajectory(ground_truth_by_keyframes)

    fig = plt.figure(figsize=(12, 5))
    plt.title('Absolute Pose Graph {} estimation error Without Loop Closure'.format(word))
    plt.ylabel(f'Absolute {angle} Error [deg]' if angle else f'Absolute {angle} Error [m]')
    plt.xlabel('Keyframe [#]')

    if angle:
        camera_rel_matrices = [cam.rotation().matrix() for cam in opt_cameras]
        no_loop_angles = utils.calculate_camera_angles(camera_rel_matrices) * 180 / np.pi
        yz_error, xz_error, xy_error = calc_angle_diffs(no_loop_angles, gt_angles)
        plt.plot(range(len(yz_error)), yz_error, label='YZ error')
        plt.plot(range(len(xz_error)), xz_error, label='XZ error')
        plt.plot(range(len(xy_error)), xy_error, label='XY error')
    else:
        no_loop_trajectory = projection_utils.get_trajectory_from_gtsam_poses(opt_cameras)
        x_diff = np.abs(no_loop_trajectory[:, 0] - gt_locations[:, 0])
        y_diff = np.abs(no_loop_trajectory[:, 1] - gt_locations[:, 1])
        z_diff = np.abs(no_loop_trajectory[:, 2] - gt_locations[:, 2])
        total_error = np.sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)
        plt.plot(range(len(x_diff)), x_diff, label='X axis error')
        plt.plot(range(len(y_diff)), y_diff, label='Y axis error')
        plt.plot(range(len(z_diff)), z_diff, label='Z axis error')
        plt.plot(range(len(total_error)), total_error, label='Total norm error Without')

    plt.legend()
    fig.savefig('Absolute Pose Graph {} estimation error Without Loop Closure'.format(word))
    plt.close(fig)
    print('Plot saved')
    if not angle:
        return total_error
    else:
        return yz_error, xz_error, xy_error


""" Relative Estimation Error Plots """


def rel_pose_move(pos):
    """
    Calculate the relative movement each two consecutive poses.
    """
    return np.abs(np.diff(pos, axis=0))


def calc_total_error(seq_lenths, total_len, est_locations, gt_locations, kf=None):
    total_error = [[] for _ in range(len(seq_lenths))]
    start_locs = [[] for _ in range(len(seq_lenths))]
    for i, seq_len in enumerate(seq_lenths):
        for d_from_start in range(0, total_len - seq_len, seq_len // 8):
            end_of_seq = d_from_start + seq_len
            start_of_seq = d_from_start

            start_loc = d_from_start
            end_loc = end_of_seq
            if kf:
                # the index of the element in kf that is bigger than d_from_start, and closest to end_of_seq
                # find the index of the closest kf element to d_from_start
                start_loc = np.searchsorted(kf, d_from_start, side='left')
                end_loc = np.searchsorted(kf, end_of_seq, side='left')
                end_of_seq = kf[end_loc]
                start_of_seq = kf[start_loc]

            A = np.abs(est_locations[end_loc] - est_locations[start_loc])
            B = np.abs(gt_locations[end_of_seq] - gt_locations[start_of_seq])
            C = np.sum(np.linalg.norm(np.diff(gt_locations[d_from_start:d_from_start + seq_len + 1], axis=0), axis=1))
            total_error[i].append(np.linalg.norm(A - B) / C)
            start_locs[i].append(start_of_seq)
    return total_error, start_locs


def calc_total_angle_error(seq_lenths, total_len, est_matrices, gt_matrices, gt_locations, kf=None):
    angle_error = [[] for _ in range(len(seq_lenths))]
    start_locs = [[] for _ in range(len(seq_lenths))]
    for i, seq_len in enumerate(seq_lenths):
        for d_from_start in range(0, total_len - seq_len, seq_len // 8):
            end_of_seq = d_from_start + seq_len
            start_of_seq = d_from_start

            start_loc = d_from_start
            end_loc = end_of_seq
            if kf:
                # the index of the element in kf that is bigger than d_from_start, and closest to end_of_seq
                # find the index of the closest kf element to d_from_start
                start_loc = np.searchsorted(kf, d_from_start, side='left')
                end_loc = np.searchsorted(kf, end_of_seq, side='left')
                end_of_seq = kf[end_loc]
                start_of_seq = kf[start_loc]

            angle_diff_gt = utils.rotation_matrix_to_euler_angles(
                projection_utils.composite_transformations(gt_matrices[end_of_seq],
                                                           gt_matrices[start_of_seq])) * 180 / np.pi
            if isinstance(est_matrices[start_loc], gtsam.Pose3):
                angle_diff_est = est_matrices[end_loc].rotation().between(
                    est_matrices[start_loc].rotation()).xyz() * 180 / np.pi
            else:
                angle_diff_est = utils.rotation_matrix_to_euler_angles(
                    projection_utils.composite_transformations(est_matrices[end_loc],
                                                               est_matrices[start_loc])) * 180 / np.pi
            C = np.sum(np.linalg.norm(np.diff(gt_locations[d_from_start:d_from_start + seq_len + 1], axis=0), axis=1))
            angle_error[i].append(np.linalg.norm(angle_diff_gt - angle_diff_est) / C)
            start_locs[i].append(start_of_seq)
    return angle_error, start_locs


def rel_est_error_v2(model, gt_locations, est_locations, gt_matrices=None, est_matrices=None, kf=None):
    seq_lenths = [100, 300, 800]
    total_len = len(gt_locations)
    total_error, start_locs = calc_total_error(seq_lenths, total_len, est_locations, gt_locations, kf)
    total_angles_error, start_locs_angles, = calc_total_angle_error(seq_lenths, total_len, est_matrices, gt_matrices,
                                                                    gt_locations, kf)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    plt.title(f'Relative {model} pose estimation on variable sequence length v2')
    ax2 = ax1.twinx()
    for i, seq_len in enumerate(seq_lenths):
        print(f"Total error on sequence length {seq_len}: {np.mean(total_error[i])}")
        ax1.plot(start_locs[i], total_error[i], label="Total error on sequence length {}".format(seq_len),
                 linestyle='-.')

    for i, seq_len in enumerate(seq_lenths):
        print(f"Total angle error on sequence length {seq_len}: {np.mean(total_angles_error[i])}")
        ax2.plot(start_locs_angles[i], total_angles_error[i],
                 label="Total angle error on sequence length {}".format(seq_len), linestyle='-')

    # set up neat croppings of the different y axes, based on the data
    max_x_cut = total_len - max(seq_lenths)
    plt.xlim(0, max_x_cut)

    flat_loc_error = [item for sublist in total_error for item in sublist]
    flat_angle_error = [item for sublist in total_angles_error for item in sublist]

    ax1.set_ylim(0, np.percentile(flat_loc_error, 99) * 1.1)
    ax2.set_ylim(0, np.percentile(flat_angle_error, 99) * 1.1)
    ax1.legend(loc='upper left')
    ax1.set_ylabel("Total Error [m/m]")
    ax2.legend(loc='upper right')
    ax2.set_ylabel("Angle Error [deg/m]")
    fig.savefig(f'Relative {model} Total pose estimation error on variable sequence length v3')
    plt.close(fig)


def rel_pnp_est_error_v2():
    rel_pnp_matrices = Data().get_rel_T_arr()
    ground_truth = np.array(ex3_utils.get_ground_truth_transformations())
    gt_locations = utils.calculate_camera_trajectory(ground_truth)
    pnp_locations = utils.calculate_camera_trajectory(rel_pnp_matrices)
    rel_est_error_v2('PnP', gt_locations, pnp_locations, ground_truth, rel_pnp_matrices)


def rel_ba_est_error_v2():
    ba = Data().get_ba()
    keyframes = ba.keyframes
    ground_truth = ex3_utils.get_ground_truth_transformations()
    ground_truth_keyframes = np.array(ground_truth)
    gt_locations = utils.calculate_camera_trajectory(ground_truth_keyframes)
    rel_est_matrices = projection_utils.convert_rel_gtsam_trans_to_global(ba.cameras_rel_pose)
    ba_locations = projection_utils.get_trajectory_from_gtsam_poses(rel_est_matrices)
    print(ba_locations.shape, gt_locations.shape)
    rel_est_error_v2('BA', gt_locations, ba_locations, ground_truth, rel_est_matrices, keyframes)


def rel_pnp_est_error():
    """
    The error of the relative pose estimation compared to the ground truth relative pose,
    evaluated on sequence lengths of (100, 300, 800).
    o X axis, Y axis, Z axis, Total error norm (measure as error%: m/m)
    o Angle error (measure as deg/m)
    o For each graph calculate the average error of all the sequences for total norm
    and angle error (a single number for each).
    """
    seq_lenths = [100, 300, 800]
    rel_pnp_matrices = Data().get_rel_T_arr()
    ground_truth = np.array(ex3_utils.get_ground_truth_transformations())
    gt_locations = utils.calculate_camera_trajectory(ground_truth)
    pnp_locations = utils.calculate_camera_trajectory(rel_pnp_matrices)
    total_error = [[]] * len(seq_lenths)
    angle_error = [[]] * len(seq_lenths)

    total_len = len(gt_locations)
    for i, seq_len in enumerate(seq_lenths):
        x_error, y_error, z_error = [], [], []
        # Take the last frames for better observation of the error
        for d_to_end in range(total_len - seq_len):
            relative_poses = rel_pose_move(pnp_locations[d_to_end:d_to_end + seq_len])
            gt_relative_poses = rel_pose_move(gt_locations[d_to_end:d_to_end + seq_len])
            x_error.append(
                np.sum(np.abs(relative_poses[:, 0] - gt_relative_poses[:, 0])) / np.sum(gt_relative_poses[:, 0]))
            y_error.append(
                np.sum(np.abs(relative_poses[:, 1] - gt_relative_poses[:, 1])) / np.sum(gt_relative_poses[:, 1]))
            z_error.append(
                np.sum(np.abs(relative_poses[:, 2] - gt_relative_poses[:, 2])) / np.sum(gt_relative_poses[:, 2]))
            total_error[i].append(np.sqrt(x_error[-1] ** 2 + y_error[-1] ** 2 + z_error[-1] ** 2))

    fig = plt.figure()
    plt.title('Relative PnP pose estimation on variable sequence length')
    for i, seq_len in enumerate(seq_lenths):
        plt.plot(range(len(total_error[i])), total_error[i], label="Total error on sequence length {}".format(seq_len))

    plt.ylabel("Error Percentage [%/m]")
    plt.xlabel("Frame [#]")
    plt.legend()
    fig.savefig('Relative PnP Total pose estimation error on variable sequence length')
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
        for i in range(len(gt_locations) - length):
            relative_poses = rel_pose_move(ba_cameras[i:i + length])
            gt_relative_poses = rel_pose_move(gt_locations[i:i + length])
            x_error.append(
                np.sum(np.abs(relative_poses[:, 0] - gt_relative_poses[:, 0])) / np.sum(gt_relative_poses[:, 0]))
            y_error.append(
                np.sum(np.abs(relative_poses[:, 1] - gt_relative_poses[:, 1])) / np.sum(gt_relative_poses[:, 1]))
            z_error.append(
                np.sum(np.abs(relative_poses[:, 2] - gt_relative_poses[:, 2])) / np.sum(gt_relative_poses[:, 2]))
            total_error.append(np.sqrt(x_error[-1] ** 2 + y_error[-1] ** 2 + z_error[-1] ** 2))

        fig = plt.figure()
        plt.title('Relative BA pose estimation on sequence length {}'.format(length))
        plt.plot(range(len(x_error)), x_error, label='X relative error %')
        plt.plot(range(len(y_error)), y_error, label='Y relative error %')
        plt.plot(range(len(z_error)), z_error, label='Z relative error %')
        plt.plot(range(len(total_error)), total_error, label="Total error norm %")
        plt.ylabel("Error Percentage")
        plt.xlabel("Frame")
        plt.legend()
        fig.savefig('Relative BA pose estimation on sequence length {}.png'.format(length))
        plt.close(fig)


""" Bundle Adjustment Analysis Plots """


def proj_error_ba():
    """
    Median (or any other meaningful statistic) projection error of the different track links as
    a function of distance from the reference frame (1st frame for Bundle)
    """
    pass


def proj_error_pnp():
    """
    Median (or any other meaningful statistic) projection error of the different track links as
    a function of distance from the reference frame (triangulation frame)
    """
    pass


""" Loop Closure Analysis Plots """


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


""" Uncertainty Analysis Plots """


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
    init_covs = [no_loop_closure.get_marginals().marginalCovariance(gtsam.symbol('c', i)) for i in
                 range(len(keyframes))]
    weight_init_covs = [utils.weight_func(cov) for cov in init_covs]
    del no_loop_closure

    with_loop_closure = load_pg(PG_LOOP_PATH)
    after_loop_covs = [with_loop_closure.get_marginals().marginalCovariance(gtsam.symbol('c', i)) for i in
                       range(len(keyframes))]
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


""" Factor Optimization Analysis Plots"""


def simple_mean_factor_error_ba():
    """
    Optimization error – Keyframes on the 𝑥 axis and mean factor error on the 𝑦 axis. The graph presents a line for
    the mean factor graph error (total error / #factors) of each bundle window (the window starting at that keyframe)
    One line for the error before optimization (initial error) and one for the error after optimization (final error).
    """
    ba = Data().get_ba()
    bundles = ba.bundle_windows
    opt_factor_errors = []
    int_factor_errors = []
    for bundle in bundles:
        opt_factor_errors.append(bundle.get_factor_error())
        int_factor_errors.append(bundle.get_factor_error(initial=True))

    fig = plt.figure(figsize=(12, 5))
    plt.title('Mean Factor Error Before and After Optimization')
    plt.xlabel('Keyframe [index]')
    plt.ylabel('Mean Factor Error [m]')
    plt.plot(range(len(opt_factor_errors)), opt_factor_errors, label='After Optimization')
    plt.plot(range(len(int_factor_errors)), int_factor_errors, label='Before Optimization')
    plt.legend()
    fig.savefig('Factor Error Before and After Optimization.png')
    plt.close(fig)


""" Track length Analysis Plots """


def choose_tracks_subset(tracks_db):
    # choose a representative subset of the tracks (e.g. 100 tracks with the longest length)
    # remeber that tracks is a dictionary where the key is the track id and the value is the Track object
    tracks = tracks_db.tracks
    tracks = sorted(tracks.values(), key=lambda track: len(track.frame_ids), reverse=True)
    tracks = tracks[:100]
    return tracks


def projection_error_track_length_pnp(tracks_subset):
    rel_t_arr = Data().get_rel_T_arr()
    projection_errors = [[] for _ in range(len(tracks_subset))]
    for i, track in enumerate(tracks_subset):
        left_proj, right_proj, initial_estimates, factors = ex5_utils.triangulate_and_project(track, None,
                                                                                              T_arr=rel_t_arr)
        left_locations, right_locations = track.get_left_kp(), track.get_right_kp()
        _, right_proj_dist, left_proj_dist = ex5_utils.calculate_reprojection_error((left_proj, right_proj),
                                                                                    (left_locations, right_locations))
        projection_errors[i] = left_proj_dist

    # calculate the median projection error for each distance from the reference
    # (e.g. for all tracks with length between 0 and 1 meter, calculate the median projection error)
    # make a nparray, cols: link length, rows: projection error
    track_lengths = [len(track.frame_ids) for track in tracks_subset]
    track_lengths = np.array(track_lengths)
    track_lengths = np.unique(track_lengths)
    track_lengths = np.sort(track_lengths)
    link_len_errors = [[] for _ in range(0, max(track_lengths) + 1)]
    # insert the projection errors to the correct place in the array
    for track_error in projection_errors:
        for i, link_len_error in enumerate(track_error):
            link_len_errors[i].append(link_len_error)
    # calculate the median for each link length
    medians = [np.median(lst) for lst in link_len_errors]

    # plot the median projection error as a function of the track length
    fig = plt.figure(figsize=(12, 5))
    plt.title('Projection Error vs Track Length in PnP')
    plt.xlabel('Distance From Reference [#frames]')
    plt.ylabel('Projection Error [m]')
    plt.plot(range(0, max(track_lengths) + 1), medians[::-1], label='Median Projection Error')
    plt.legend()
    fig.savefig(f'Projection Error vs Track Length in PnP 2.png')
    plt.close(fig)


def projection_error_track_length_ba_v2(tracks_db, tracks_subset, ba):
    rel_t_arr = Data().get_rel_T_arr()
    projection_errors = [[] for _ in range(len(tracks_subset))]
    tracks_subset = set()
    for bundle_window in ba.bundle_windows:
        initial_landmarks = bundle_window.get_from_initial('landmarks')
        initial_poses = bundle_window.get_from_initial('camera_poses')


def projection_error_track_length_ba(tracks_db, tracks_subset, ba):
    rel_t_arr = Data().get_rel_T_arr()
    projection_errors = [[] for _ in range(len(tracks_subset))]
    tracks_subset = set()
    for bundle_window in ba.bundle_windows:
        bundle_frames = bundle_window.frames_idxs
        tracks_in_frames = set()
        for frame_id in bundle_frames:
            tracks_in_frames.update(tracks_db.get_track_ids(frame_id))
        tracks_in_frames = [tracks_db.get_track(track_id) for track_id in tracks_in_frames]
        tracks_subset.update(tracks_in_frames)

    #     make the track subset a bit smaller
    tracks_subset = list(tracks_subset)
    tracks_subset = sorted(tracks_subset, key=lambda track: len(track.frame_ids), reverse=True)[:100]
    projection_error_track_length_pnp(tracks_subset)

    for i, track in enumerate(tracks_subset):
        left_proj, right_proj, initial_estimates, factors = ex5_utils.triangulate_and_project(track, None,
                                                                                              T_arr=rel_t_arr)
        left_locations, right_locations = track.get_left_kp(), track.get_right_kp()
        _, right_proj_dist, left_proj_dist = ex5_utils.calculate_reprojection_error((left_proj, right_proj),
                                                                                    (left_locations, right_locations))
        projection_errors[i] = left_proj_dist

    # calculate the median projection error for each distance from the reference
    # (e.g. for all tracks with length between 0 and 1 meter, calculate the median projection error)
    # make a nparray, cols: link length, rows: projection error
    track_lengths = [len(track.frame_ids) for track in tracks_subset]
    track_lengths = np.array(track_lengths)
    track_lengths = np.unique(track_lengths)
    track_lengths = np.sort(track_lengths)
    link_len_errors = [[] for _ in range(0, max(track_lengths) + 1)]
    # insert the projection errors to the correct place in the array
    for track_error in projection_errors:
        for i, link_len_error in enumerate(track_error):
            link_len_errors[i].append(link_len_error)
    # calculate the median for each link length
    medians = [np.median(lst) for lst in link_len_errors]

    # plot the median projection error as a function of the track length
    fig = plt.figure(figsize=(12, 5))
    plt.title('Projection Error vs Track Length in PnP')
    plt.xlabel('Distance From Reference [#frames]')
    plt.ylabel('Projection Error [m]')
    plt.plot(range(0, max(track_lengths) + 1), medians[::-1], label='Median Projection Error')
    plt.legend()
    fig.savefig(f'Projection Error vs Track Length in PnP 2.png')
    plt.close(fig)


def angle_difference(angle1, angle2):
    """
    gets 2 360 range angles and returns the difference between them, ignoring the direction
    :param angle1:
    :param angle2:
    :return:
    """
    # for angles that are -20 - they are actually 340
    if angle1 < 0:
        angle1 += 360
    if angle2 < 0:
        angle2 += 360
    if angle1 > 180:
        angle1 = 180 - (angle1 % 180)
    if angle2 > 180:
        angle2 = 180 - (angle2 % 180)
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)


def calc_angle_diffs(angles1, angles2):
    yz_error = [angle_difference(ang1, ang2) for ang1, ang2 in zip(angles1[:, 0], angles2[:, 0])]
    xz_error = [angle_difference(ang1, ang2) for ang1, ang2 in zip(angles1[:, 1], angles2[:, 1])]
    xy_error = [angle_difference(ang1, ang2) for ang1, ang2 in zip(angles1[:, 2], angles2[:, 2])]

    return yz_error, xz_error, xy_error


def plot_trajectory_with_vehichle_view():
    with_loop_closure = load_pg(PG_LOOP_PATH)
    # auxilery_plot_utils.plot_pose_graphs([with_loop_closure], ['With Loop Closure'])
    animation = auxilery_plot_utils.make_moving_car_animation(with_loop_closure)
    # print("Saving animation...")
    # animation.save('final project view 0.gif', dpi=20, writer='pillow')
    auxilery_plot_utils.add_images_to_animation(animation, with_loop_closure.keyframes)
    print("Saving animation...")
    animation.save('final project view.gif', dpi=100, writer='pillow')



@utils.measure_time
def make_plots():
    """
    Make plots needed for final project submission.
    """
    # tracks_db = Data().get_tracks_db()
    # tracks_subset = choose_tracks_subset(tracks_db)
    # ba = Data().get_ba()
    # projection_error_track_length_ba(tracks_db, tracks_subset, ba)
    # # projection_error_track_length_pnp(tracks_subset)
    # uncertainty_vs_kf()
    plot_trajectory_with_vehichle_view()

def main():
    make_plots()


if __name__ == '__main__':
    main()
