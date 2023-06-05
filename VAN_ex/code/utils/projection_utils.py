from typing import List

import gtsam
import numpy as np


def convert_ext_mat_to_world(ext_mat):
    R_mat = ext_mat[:, :3]
    t_vec = ext_mat[:, 3]

    R = R_mat.T
    t = - R_mat.T @ t_vec

    ex_cam_mat_from_cam_to_world = np.hstack((R, t.reshape(3, 1)))  # Todo: check concatenation

    return ex_cam_mat_from_cam_to_world


def composite_transformations(T1, T2):
    """
    Calculate the composite transformation between two cameras.
    """
    hom1 = np.append(T1, [np.array([0, 0, 0, 1])], axis=0)
    return T2 @ hom1


def convert_rel_gtsam_trans_to_global(T_arr):
    relative_T_arr = []
    last = T_arr[0]

    for t in T_arr:
        last = last.compose(t)
        relative_T_arr.append(last)

    return relative_T_arr


def convert_bundle_rel_landmark_to_global(first_cam: gtsam.Pose3, bundle_landmarks: List[gtsam.Point3]):
    global_landmarks = []
    for landmark in bundle_landmarks:
        global_landmark = first_cam.transformFrom(landmark)
        global_landmarks.append(global_landmark)

    return global_landmarks


def convert_rel_landmarks_to_global(rel_cameras, rel_landmarks):
    global_landmarks = []
    for bundle_camera, bundle_landmarks in zip(rel_cameras, rel_landmarks):
        bundle_global_landmarks = convert_bundle_rel_landmark_to_global(bundle_camera, bundle_landmarks)
        global_landmarks += bundle_global_landmarks

    return np.array(global_landmarks)


def computed_trajectory_from_poses(poses: List[gtsam.Pose3]):
    trajectory = []
    for pose in poses:
        trajectory.append(pose.translation())
    return np.array(trajectory)


def calc_relative_camera_pos(ext_camera_mat):
    """
    Returns the relative position of a camera according to its extrinsic
     matrix.
    """
    return -1 * ext_camera_mat[:, :3].T @ ext_camera_mat[:, 3]


def calculate_camera_trajectory(relative_T_arr):
    """
    Calculate the camera trajectory according to the relative position of
     each camera.
    """
    trajectory = []
    for T in relative_T_arr:
        trajectory.append(calc_relative_camera_pos(T))
    return np.array(trajectory)
