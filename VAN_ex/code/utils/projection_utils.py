from typing import List

import gtsam
import numpy as np


def convert_ext_mat_to_world(ext_mat):
    R_mat = ext_mat[:, :3]
    t_vec = ext_mat[:, 3]

    R = R_mat.T
    t = - R_mat.T @ t_vec

    return np.hstack((R, t.reshape(3, 1)))


def composite_transformations(T1, T2):
    """
    Calculate the composite transformation between two cameras.
    """
    hom1 = np.append(T1, [np.array([0, 0, 0, 1])], axis=0)
    return T2 @ hom1


def convert_rel_gtsam_trans_to_global(T_arr):
    global_trans = []
    last = T_arr[0]

    for t in T_arr:
        last = last.compose(t)
        global_trans.append(last)

    return global_trans


def convert_from_bundel_to_world(first_cam: gtsam.Pose3, bundle_landmarks: List[gtsam.Point3]):
    """
    Convert the points to world coordinates, from already "world" coordinates, but they are according to the first
    camera of the bundle
    """
    global_landmarks = []
    for landmark in bundle_landmarks:
        global_landmark = first_cam.transformFrom(landmark)
        global_landmarks.append(global_landmark)

    return global_landmarks


def convert_rel_landmarks_to_global(rel_cameras, rel_landmarks):
    """
    convert the points to world coordinates, from already "world" coordinates, but they are according to the first
    camera of every bundle
    """
    global_landmarks = []
    for bundle_camera, bundle_landmarks in zip(rel_cameras, rel_landmarks):
        bundle_global_landmarks = convert_from_bundel_to_world(bundle_camera, bundle_landmarks)
        global_landmarks += bundle_global_landmarks

    return np.array(global_landmarks)


def get_trajectory_from_gtsam_poses(poses: List[gtsam.Pose3]):
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


