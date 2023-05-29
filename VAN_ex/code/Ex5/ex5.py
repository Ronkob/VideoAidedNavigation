import os
import gtsam
import numpy as np
import matplotlib.pyplot as plt

import VAN_ex.code.Ex3.ex3 as ex3_utils
from VAN_ex.code.Ex4.ex4 import TracksDB
from VAN_ex.code.Ex4.ex4 import Track  # Don't remove

DB_PATH = os.path.join('..', 'Ex4', 'tracks_db.pkl')
T_ARR_PATH = os.path.join('..', 'Ex3', 'T_arr.npy')
old_k, m1, m2 = ex3_utils.k, ex3_utils.m1, ex3_utils.m2


def q5_1(track_db):
    # track = ex4_utils.get_rand_track(10, track_db)
    track = track_db.tracks[12]  # For debugging
    locations = track.kp
    left_proj, right_proj, initial_estimates, factors = triangulate_and_project(track, track_db)
    left_locations, right_locations = [], []
    for frame_id in track.get_frame_ids():
        left_locations.append(locations[frame_id][0])
        right_locations.append(locations[frame_id][1])

    # Present a graph of the reprojection error size (L2 norm) over the track’s images.
    # total_proj_dist = \
    #     plot_reprojection_error((left_proj, right_proj), (left_locations, right_locations), track.get_frame_ids())

    # Present a graph of the factor error over the track’s images.
    # errors = plot_factor_error(factors, initial_estimates, track.get_frame_ids())

    # Present a graph of the factor error as a function of the reprojection error.
    # plot_factor_vs_reprojection_error(errors, total_proj_dist)


def q5_2(tracks_db):
    """
    We perform local Bundle Adjustment on a small window consisting of consecutive frames.
     Each bundle ‘window’ starts and ends in special frames we call keyframes.
     Keyframes should be chosen such that the number of frames in the window is small (5-20
     frames) with some meaningful movement between them.
    We use the tracking from the previous exercises as constraints for the optimization - the
    tracking is used to construct factors (reprojection measurement constraints) between the
    frames and the tracked landmarks. As an initialization for the optimization, we use the relative
    locations calculated by the PnP and triangulated 3d landmarks locations.
    The first Bundle window consists of the first two keyframes with all the frames between them,
    with all the relevant tracking data.
    """
    # keyframes = keyframes_criterion(tracks_db)
    keyframes = [0, 5]  # For debugging - first two keyframes
    bundle_windows = get_bundle_windows(keyframes)
    first_bundle = bundle_windows[0]
    graph, initial_estimates, points, cameras = factor_graph_for_bundle(first_bundle, tracks_db)

    # Print the total factor graph error before and after the optimization.
    print('Initial error = {}'.format(graph.error(initial_estimates)))
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
    result = optimizer.optimize()
    print('Final error = {}'.format(graph.error(result)))

    # Plot the resulting positions of the first bundle both as a 3D graph, and as a view-from-above (2d)
    # of the scene, with all cameras and points.
    # gtsam.utils.set_axes_equal(0)
    # gtsam.utils.plot_trajectory(result)  # / gtsam.utils.plot_3d_points(result)
    # plot_view_from_above(result, first_bundle, cameras, points)


def q5_3(tracks_db):
    """
    Choose all the keyframes along the trajectory and solve all resulting Bundle windows.
    Extract the relative pose between each keyframe and its predecessor (location + angles).
    Calculate the absolute pose of the keyframes in global (camera 0) coordinate system.
    """
    # keyframes = keyframes_criterion(tracks_db)  # Choose all the keyframes along the trajectory
    keyframes = [0, 5, 10, 15, 20, 25]  # For debugging
    bundle_windows = get_bundle_windows(keyframes)
    cameras_rel_pose, points_rel_pose = [], []

    # Solve all resulting Bundle windows
    for window in bundle_windows:
        graph, initial_estimates, points, cameras = factor_graph_for_bundle(window, tracks_db)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
        result = optimizer.optimize()
        cameras_rel_pose.append(result.atPose3(gtsam.symbol('c', window[-1])))  # Between each keyframe and its predecessor
        points_rel_pose.append([result.atPoint3(point) for point in points])

    # Present a view from above (2d) of the scene, with all keyframes (left camera only) and 3D points.
    # Overlay the estimated keyframes with the Ground Truth poses of the keyframes.
    abs_cameras, abs_points = relative_to_absolute_poses(cameras_rel_pose, points_rel_pose)
    ground_truth_keyframes = np.array(ex3_utils.get_ground_truth_transformations())[keyframes][:, :, [-1]].squeeze()
    abs_cameras = np.array([camera.translation() for camera in abs_cameras])
    # plot_view_from_above2(abs_cameras, abs_points, ground_truth_keyframes)

    # Present the keyframe localization error in meters (location difference only - Euclidean
    # distance) over time.
    euclidean_distance = calculate_euclidian_dist(abs_cameras, ground_truth_keyframes)
    plot_keyframe_localization_error(len(keyframes), euclidean_distance)


# ===== Helper functions =====


def triangulate_and_project(track, tracks_db):
    """
    For all the frames participating in this track, define a gtsam.StereoCamera
     using the global camera matrices calculated in exercise 3 (PnP).
     Using methods in StereoCamera, triangulate a 3d point in global coordinates
     from the last frame of the track, and project this point to all the frames
     of the track (both left and right cameras).
    Moreover, Create a factor for each frame projection and present a graph of the
     factor error over the track’s frames.
    """
    # Load and set initial values
    T_arr = np.load(T_ARR_PATH)
    initial_estimates = gtsam.Values()
    K = compute_K()

    track_frames = track.get_frame_ids()
    last_frame_id = track.frame_ids[-1]
    last_frame_ext_mat = T_arr[last_frame_id]
    first_frame_ext_mat = T_arr[0]

    world_base_camera = fix_ext_mat(first_frame_ext_mat)  # World coordinates for transformations
    last_frame_in_world = ex3_utils.composite_transformations(world_base_camera, last_frame_ext_mat)
    last_camera = fix_ext_mat(last_frame_in_world)

    point_symbol = gtsam.symbol('q', 0)
    base_pose = gtsam.Pose3(last_camera)
    base_stereo_frame = gtsam.StereoCamera(base_pose, K)
    xl, xr, y = tracks_db.feature_location(last_frame_id, track.get_track_id())
    point = gtsam.StereoPoint2(xl, xr, y)
    p3d = base_stereo_frame.backproject(point)
    initial_estimates.insert(point_symbol, p3d)

    # Create a factor for each frame projection and present a graph of the factor error over the track’s frames.
    factors, left_proj, right_proj = [], [], []

    for frame_id in track_frames:
        ext_mat = T_arr[frame_id]
        cur_ext_mat = ex3_utils.composite_transformations(world_base_camera, ext_mat)
        cur_cam_in_world = fix_ext_mat(cur_ext_mat)

        cam_symbol = gtsam.symbol('c', track_frames[frame_id])
        pose = gtsam.Pose3(cur_cam_in_world)
        initial_estimates.insert(cam_symbol, pose)
        stereo_frame = gtsam.StereoCamera(pose, K)
        projection = stereo_frame.project(p3d)
        left_proj.append((projection.uL(), projection.v()))
        right_proj.append((projection.uR(), projection.v()))

        xl, xr, y = tracks_db.feature_location(frame_id, track.get_track_id())
        point = gtsam.StereoPoint2(xl, xr, y)

        # Create a factor for each frame projection and present a graph of the factor error over the track’s frames.
        factor = gtsam.GenericStereoFactor3D(point, gtsam.noiseModel.Isotropic.Sigma(3, 1.0), cam_symbol, point_symbol, K)
        factors.append(factor)

    return left_proj, right_proj, initial_estimates, factors


def compute_K():
    """
    Compute the camera matrix K from the old camera matrix and the new baseline.
    """
    fx, fy, skew = old_k[0, 0], old_k[1, 1], old_k[0, 1]
    cx, cy = old_k[0, 2], old_k[1, 2]
    baseline = m2[0, 3]  # Just like t[0]
    K = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -baseline)
    return K


def fix_ext_mat(ext_mat):
    ext_mat[:, -1] *= -1  # t
    return ext_mat  # gtsam.Rot3(R), gtsam.Point3(t)


def plot_reprojection_error(projections, locations, frame_ids):

    """
    Present a graph of the reprojection error size (L2 norm) over the track’s images.
    """
    left_projections, right_projections = projections
    left_locations, right_locations = np.array(locations)
    left_proj_dist = np.linalg.norm(left_projections - left_locations, axis=1, ord=2)
    right_proj_dist = np.linalg.norm(right_projections - right_locations, axis=1, ord=2)
    total_proj_dist = (left_proj_dist + right_proj_dist) / 2

    plt.plot(frame_ids, total_proj_dist)
    plt.title("Reprojection error over track's images")
    plt.ylabel('Error')
    plt.xlabel('Frames')
    plt.show()

    return total_proj_dist


def plot_factor_error(factors, values, frame_ids):
    """
    Present a graph of the factor error over the track’s frames.
    """
    errors = [factor.error(values) for factor in factors]

    plt.plot(frame_ids, errors)
    plt.title("Factor error over track's frames")
    plt.ylabel('Error')
    plt.xlabel('Frames')
    plt.show()

    return errors


def plot_factor_vs_reprojection_error(errors, total_proj_dist):
    """
    Present a graph of the factor error as a function of the reprojection error.
    """
    plt.plot(total_proj_dist, errors)
    plt.title("Factor error as a function of the reprojection error")
    plt.ylabel('Factor error')
    plt.xlabel('Reprojection error')
    plt.show()


def keyframes_criterion(track_db):
    """
    Decide on a criterion to mark specific frames as keyframes - the decision can use
    the estimated distance travelled, lapsed time, features tracked or any other
    relevant (and available) criterion.
    """
    pass


def get_bundle_windows(keyframes):
    """
    Create a list of bundle windows, where each window is a list of frame ids.
    """
    bundle_windows = []
    for i in range(len(keyframes) - 1):
        bundle_windows.append(list(range(keyframes[i], keyframes[i + 1] + 1)))

    return bundle_windows


def factor_graph_for_bundle(bundle_window, tracks_db):
    """
    Create a factor graph for a given bundle window.
    """
    graph = gtsam.NonlinearFactorGraph()
    K = compute_K()
    T_arr = np.load(T_ARR_PATH)
    initial_estimates = gtsam.Values()
    points, cameras = [], []

    first_frame_ext_mat = T_arr[bundle_window[0]]
    world_base_camera = fix_ext_mat(first_frame_ext_mat)  # World coordinates for transformations

    # Create a pose for each camera in the bundle window
    for frame_id in bundle_window:
        ext_mat = T_arr[frame_id]
        cur_ext_mat = ex3_utils.composite_transformations(world_base_camera, ext_mat)
        cur_cam_in_world = fix_ext_mat(cur_ext_mat)

        pose = gtsam.Pose3(cur_cam_in_world)
        cam_symbol = gtsam.symbol('c', frame_id)
        cameras.append(cam_symbol)
        initial_estimates.insert(cam_symbol, pose)

        # Add a prior factor for first camera pose
        if frame_id == bundle_window[0]:  # Constraints for first frame
            factor = gtsam.PriorFactorPose3(cam_symbol, pose,
                                            gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])))
            graph.add(factor)

    for track_id in tracks_db.get_track_ids(bundle_window[0]):
        track = tracks_db.tracks[track_id]
        last_frame_id = track.frame_ids[-1]
        track_frames = track.get_frame_ids()

        # Create a point for each track in the first keypoint frame
        point_symbol = gtsam.symbol('q', track.get_track_id())
        points.append(point_symbol)
        base_stereo_frame = gtsam.StereoCamera(pose, K)  # Pose of last frame in bundle window
        xl, xr, y = tracks_db.feature_location(last_frame_id, track_id)
        point = gtsam.StereoPoint2(xl, xr, y)
        p3d = base_stereo_frame.backproject(point)
        initial_estimates.insert(point_symbol, p3d)

        # Create a factor for each frame of track
        for frame_id in bundle_window:
            if frame_id not in track_frames:
                continue
            cam_symbol = gtsam.symbol('c', frame_id)
            xl, xr, y = tracks_db.feature_location(frame_id, track_id)
            point = gtsam.StereoPoint2(xl, xr, y)

            factor = gtsam.GenericStereoFactor3D(point, gtsam.noiseModel.Isotropic.Sigma(3, 1.0), cam_symbol,
                                                 point_symbol, K)
            graph.add(factor)

    return graph, initial_estimates, points, cameras


def plot_view_from_above(result, bundle_window, cameras, points):
    """
    Plot the resulting positions as a view-from-above (2d) of the scene, with all cameras and points.
    """
    cameras = np.array([result.atPose3(camera).translation() for camera in cameras])
    points = np.array([result.atPoint3(point) for point in points])
    fig, ax = plt.subplots()

    ax.scatter(cameras[:, 0], cameras[:, 2], s=1, c='red', label="Cameras")
    ax.scatter(points[:, 0], points[:, 2], s=1, c='cyan', label="Points")

    ax.set_title("Points and cameras as a view from above of the scene")
    # ax.set_xlim(-250, 350)
    # ax.set_ylim(-100, 430)
    plt.legend()
    plt.show()


def plot_view_from_above2(relative_cameras, relative_points, ground_truth_keyframes):
    """
    Plot a view from above (2d) of the scene, with all keyframes (left camera only) and 3D points.
    Overlay the estimated keyframes with the Ground Truth poses of the keyframes.
    """
    fig, ax = plt.subplots()

    ax.scatter(relative_cameras[:, 0], relative_cameras[:, 2], s=1, c='red', label="Keyframes")
    ax.scatter(relative_points[:, 0], relative_points[:, 2], s=1, c='cyan', label="Points")
    ax.scatter(ground_truth_keyframes[:, 0], ground_truth_keyframes[:, 2], s=1, c='green', label="Keyframes ground truth")

    ax.set_title("Left cameras, 3D points and GT Poses of keyframes as a view from above of the scene")
    # ax.set_xlim(-250, 350)
    # ax.set_ylim(-100, 430)
    plt.legend()
    plt.show()


def relative_to_absolute_poses(cameras, points):
    """
    Convert relative poses to absolute poses.
    """
    abs_points, abs_cameras = [], [cameras[0]]

    for bundle_camera, bundle_points in zip(cameras, points):
        abs_cameras.append(abs_cameras[0].compose(bundle_camera))
        bundle_abs_points = [abs_cameras[-1].transformFrom(point) for point in bundle_points]
        abs_points.extend(bundle_abs_points)

    return np.array(abs_cameras), np.array(abs_points)


def plot_keyframe_localization_error(keyframes_len, errors):
    """
    Present the keyframe localization error in meters (location difference only - Euclidean distance) over time.
    """
    plt.plot(range(keyframes_len), errors)
    plt.title("Keyframe localization error in meters over time")
    plt.ylabel('Error')
    plt.xlabel('Time')
    plt.show()


def calculate_euclidian_dist(abs_cameras, ground_truth_cameras):
    pts_sub = abs_cameras - ground_truth_cameras
    sum_of_squared_diffs = np.linalg.norm(pts_sub, axis=1)
    return np.sqrt(sum_of_squared_diffs)


# ===== End of Helper Functions =====


def run_ex5():
    """
    Runs all exercise 5 sections.
    """
    # Load tracks DB
    tracks_db = TracksDB.deserialize(DB_PATH)

    q5_1(tracks_db)

    q5_2(tracks_db)

    q5_3(tracks_db)


def main():
    run_ex5()


if __name__ == '__main__':
    main()
