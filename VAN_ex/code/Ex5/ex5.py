import gtsam

import numpy as np
import matplotlib.pyplot as plt

import VAN_ex.code.Ex1.ex1 as ex1_utils
import VAN_ex.code.Ex3.ex3 as ex3_utils
from VAN_ex.code.PreCalcData.paths_to_data import DB_PATH, T_ARR_PATH, BA_PATH
from VAN_ex.code.utils import utils, projection_utils, auxilery_plot_utils
from VAN_ex.code.DataBase.TracksDB import TracksDB
from VAN_ex.code.BundleAdjustment import BundleWindow
from VAN_ex.code.BundleAdjustment import BundleAdjustment
from VAN_ex.code.utils.auxilery_plot_utils import plot_scene_from_above, plot_scene_3d

# from VAN_ex.code.utils import gtsam_plot_utils

old_k, m1, m2 = ex3_utils.k, ex3_utils.m1, ex3_utils.m2


def q5_1(track_db: TracksDB):
    track = utils.get_rand_track(10, track_db, seed=5)
    left_proj, right_proj, initial_estimates, factors = triangulate_and_project(track, track_db)
    left_locations, right_locations = track.get_left_kp(), track.get_right_kp()

    # Present a graph of the reprojection error size (L2 norm) over the track’s images
    total_proj_dist, right_proj_dist, left_proj_dist = calculate_reprojection_error((left_proj, right_proj),
                                                                                    (left_locations, right_locations))

    fig = plot_reprojection_error(right_proj_dist, left_proj_dist, track.get_frame_ids())

    # Present a graph of the factor error over the track’s images.
    errors = plot_factor_error(factors, initial_estimates, track.get_frame_ids(), fig)

    # Present a graph of the factor error as a function of the re-projection error.
    plot_factor_vs_reprojection_error(errors, total_proj_dist)

    # What is the factor error as a function of the reprojection error? What is the slope of the line?
    print('The slope of the line is: {}'.format(np.polyfit(total_proj_dist, errors, 1)[0]))


def q5_2(tracks_db, t_arr):
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
    keyframes = [0, 7]  # First two keyframes
    bundle_window = BundleWindow.Bundle(keyframes[0], keyframes[1])
    bundle_window.create_graph_v2(t_arr, tracks_db)
    result = bundle_window.optimize()

    print('Initial error = {}'.format(bundle_window.get_factor_error(initial=True)))
    print('Final error = {}'.format(bundle_window.get_factor_error(initial=False)))

    # Pick a projection factor between frame c and 3d point q.
    random_factor = bundle_window.graph.at(3)
    p2d = random_factor.measured()
    xl, xr, y = p2d.uL(), p2d.uR(), p2d.v()
    left_point, right_point = (xl, y), (xr, y)

    # Print its error for the initial values of c and q.
    print('Initial error of random factor = {}'.format(random_factor.error(bundle_window.initial_estimates)))

    # Initialize a StereoCamera with the initial pose for c, use it to project the initial
    # position of q.
    c, q = random_factor.keys()
    pose = bundle_window.initial_estimates.atPose3(c)
    stereo_camera = gtsam.StereoCamera(pose, utils.create_gtsam_K())
    p3d = bundle_window.initial_estimates.atPoint3(q)
    first_proj = stereo_camera.project(p3d)
    first_lproj, first_rproj = (first_proj.uL(), first_proj.v()), (first_proj.uR(), first_proj.v())

    # Present the left and right projections on both images,
    # along with the measurement.
    left_image, right_image = ex1_utils.read_images(0)
    plot_proj_on_images(first_lproj, first_rproj, left_point, right_point, left_image, right_image, 'before')

    # Repeat this process for the final (optimized) values of c and q.
    print('Final error of random factor = {}'.format(random_factor.error(result)))
    pose = bundle_window.result.atPose3(c)
    stereo_camera = gtsam.StereoCamera(pose, utils.create_gtsam_K())
    p3d = bundle_window.result.atPoint3(q)
    projection = stereo_camera.project(p3d)
    left_proj, right_proj = (projection.uL(), projection.v()), (projection.uR(), projection.v())
    plot_proj_on_images(left_proj, right_proj, left_point, right_point, left_image, right_image, 'after')

    # Plot the resulting positions of the first bundle both as a 3D graph, and as a view-from-above (2d)
    # of the scene, with all cameras and points.
    plot_scene_from_above(result, question='5_2')
    plot_scene_3d(result, question='5_2')


def q5_3(tracks_db, T_arr):
    """
    Choose all the keyframes along the trajectory and solve all resulting Bundle windows.
    Extract the relative pose between each keyframe and its predecessor (location + angles).
    Calculate the absolute pose of the keyframes in global (camera 0) coordinate system.
    """
    bundle_adjustment = BundleAdjustment.BundleAdjustment(tracks_db, T_arr)
    bundle_adjustment.choose_keyframes(type='end_frame', parameter=200)
    bundle_adjustment.solve()
    bundle_adjustment.serialize(BA_PATH)


    # convert relative poses to absolute poses
    cameras, landmarks = bundle_adjustment.get_relative_poses()

    ground_truth_keyframes = \
        np.array(ex3_utils.calculate_camera_trajectory(ex3_utils.get_ground_truth_transformations()))[
            bundle_adjustment.keyframes]

    cameras_trajectory = projection_utils.get_trajectory_from_gtsam_poses(cameras)
    initial_est = utils.get_initial_estimation(rel_t_arr=T_arr)[bundle_adjustment.keyframes]

    fig, axes = plt.subplots(figsize=(6, 6))
    fig = auxilery_plot_utils.plot_camera_trajectory(camera_pos=landmarks, fig=fig, label="projected landmarks",
                                                     color='grey', size=1, alpha=0.3)
    fig = auxilery_plot_utils.plot_ground_truth_trajectory(ground_truth_keyframes, fig)
    fig = auxilery_plot_utils.plot_camera_trajectory(camera_pos=initial_est, fig=fig, label="initial estimate",
                                                     color='green')
    fig = auxilery_plot_utils.plot_camera_trajectory(camera_pos=cameras_trajectory, fig=fig, label="BA", color='pink')
    legend_element = plt.legend(loc='upper left', fontsize=12)
    fig.gca().add_artist(legend_element)
    fig.savefig('q5_3 trajectory.png')
    fig.show()
    plt.clf()

    # For the last bundle window print the final position of the first frame of that bundle and the anchoring factor
    # final error. Why is that the error?
    last_bundle = bundle_adjustment.bundle_windows[-1]
    print('Final error = {}'.format(last_bundle.prior_factor.error(last_bundle.result)))
    print('Final position of the first frame = {}'.format(last_bundle.get_from_optimized(obj='camera_poses')[0]))

    euclidean_distance = calculate_euclidian_dist(cameras_trajectory, ground_truth_keyframes)
    plot_keyframe_localization_error(len(bundle_adjustment.keyframes), euclidean_distance)


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
    T_arr = ex3_utils.calculate_relative_transformations(T_arr)
    initial_estimates = gtsam.Values()
    K = utils.create_gtsam_K()

    track_frames = track.get_frame_ids()
    last_frame_id = track_frames[-1]
    last_frame_ext_mat = T_arr[last_frame_id]
    first_frame_id = track_frames[0]
    first_frame_ext_mat = T_arr[first_frame_id]

    # world_base_camera = fix_ext_mat(first_frame_ext_mat)  # World coordinates for transformations
    last_frame_in_world = ex3_utils.composite_transformations(first_frame_ext_mat, last_frame_ext_mat)

    point_symbol = gtsam.symbol('q', 0)
    base_pose = gtsam.Pose3(fix_ext_mat(last_frame_in_world))
    base_stereo_frame = gtsam.StereoCamera(base_pose, K)
    xl, xr, y = tracks_db.feature_location(last_frame_id, track.get_track_id())
    point = gtsam.StereoPoint2(xl, xr, y)
    p3d = base_stereo_frame.backproject(point)
    initial_estimates.insert(point_symbol, p3d)

    # Create a factor for each frame projection and present a graph of the factor error over the track’s frames.
    factors, left_proj, right_proj = [], [], []

    for frame_id in track_frames:
        ext_mat = T_arr[frame_id]
        cur_ext_mat = ex3_utils.composite_transformations(first_frame_ext_mat, ext_mat)

        cam_symbol = gtsam.symbol('c', frame_id)
        pose = gtsam.Pose3(fix_ext_mat(cur_ext_mat))
        initial_estimates.insert(cam_symbol, pose)
        stereo_frame = gtsam.StereoCamera(pose, K)
        projection = stereo_frame.project(p3d)  # Project point for each frame in track
        left_proj.append((projection.uL(), projection.v()))
        right_proj.append((projection.uR(), projection.v()))

        xl, xr, y = tracks_db.feature_location(frame_id, track.get_track_id())
        point = gtsam.StereoPoint2(xl, xr, y)

        # Create a factor for each frame projection and present a graph of the factor error over the track’s frames.
        factor = gtsam.GenericStereoFactor3D(point, gtsam.noiseModel.Isotropic.Sigma(3, 1.0), cam_symbol, point_symbol,
                                             K)
        factors.append(factor)

    return left_proj, right_proj, initial_estimates, factors


def fix_ext_mat(ext_mat):
    """
    Fix the extrinsic matrix to be in the correct format for gtsam.
    """
    R = ext_mat[:, :3]
    t = ext_mat[:, 3]
    new_t = -R.T @ t
    return np.hstack((R.T, new_t.reshape(3, 1)))


def calculate_reprojection_error(projections, locations):
    """
    Calculate the reprojection error size (L2 norm) over the track’s images.
    """
    left_projections, right_projections = projections
    left_locations, right_locations = locations
    left_locations, right_locations = np.array(list(left_locations.values())), np.array(list(right_locations.values()))
    left_proj_dist = np.linalg.norm(left_projections - left_locations, axis=1, ord=2)
    right_proj_dist = np.linalg.norm(right_projections - right_locations, axis=1, ord=2)
    total_proj_dist = (left_proj_dist + right_proj_dist) / 2
    return total_proj_dist, left_proj_dist, right_proj_dist


def plot_reprojection_error(right_proj_dist, left_proj_dist, frame_ids):
    """
    Present a graph of the reprojection error size (L2 norm) over the track’s images.
    """
    fig, ax = plt.subplots()
    ax.plot(frame_ids, right_proj_dist, label='Right')
    ax.plot(frame_ids, left_proj_dist, label='Left')

    ax.set_title("Reprojection error over track's images")
    ax.set_ylabel('Error')
    ax.set_xlabel('Frames')
    ax.legend()
    fig.savefig('reprojection_error.png')
    return fig


def plot_factor_error(factors, values, frame_ids, fig=None):
    """
    Present a graph of the factor error over the track’s frames.
    """
    errors = [factor.error(values) for factor in factors]
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()
    ax.legend_.remove()
    ax.set_title("Factor error over track's frames")
    ax.set_ylabel('Error')
    ax.set_xlabel('Frames')
    ax.plot(frame_ids, errors, label='Factor error')
    ax.legend(loc='upper left')
    fig.savefig('factor_error.png')

    return errors


def plot_factor_vs_reprojection_error(errors, total_proj_dist):
    """
    Present a graph of the factor error as a function of the reprojection error.
    """
    fig, ax = plt.subplots()
    ax.plot(total_proj_dist, errors)
    ax.set_title("Factor error as a function of the reprojection error")
    ax.set_ylabel('Factor error')
    ax.set_xlabel('Reprojection error')
    plt.savefig('factor_vs_reprojection_error.png')


def plot_keyframe_localization_error(keyframes_len, errors):
    """
    Present the keyframe localization error in meters (location difference only - Euclidean distance) over time.
    """
    fig, ax = plt.subplots()
    ax.plot(range(keyframes_len), errors)
    ax.set_title("Keyframe localization error in meters over time")
    ax.set_ylabel('Error')
    ax.set_xlabel('Time')
    plt.savefig('keyframe_localization_error.png')


def calculate_euclidian_dist(abs_cameras, ground_truth_cameras):
    pts_sub = abs_cameras - ground_truth_cameras
    sum_of_squared_diffs = np.linalg.norm(pts_sub, axis=1)
    return np.sqrt(sum_of_squared_diffs)


def plot_proj_on_images(left_proj, right_proj, left_point, right_point, left_image, right_image, type):
    """
    Plot the projection of the 3D points on the images.
    """
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(left_image, cmap='gray')
    ax[0].scatter(left_proj[0], left_proj[1], s=1, c='cyan', label='Projection')
    ax[0].scatter(left_point[0], left_point[1], s=1, c='green', label='Point')
    ax[0].set_title("Left image")
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    ax[1].imshow(right_image, cmap='gray')
    ax[1].scatter(right_proj[0], right_proj[1], s=1, c='cyan', label='Projection')
    ax[1].scatter(right_point[0], right_point[1], s=1, c='green', label='Point')
    ax[1].set_title("Right image")
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')

    plt.legend(fontsize="7")
    plt.savefig('proj_on_images_{}.png'.format(type))


# ===== End of Helper Functions =====


def run_ex5():
    """
    Runs all exercise 5 sections.
    """
    np.random.seed(5)
    # Load tracks DB
    tracks_db = TracksDB.deserialize(DB_PATH)
    T_arr = np.load(T_ARR_PATH)
    rel_t_arr = ex3_utils.calculate_relative_transformations(T_arr)

    # q5_1(tracks_db)
    # q5_2(tracks_db, rel_t_arr)
    q5_3(tracks_db, rel_t_arr)


def main():
    run_ex5()


if __name__ == '__main__':
    main()
