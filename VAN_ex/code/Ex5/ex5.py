import os
import gtsam
from gtsam.utils import plot as gtsam_plot

import numpy as np
import matplotlib.pyplot as plt

import VAN_ex.code.Ex1.ex1 as ex1_utils
import VAN_ex.code.Ex3.ex3 as ex3_utils
import VAN_ex.code.Ex4.ex4 as ex4_utils
from VAN_ex.code.utils import utils, projection_utils, auxilery_plot_utils
from VAN_ex.code.Ex4.ex4 import TracksDB, Track
from VAN_ex.code.BundleAdjustment import BundleWindow
from VAN_ex.code.BundleAdjustment import BundleAdjustment
from VAN_ex.code.utils import gtsam_plot_utils

DB_PATH = os.path.join('..', 'Ex4', 'tracks_db.pkl')
T_ARR_PATH = os.path.join('..', 'Ex3', 'T_arr.npy')
old_k, m1, m2 = ex3_utils.k, ex3_utils.m1, ex3_utils.m2


def q5_1(track_db: TracksDB, T_arr):
    track = utils.get_rand_track(10, track_db, seed=0)
    # track = track_db.tracks[12]
    # ex4_utils.plot_random_track(track)
    # factors, initial_estimates, left_proj, right_proj = triangulate_and_project_frame(track, T_arr)
    # plot_track_projection_from_above(left_projections=left_proj)
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
    bundle_window.alternate_ver_create_graph(t_arr, tracks_db)
    result = bundle_window.optimize()

    print('Initial error = {}'.format(bundle_window.get_factor_error(initial=True)))
    print('Final error = {}'.format(bundle_window.get_factor_error(initial=False)))

    # Pick a projection factor between frame c and 3d point q.
    random_factor = bundle_window.graph.at(3)

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
    plot_proj_on_images(first_lproj, first_rproj, left_image, right_image)

    # Repeat this process for the final (optimized) values of c and q.
    print('Final error of random factor = {}'.format(random_factor.error(result)))
    pose = bundle_window.result.atPose3(c)
    stereo_camera = gtsam.StereoCamera(pose, utils.create_gtsam_K())
    p3d = bundle_window.result.atPoint3(q)
    projection = stereo_camera.project(p3d)
    left_proj, right_proj = (projection.uL(), projection.v()), (projection.uR(), projection.v())
    plot_proj_on_images(left_proj, right_proj, left_image, right_image, before=(first_lproj, first_rproj), type='after')

    # Plot the resulting positions of the first bundle both as a 3D graph, and as a view-from-above (2d)
    # of the scene, with all cameras and points.
    plot_scene_from_above(result)
    plot_scene_3d(result)


def q5_3(tracks_db, T_arr):
    """
    Choose all the keyframes along the trajectory and solve all resulting Bundle windows.
    Extract the relative pose between each keyframe and its predecessor (location + angles).
    Calculate the absolute pose of the keyframes in global (camera 0) coordinate system.
    """
    bundle_adjustment = BundleAdjustment.BundleAdjustment(tracks_db, T_arr)
    bundle_adjustment.choose_keyframes(type='end_frame', parameter=200)
    bundle_adjustment.solve_iterative()

    # convert relative poses to absolute poses
    cameras, landmarks = bundle_adjustment.get_relative_poses()

    ground_truth_keyframes = \
        np.array(ex3_utils.calculate_camera_trajectory(ex3_utils.get_ground_truth_transformations()))[
            bundle_adjustment.keyframes]

    cameras_trajectory = projection_utils.computed_trajectory_from_poses(cameras)
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
    fig.show()

    euclidean_distance = calculate_euclidian_dist(cameras_trajectory, ground_truth_keyframes)
    plot_keyframe_localization_error(len(bundle_adjustment.keyframes), euclidean_distance)


# ===== Helper functions =====
# a function that plots a scene of a certain bundle window from above
def plot_scene_from_above(result, points=None):
    plot_scene_3d(result, points=points, init_view={'azim': 0, 'elev': -90, 'vertical_axis': 'y'},
                  title="scene from above")


def plot_scene_3d(result, init_view=None, points=None, title="3d scene"):
    if init_view is None:
        init_view = {'azim': -15, 'elev': 200, 'vertical_axis': 'y'}
    fig = plt.figure(num=0, figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    if points is None:
        points = gtsam.utilities.extractPoint3(result)

    for point in points:
        gtsam_plot_utils.plot_point3_on_axes(ax, point, 'k.')

    gtsam_plot_utils.plot_trajectory(0, result, scale=5)

    # gtsam_plot_utils.set_axes_equal(0)
    ax.set_zlim3d(0, 50)
    ax.set_xlim3d(-20, 30)
    ax.set_ylim3d(-45, 5)
    ax.view_init(**init_view)
    # set the title of the plot
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.savefig("q5_2 " + title + '.png')  # fig.show()


# a function that plots the projection of the points in the worlds coordinate system
def plot_track_projection_from_above(left_projections):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Projection of the track in the world coordinate system')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(np.array(left_projections)[:, 0], np.array(left_projections)[:, 1])
    plt.show()


def triangulate_and_project_frame(track, t_arr, frame_to_triangulate=-1):
    """
    @brief triangulate a 3d point in global coordinates from the last frame of the track, and project this point to
    all the frames of the track (both left and right cameras).

    1. Triangulate a point from the frame to global
    2. Project this point to all frames in the track
    3. Compute projection error

    :param track: Track object
    :param t_arr: np.array of ext_matrices, [R, t]
    :param frame_to_triangulate: the frame, defaults to last frame
    :return: None
    """

    factors = []
    values = gtsam.Values()

    # get necessary data
    left_locations = track.get_left_kp()
    right_locations = track.get_right_kp()

    frame_idx_to_triangulate = track.get_frame_ids()[frame_to_triangulate]
    last_frame_ext_mat = t_arr[frame_idx_to_triangulate]

    last_left_locations = left_locations[frame_idx_to_triangulate]
    last_right_locations = right_locations[frame_idx_to_triangulate]

    # triangulation data
    gtsam_calib_mat = utils.create_gtsam_K()

    first_frame_cam_to_world_mat = projection_utils.convert_ext_mat_to_world(last_frame_ext_mat)
    rel_cam_transformation_first_frame = projection_utils.composite_transformations(first_frame_cam_to_world_mat,
                                                                                    t_arr[frame_idx_to_triangulate])

    gtsam_camera_pose = projection_utils.convert_ext_mat_to_world(rel_cam_transformation_first_frame)
    gtsam_left_cam_pose = gtsam.Pose3(gtsam_camera_pose)

    # create gtsam StereoCamera object
    gtsam_frame_to_triangulate = gtsam.StereoCamera(gtsam_left_cam_pose, gtsam_calib_mat)

    # first try
    xl, xr, y = last_left_locations[0], last_right_locations[0], last_left_locations[1]
    gtsam_q_for_triangulation = gtsam.StereoPoint2(xl, xr, y)
    gtsam_p3d = gtsam_frame_to_triangulate.backproject(gtsam_q_for_triangulation)

    # define symbols for gtsam calculations
    p3d_sym = gtsam.symbol("q", 0)
    values.insert(p3d_sym, gtsam_p3d)

    left_projections = []
    right_projections = []

    # for each frame in track, do the procedure above - update values and create factors
    for frame_idx in track.get_frame_ids():
        # update values
        gtsam_left_cam_pose_sym = gtsam.symbol("c", frame_idx)
        rel_cam_transformation_first_frame = projection_utils.composite_transformations(first_frame_cam_to_world_mat,
                                                                                        t_arr[frame_idx])
        gtsam_camera_pose = projection_utils.convert_ext_mat_to_world(rel_cam_transformation_first_frame)
        gtsam_left_cam_pose = gtsam.Pose3(gtsam_camera_pose)
        values.insert(gtsam_left_cam_pose_sym, gtsam_left_cam_pose)

        # measure projection error
        measure_xl, measure_xr, measure_y = left_locations[frame_idx][0], right_locations[frame_idx][0], \
            left_locations[frame_idx][1]
        gtsam_measurement_pt2 = gtsam.StereoPoint2(measure_xl, measure_xr, measure_y)
        gtsam_frame_to_triangulate = gtsam.StereoCamera(gtsam_left_cam_pose, gtsam_calib_mat)

        # project the homogenous point on the frame
        gtsam_projected_stereo_point2 = gtsam_frame_to_triangulate.project(gtsam_p3d)
        xl, xr, y = gtsam_projected_stereo_point2.uL(), gtsam_projected_stereo_point2.uR(), \
            gtsam_projected_stereo_point2.v()
        left_projections.append([xl, y])
        right_projections.append([xr, y])

        # Factor creation
        projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        factor = gtsam.GenericStereoFactor3D(gtsam_measurement_pt2, projection_uncertainty,
                                             gtsam.symbol("c", frame_idx), p3d_sym, gtsam_calib_mat)
        factors.append(factor)

    return factors, values, left_projections, right_projections


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


def plot_proj_on_images(left_proj, right_proj, left_image, right_image, before=None, type='before'):
    """
    Plot the projection of the 3D points on the images.
    """
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(left_image, cmap='gray')
    ax[0].scatter(left_proj[0], left_proj[1], s=1, c='cyan', label='Point')
    ax[0].set_title("Left image")
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    ax[1].imshow(right_image, cmap='gray')
    ax[1].scatter(right_proj[0], right_proj[1], s=1, c='cyan', label='Point')
    ax[1].set_title("Right image")
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')

    if before:
        ax[0].scatter(before[0][0], before[0][1], s=1, c='red', label='Before')
        ax[1].scatter(before[1][0], before[1][1], s=1, c='red', label='Before')

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

    # q5_1(tracks_db, rel_t_arr)
    #
    q5_2(tracks_db, rel_t_arr)  # # q5_3(tracks_db, rel_t_arr)


def main():
    run_ex5()


if __name__ == '__main__':
    main()
