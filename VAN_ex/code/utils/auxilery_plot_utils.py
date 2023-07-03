# a function that gets a Track object, and plots its points tracking
import numpy as np
import matplotlib.path as mpath
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from VAN_ex.code.DataBase.Track import Track
from VAN_ex.code.Ex3 import ex3
from VAN_ex.code.PoseGraph.PoseGraph import load_pg
from VAN_ex.code.utils import gtsam_plot_utils, projection_utils, utils


def plot_tracking_quality(track: Track):
    pass


def plot_ground_truth_trajectory(ground_truth_pos, fig=None, color='blue'):
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()
    # make the title in a beautiful, large font
    ax.set_title('Trajectory of the left camera', fontsize=20, fontweight='bold')

    # draw_car_on_trajectory(ground_truth_pos[::10], color=color, ax=ax)

    # plot the line connecting the ground truth trajectory points
    ax.plot(ground_truth_pos[:, 0], ground_truth_pos[:, 2], color=color, alpha=0.5, linewidth=5, label='ground truth')

    # get the min and max values of the x and z coordinates of the ground truth trajectory
    min_x = min(ground_truth_pos[:, 0])
    max_x = max(ground_truth_pos[:, 0])
    min_z = min(ground_truth_pos[:, 2])
    max_z = max(ground_truth_pos[:, 2])
    ax.set_xlim(min_x - 50, max_x + 50)
    ax.set_ylim(min_z - 50, max_z + 50)
    # set the labels
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    fig.savefig("groundTruthTrajectory.png")
    return fig


def get_alpha_from_poses(pose1, pose2):
    """
    Get the alpha angle between two poses.
    :param pose1: the first pose
    :param pose2: the second pose
    :return: the alpha angle between the two poses
    """
    # calculate the angle in degrees between the two poses in the x-z plane
    alpha = np.arctan2(pose2[2] - pose1[2], pose2[0] - pose1[0])
    alpha = np.rad2deg(alpha)
    return alpha


def draw_car_on_trajectory(camera_pos, ax, color):
    # draw the car on the trajectory
    for i in range(camera_pos.shape[0])[::30]:
        alpha = get_alpha_from_poses(camera_pos[i, :], camera_pos[i + 1, :])
        # car_marker = get_rotated_car_marker(alpha)
        car_marker = get_rotated_svg_marker(alpha=alpha)
        ax.plot(camera_pos[i, 0], camera_pos[i, 2], marker=car_marker, color=color, markersize=25, alpha=0.8)


def plot_camera_trajectory(camera_pos, fig, label: str, color: str = 'red', size: int = 7, alpha: float = 0.5,
                           marker: str = 'o', draw_car: bool = False):
    """
    Plot the trajectory of the left camera.
    :param marker:
    :param draw_car:
    :param alpha:
    :param size:
    :param fig: the figure to plot on
    :param color: the color of the trajectory
    :param label: the label of the trajectory
    :param camera_pos: the camera positions
    :return:
    """
    # plot the calculated trajectory
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()
    # ax.scatter(camera_pos[:, 0], camera_pos[:, 2], s=size, alpha=alpha, color=color, label=label, marker=marker)
    ax.plot(camera_pos[:, 0], camera_pos[:, 2], color=color, linewidth=5, alpha=0.5, label=label)
    if draw_car:
        draw_car_on_trajectory(camera_pos, ax, color)
    return fig


# a function that plots the landmarks in the global 2d coordinate system, and the camera positions,
# and the ground_truth positions
def plot_landmarks_and_camera_poses(landmarks, camera_pos, ground_truth_pos, fig=None):
    if fig is None:
        fig = plot_camera_trajectory(camera_pos, ground_truth_pos)

    ax = fig.gca()
    ax.set_title('Landmarks and camera poses')
    # plot the landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 2], s=1, alpha=0.3, c='grey', label='landmarks')
    fig.legend()
    fig.savefig("landmarks_and_camera_poses.png")
    fig.show()
    return fig


def plot_scene_from_above(result, question=None, marginals=None, scale=1, fig_num=0):
    """
    Function that plots a scene of a certain bundle window from above.
    """
    plot_scene_3d(result, init_view={'azim': 0, 'elev': -90, 'vertical_axis': 'y'}, title="scene from above",
                  question=question, marginals=marginals, scale=scale, fig_num=fig_num)


def plot_scene_3d(result, init_view=None, title="3d scene", marginals=None, scale=1, question='q5_2', fig_num=0):
    """
    Function that plots a scene of a certain bundle window in 3D.
    """
    if init_view is None:
        init_view = {'azim': -15, 'elev': 200, 'vertical_axis': 'y'}
    fig = plt.figure(num=fig_num, figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    gtsam_plot_utils.plot_trajectory(fig_num, result, scale=scale, marginals=marginals)
    gtsam_plot_utils.set_axes_equal(fig_num)

    ax.view_init(**init_view)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.savefig(question + ' ' + title + '.png')  # fig.show()  # plt.clf()


def get_rotated_car_marker(alpha):
    # Create a sedan car shape
    sedan_car = np.array([[-0.4, 0.8],  # Left windshield
                          [-1, 0],  # Bottom left
                          [-1, 0.5],  # Top left
                          [-0.4, 0.8],  # Left windshield
                          [0.4, 0.7],  # Right windshield
                          [1, 0.5],  # Top right
                          [1, 0],  # Bottom right
                          [0.7, 0],  # Right wheel
                          [0.5, -0.2],  # Right wheel bottom
                          [0.4, 0],  # Right wheel inner
                          [-0.4, 0],  # Left wheel inner
                          [-0.5, -0.2],  # Left wheel bottom
                          [-0.7, 0],  # Left wheel
                          [-1, 0]  # Back to start
                          ])
    # Convert alpha to radians
    alpha_rad = np.radians(alpha)

    # Define the rotation matrix
    rotation_matrix = np.array([[np.cos(alpha_rad), -np.sin(alpha_rad)], [np.sin(alpha_rad), np.cos(alpha_rad)]])

    # Apply the rotation to each point
    car_rotated = np.dot(sedan_car, rotation_matrix.T)

    # Create the Path object
    car_marker = mpath.Path(car_rotated, None, closed=True)

    return car_marker


def get_rotated_svg_marker(path='car.svg', alpha=0):
    import matplotlib as mpl
    from svgpathtools import svg2paths
    from svgpath2mpl import parse_path

    planet_path, attributes = svg2paths(path)
    planet_marker = parse_path(attributes[0]['d'])
    planet_marker.vertices -= planet_marker.vertices.mean(axis=0)
    planet_marker = planet_marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
    planet_marker = planet_marker.transformed(mpl.transforms.Affine2D().scale(-1, 1))
    planet_marker = planet_marker.transformed(mpl.transforms.Affine2D().rotate_deg(alpha))
    return planet_marker


def plot_pose_graphs(graph_lst, titles):
    fig, axes = plt.subplots(figsize=(8, 8))
    color_map = plt.get_cmap('rainbow')  # Use colormap of your choice
    num_graphs = len(graph_lst)
    colors = [color_map(i) for i in np.linspace(0, 1, num_graphs + 2)]

    ground_truth_keyframes = np.array(ex3.calculate_camera_trajectory(ex3.get_ground_truth_transformations()))
    fig = plot_ground_truth_trajectory(ground_truth_keyframes, fig, colors[0])

    rel_arr = ex3.calculate_relative_transformations(T_arr=graph_lst[0].T_arr)
    initial_est = utils.get_initial_estimation(rel_t_arr=rel_arr)[graph_lst[0].keyframes]
    fig = plot_camera_trajectory(camera_pos=initial_est, fig=fig, label="initial estimate", color=colors[1])

    for i, (graph, title, color) in enumerate(zip(graph_lst, titles, colors[2:])):
        curr_rel_cameras = graph.get_opt_cameras()
        curr_trajectory = projection_utils.get_trajectory_from_gtsam_poses(curr_rel_cameras)
        if i == len(graph_lst) - 1:
            draw_car = True
        else:
            draw_car = False
        fig = plot_camera_trajectory(camera_pos=curr_trajectory, fig=fig, label=title, color=color, draw_car=draw_car)

    legend_element = plt.legend(loc='upper left', fontsize=12)
    fig.gca().add_artist(legend_element)
    fig.savefig('q7_all all trajectories.png')
    fig.show()
    plt.clf()


def make_moving_car_animation(loop_closure_graph):
    fig, axes = plt.subplots(figsize=(8, 8))
    color_map = plt.get_cmap('rainbow')  # Use colormap of your choice
    num_graphs = 2
    colors = [color_map(i) for i in np.linspace(0, 1, num_graphs)]

    ground_truth_keyframes = np.array(ex3.calculate_camera_trajectory(ex3.get_ground_truth_transformations()))
    fig = plot_ground_truth_trajectory(ground_truth_keyframes, fig, colors[0])
    curr_rel_cameras = loop_closure_graph.get_opt_cameras()
    curr_trajectory = projection_utils.get_trajectory_from_gtsam_poses(curr_rel_cameras)
    fig = plot_camera_trajectory(curr_trajectory, fig, label="Loop Closure Estimation", color=colors[1], draw_car=False)

    legend_element = plt.legend(loc='upper left', fontsize=12)
    fig.gca().add_artist(legend_element)

    last_car_marker = [None]

    def update(i):
        alpha = get_alpha_from_poses(curr_trajectory[i, :], curr_trajectory[i + 1, :])
        # car_marker = get_rotated_car_marker(alpha)
        car_marker = get_rotated_svg_marker(alpha=alpha)
        if last_car_marker[0] is not None:
            last_car_marker[0][0].remove()

        last_car_marker[0] = axes.plot(curr_trajectory[i, 0], curr_trajectory[i, 2], marker=car_marker, color=colors[1],
                                       markersize=40, alpha=0.8)

    anim = FuncAnimation(fig, update, frames=np.arange(0, len(curr_trajectory) - 1, 5), interval=100)
    print("Saving animation...")
    anim.save('q7_loop_closure.gif', dpi=80, writer='pillow')
    plt.clf()
    return anim
