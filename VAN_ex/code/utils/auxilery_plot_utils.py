# a function that gets a Track object, and plots its points tracking
from matplotlib import pyplot as plt

from VAN_ex.code.Ex4.ex4 import Track


def plot_tracking_quality(track: Track):
    pass


def plot_ground_truth_trajectory(ground_truth_pos, fig=None):
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()
    # make the title in a beautiful, large font
    ax.set_title('Trajectory of the left camera', fontsize=20, fontweight='bold')

    # plot the ground truth trajectory
    plot_camera_trajectory(ground_truth_pos, color='blue', fig=fig, label='ground truth')
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


def plot_camera_trajectory(camera_pos,  fig, label:str, color: str = 'red', size: int = 7, alpha: float = 0.5):
    """
    Plot the trajectory of the left camera.
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
    ax.scatter(camera_pos[:, 0], camera_pos[:, 2], s=size, alpha=0.5, c=color, label=label)
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


def plot_initial_est_on_axs(initial_est, camera_pos, ground_truth_pos, landmarks=None, fig=None):
    if fig is None:
        fig = plot_camera_trajectory(camera_pos, ground_truth_pos)
        if landmarks:
            fig = plot_landmarks_and_camera_poses(landmarks, camera_pos, ground_truth_pos, fig)

    ax = fig.gca()
    ax.set_title('Initial estimation compare')
    # plot the initial camera positions
    ax.scatter(initial_est[:, 0], initial_est[:, 2], s=5, alpha=0.5, c='blue', label='initial_estimation')
    fig.legend()
    fig.savefig("initial_estimation.png")
    fig.show()
