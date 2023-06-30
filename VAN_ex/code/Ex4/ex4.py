import os
import numpy as np
from matplotlib import animation

from VAN_ex.code.DataBase.TracksDB import TracksDB
from VAN_ex.code.utils import utils as utils
import matplotlib.pyplot as plt
import VAN_ex.code.Ex1.ex1 as ex1_utils
import VAN_ex.code.Ex2.ex2 as ex2_utils
import VAN_ex.code.Ex3.ex3 as ex3_utils

# Constants #
MOVIE_LENGTH = 2559
CAM_TRAJ_PATH = os.path.join('..', '..', 'dataset', 'poses', '05.txt')
N_FEATURES = 1000
PNP_POINTS = 4
CONSENSUS_ACCURACY = 2
MAX_RANSAC_ITERATIONS = 1000
START_FRAME = 0
END_FRAME = 50
TRACK_MIN_LEN = 10
DB_PATH = "tracks_db.pkl"
k, m1, m2 = ex2_utils.read_cameras()
# End Constants #


@utils.measure_time
def create_gif(start_frame, end_frame, tracks_db):
    """
    Create a gif of some frames of the video.
    """
    # add the frames to a list
    images = []
    for frame in range(start_frame, end_frame):
        left0_image, _ = ex1_utils.read_images(frame)
        images.append(left0_image)

    fig, axes = plt.subplots(figsize=(12, 6))
    plt.axis("off")
    fig.suptitle(f"Run", fontsize=16)
    fig.tight_layout()
    ims = [[axes.imshow(i, animated=True, cmap='gray')] for i in images]
    # add a scatter plot of the tracks
    # create a dictionary of colors from mpl colormap
    cmap = plt.get_cmap('tab20')
    # reverse order of tracks_db.track_ids
    # reversed_idx = tracks_db.track_ids[::-1]
    # only tracks that have at least 10 frames
    tracks_to_show = [track_id for track_id in tracks_db.track_ids if
                      len(tracks_db.tracks[track_id].frame_ids) > TRACK_MIN_LEN]
    for i, track_id in enumerate(tracks_to_show):
        track = tracks_db.tracks[track_id]
        color = cmap(i % 20)
        for frame_id in track.frame_ids:
            # if frame_id is in ims range
            if frame_id - start_frame < len(ims):
                ims[frame_id].append(
                    axes.scatter(track.kp[frame_id][0][0], track.kp[frame_id][0][1], color=color, animated=True))

    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=3000, blit=True)
    # save but compress it first so it won't be too big
    ani.save('run.gif', writer='pillow', fps=5, dpi=80)


def get_rand_track(track_len, tracks):
    """
    Get a randomized track with length of at least track_len.
    """
    track_id = np.random.choice(tracks.track_ids)
    track = tracks.tracks[track_id]
    while len(track.frame_ids) < track_len:
        track_id = np.random.choice(tracks.track_ids)
        track = tracks.tracks[track_id]
    return track


# q4.3
def plot_random_track(track):
    """
    Randomize track, and display the feature locations on all the relevant
    images. Cut a region of 100x100 pixels (subject to image boundaries)
    around the feature from both left and right images andmark the feature
    as a dot. Present this for all images in the track.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.set_title('Left image')
    ax2.set_title('Right image')
    ims = []
    for frame_id in track.frame_ids:
        left0_image, right0_image = ex1_utils.read_images(frame_id)
        left_x_cor = track.kp[frame_id][0][0]
        left_y_cor = track.kp[frame_id][0][1]
        right_x_cor = track.kp[frame_id][1][0]
        right_y_cor = track.kp[frame_id][1][1]
        left_x_cor_rounded = int(np.floor(left_x_cor))
        left_y_cor_rounded = int(np.floor(left_y_cor))
        right_x_cor_rounded = int(np.floor(right_x_cor))
        right_y_cor_rounded = int(np.floor(right_y_cor))
        left0_image = left0_image[left_y_cor_rounded - 50:left_y_cor_rounded + 50,
                      left_x_cor_rounded - 50:left_x_cor_rounded + 50]
        right0_image = right0_image[right_y_cor_rounded - 50:right_y_cor_rounded + 50,
                       right_x_cor_rounded - 50:right_x_cor_rounded + 50]

        ims.append([ax1.imshow(left0_image, cmap='gray'),
                    ax1.scatter(50 + (left_x_cor - left_x_cor_rounded), 50 + (left_y_cor - left_y_cor_rounded),
                                color='red', marker='^'), ax2.imshow(right0_image, cmap='gray'),
                    ax2.scatter(50 + (right_x_cor - right_x_cor_rounded), 50 + (right_y_cor - right_y_cor_rounded),
                                color='red', marker='^'), ])
    ani = animation.ArtistAnimation(fig, ims, interval=100)
    ani.save(f"track_cut_around_{track.track_id}.gif", writer="pillow", fps=5)

    plt.close()
    plt.clf()


# q4.4
def plot_connectivity_graph(tracks_db):
    """
    Plot a connectivity graph of the tracks. For each frame, the number of
    tracks outgoing to the next frame (the number of tracks on the frame with
     links also in the next frame)
    """
    outgoing_tracks = []
    frames = list(tracks_db.frame_ids)[:-1]  # Exclude the last frame

    for frame in frames:
        curr_tracks = tracks_db.get_track_ids(frame)
        next_tracks = tracks_db.get_track_ids(frame + 1)
        # Count the shared tracks between the two frames
        num_tracks = len(set(curr_tracks).intersection(next_tracks))
        outgoing_tracks.append(num_tracks)

    plt.title('Connectivity Graph')
    plt.xlabel('Frame')
    plt.ylabel('Outgoing tracks')
    plt.axhline(y=np.array(outgoing_tracks).mean(), color='green', linestyle='--')
    plt.plot(frames, outgoing_tracks)
    plt.show()


# q4.6
def plot_track_length_histogram(tracks_db):
    """
    Present a track length histogram graph, according to the tracks in the db.
    """
    track_lengths = [len(tracks_db.tracks[track_id].frame_ids) for track_id in tracks_db.track_ids]
    x_axis = [i for i in range(max(track_lengths))]
    num_tracks = [track_lengths.count(i) for i in x_axis]

    plt.title('Track length histogram')
    plt.xlabel('Track length')
    plt.ylabel('Track #')
    plt.plot(x_axis, num_tracks)
    plt.show()


def read_gt_cam_mat():
    """
    Read the ground truth camera matrices (in \poses\05.txt).
    """
    gt_cam_matrices = list()

    with open(CAM_TRAJ_PATH) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        gt_cam_matrices.append(np.array(line).reshape(3, 4).astype(np.float64))

    return gt_cam_matrices


# q4.7
def plot_reprojection_error(tracks_db):
    """
    Present a graph of the reprojection error over the track’s images
    """
    # Read the ground truth camera matrices (in \poses\05.txt)
    gt_cam_matrices = ex3_utils.get_ground_truth_transformations()

    # Triangulate a 3d point in world coordinates from the features in the last frame of the track
    track = get_rand_track(TRACK_MIN_LEN, tracks_db)

    left_locations = track.get_left_kp()
    right_locations = track.get_right_kp()

    last_gt_mat = gt_cam_matrices[max(track.frame_ids)]
    last_left_proj_mat = k @ last_gt_mat
    last_right_proj_mat = k @ ex3_utils.composite_transformations(last_gt_mat, m2)

    last_left_img_coords = left_locations[track.frame_ids[-1]]
    last_right_img_coords = right_locations[track.frame_ids[-1]]
    p3d = utils.triangulate_points(last_left_proj_mat, last_right_proj_mat, [last_left_img_coords],
                                   [last_right_img_coords])

    # Project this point to all the frames of the track (both left and right cameras)
    left_projections, right_projections = [], []

    for gt_cam_mat in gt_cam_matrices[min(track.frame_ids):max(track.frame_ids) + 1]:
        left_proj_cam = k @ gt_cam_mat
        left_projections.append(utils.project(p3d, left_proj_cam)[0])

        right_proj_cam = k @ ex3_utils.composite_transformations(gt_cam_mat, m2)
        right_projections.append(utils.project(p3d, right_proj_cam)[0])

    left_projections, right_projections = np.array(left_projections), np.array(right_projections)

    # We’ll define the reprojection error for a given camera as the distance between the projection
    # and the tracked feature location on that camera.

    # Calculate the reprojection error for each frame of the track
    left_locations = np.array(list(left_locations.values()))
    right_locations = np.array(list(right_locations.values()))

    left_proj_dist = np.linalg.norm(left_projections - left_locations, axis=1)
    right_proj_dist = np.linalg.norm(right_projections - right_locations, axis=1)
    total_proj_dist = (left_proj_dist + right_proj_dist) / 2

    # Present a graph of the reprojection error over the track’s images.
    plt.title("Reprojection error over track's images")
    plt.ylabel('Error')
    plt.xlabel('Frames')
    plt.scatter(range(min(track.frame_ids), max(track.frame_ids) + 1), total_proj_dist)
    plt.show()


# q4.5
def plot_inliers_per_frame(inliers_precent, frames):
    """
    Present a graph of the percentage of inliers per frame.
    """
    plt.title('Inliers per frame')
    plt.xlabel('Frame')
    plt.ylabel('Inliers')
    plt.plot(frames, inliers_precent)
    plt.axhline(y=np.array(inliers_precent).mean(), color='green', linestyle='--')
    plt.show()


@utils.measure_time
def run_sequence(start_frame, end_frame):
    db = TracksDB()
    for idx in range(start_frame, end_frame):
        left_ext_mat, inliers, inliers_precent = ex3_utils.track_movement_successive([idx, idx + 1])
        if left_ext_mat is not None:
            left0_kp, right0_kp, left1_kp, right1_kp = inliers
            db.extend_tracks(idx, (left0_kp, right0_kp), (left1_kp, right1_kp))
        else:
            print("something went wrong, no left_ext_mat")
        print(" -- Step {} -- ".format(idx))
    # frames = [i for i in range(start_frame, end_frame)]
    # plot_inliers_per_frame(inliers_precent_lst, frames)  # q4.5
    # db.remove_short_tracks(short=2)
    db.serialize(DB_PATH)
    return db


def run_ex4():
    """
    Runs all exercise 4 sections.
    """
    np.random.seed(7)
    tracks_db = None
    tracks_db = run_sequence(START_FRAME, MOVIE_LENGTH)  # Build the tracks database
    if tracks_db is None:
        tracks_db = TracksDB.deserialize(DB_PATH)

    # # q4.2
    tracks_db.get_statistics()
    #
    # # q4.3
    track = get_rand_track(10, tracks_db)
    plot_random_track(track)
    #
    # # q4.4
    plot_connectivity_graph(tracks_db)
    #
    # #q4.6
    plot_track_length_histogram(tracks_db)
    #
    # # q4.7
    plot_reprojection_error(tracks_db)

    create_gif(START_FRAME, END_FRAME, tracks_db)


def main():
    run_ex4()


if __name__ == '__main__':
    main()
