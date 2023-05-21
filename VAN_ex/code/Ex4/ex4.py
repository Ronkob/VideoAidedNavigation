import os
import pickle
import numpy as np
from matplotlib import animation

import VAN_ex.code.utils as utils
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


class Track:
    """
    A class that represents a track.
    A track is a 3D landmark that was matched across multiple pairs of stereo images (frames).
    Every track will have a unique id, we will refer to it as track_id.
    Every image stereo pair will have a unique id, we will refer to it as frame_id.
    """

    def __init__(self, track_id, frame_ids, kp):
        """
        Initialize a track.
        :param track_id: Track ID.
        :param frame_ids: Frame IDs.
        :param kp: list of tuples of key points in both images, for each frame.
        """
        self.track_id = track_id
        self.frame_ids = frame_ids
        self.kp = kp  # one tuple of key points for each frame

    def __str__(self):
        return f"Track ID: {self.track_id}, Frame IDs: {self.frame_ids}, " \
               f"Key-points: {len(self.kp)}, length: {len(self.frame_ids)}"

    def __repr__(self):
        return str(self)

    def get_track_id(self):
        return self.track_id

    def get_frame_ids(self):
        return self.frame_ids

    def add_frame(self, frame_id, next_kp):
        """
        Add a frame to the track.
        :param frame_id: Frame ID.
        :param curr_kp: Key-points of the current frame.
        :param next_kp: Key-points of the next frame.
        """
        self.kp[frame_id] = next_kp
        self.frame_ids.append(frame_id)

    # get all the left kp of the track
    def get_left_kp(self):
        return {kp: self.kp[kp][0] for kp in self.kp}

    # get all the right kp of the track as a dictionary
    def get_right_kp(self):
        return {kp: self.kp[kp][1] for kp in self.kp}


# q4.1
class TracksDB:
    """
    A class that represents a database for tracks.
    """

    def __init__(self):
        """
        Initialize a tracks database.
        """
        self.tracks = {}  # Dictionary of all tracks
        self.frame_ids = set()
        self.track_ids = []
        self.track_id = 0  # Track ID counter

    def __str__(self):
        return f"Tracks: {self.tracks}, Frame IDs: {self.frame_ids}, " \
               f"Track IDs: {self.track_ids}, Track ID: {self.track_id}"

    def __repr__(self):
        return str(self)

    def get_track_ids(self, frame_id):
        """
        Get all the track_ids that appear on a given frame_id.
        :param frame_id: Frame ID.
        :return: Track IDs.
        """
        return [track_id for track_id in self.track_ids if frame_id in self.tracks[track_id].frame_ids]

    def get_frame_ids(self, track_id):
        """
        Get all the frame_ids that are part of a given track_id.
        :param track_id: Track ID.
        :return: Frame IDs.
        """
        return self.tracks[track_id].frame_ids

    def remove_short_tracks(self, short=2):
        """
        Remove tracks that are too short (less than 2 frames).
        """
        id_to_remove = []
        for track_id in self.track_ids:
            if len(self.tracks[track_id].frame_ids) < short:
                id_to_remove.append(track_id)

        for track_id in id_to_remove:
            self.remove_track(track_id)

    def remove_track(self, track_id):
        """
        Remove a track from the database.
        :param track_id: TrackID to remove.
        """
        self.tracks.pop(track_id)
        self.track_ids.remove(track_id)

    # Implement an ability to extend the database with new tracks on a new
    # frame as we match new stereo pairs to the previous ones.
    def extend_tracks(self, frame_id, curr_frame_supporters_kp, next_frame_supporters_kp):
        """
        Get the matches of a new frame, and add the matches that consistent
         with the previous frames in the tracks as a new frame in every track.
        """
        # treats the kps as unique objects
        # get the tracks that include the previous frame_id
        relevant_tracks = [track_id for track_id in self.track_ids if frame_id - 1 in self.tracks[track_id].frame_ids]

        taken_kp_idxs = []

        left_kp, right_kp = curr_frame_supporters_kp
        for i in range(len(left_kp)):
            for track_id in relevant_tracks:
                track = self.tracks[track_id]
                if left_kp[i] in track.kp[frame_id - 1][0] and right_kp[i] in track.kp[frame_id - 1][1]:
                    track.add_frame(frame_id, (next_frame_supporters_kp[0][i], next_frame_supporters_kp[1][i]))
                    taken_kp_idxs.append(i)
                    break  # advance to the next kp

        # Remove tracks that are too short (less than 2 frames)
        self.remove_short_tracks(short=2)

        # Create new tracks for the kps that were not taken
        reminder_left_kp, reminder_right_kp = self.get_reminder_kp(taken_kp_idxs, next_frame_supporters_kp)
        self.create_new_tracks(frame_id, reminder_left_kp, reminder_right_kp)

    def create_new_tracks(self, frame_id, left_kp, right_kp):
        """
        Creates new tracks, one for each kp in the new frame.
        Get the matches of a new frame, and add the matches that consistent
         with the previous frames in the tracks as a new frame in every track.
        """
        # add the new track to the tracks db
        for i in range(len(left_kp)):
            self.add_new_track(Track(self.get_new_id(), [frame_id], {frame_id: (left_kp[i], right_kp[i])}))

    # Implement an ability to add a new track to the database.
    def add_new_track(self, track):
        """
        Add a new track to the database.
        :param track: Track to add.
        """
        self.tracks[track.track_id] = track
        self.frame_ids = self.frame_ids.union(track.frame_ids)
        self.track_ids.append(track.track_id)

    def get_new_id(self):
        """
        Get a new track ID.
        :return: New track ID.
        """
        self.track_id += 1
        return self.track_id

    def feature_location(self, frame_id, track_id):
        """
        Returns feature locations of track TrackId on both left and right
         images as a triplet (xl, xr, y).
        """
        if self.tracks[track_id] and frame_id in self.tracks[track_id].frame_ids:
            xl = self.tracks[track_id].kp[frame_id][0][0]
            xr = self.tracks[track_id].kp[frame_id][1][0]
            y = self.tracks[track_id].kp[frame_id][0][1]
            return xl, xr, y
        else:
            return None


    # Implement functions to serialize the database to a file and read it from a file.
    def serialize(self, file_name):
        """
        Serialize the database to a file.
        :param file_name: File name.
        """
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def deserialize(file_name):
        """
        Deserialize the database from a file, and return a new TracksDB object.
        :param file_name: File name.
        """
        with open(file_name, 'rb') as file:
            tracks_db = pickle.load(file)
        return tracks_db

    # q4.2
    def get_statistics(self):
        """
        Present a plot of the following tracking statistics:
        • Total number of tracks
        • Number of frames
        • Mean track length, maximum and minimum track lengths
        • Mean number of frame links (number of tracks on an average image)
        """
        # get the number of tracks
        num_tracks = len(self.track_ids)
        # get the number of frames
        num_frames = len(self.frame_ids)
        # get the track lengths
        track_lengths = [len(self.tracks[track_id].frame_ids) for track_id in self.track_ids]
        # get the mean track length
        mean_track_length = np.mean(track_lengths)
        # get the maximum track length
        max_track_length = np.max(track_lengths)
        # get the minimum track length
        min_track_length = np.min(track_lengths)
        # get the mean number of frame links
        mean_num_frame_links = np.mean([len(self.get_track_ids(frame_id)) for frame_id in self.frame_ids])
        # print the statistics
        print('Total number of tracks: {}'.format(num_tracks))
        print('Number of frames: {}'.format(num_frames))
        print('Mean track length: {}'.format(mean_track_length))
        print('Maximum track length: {}'.format(max_track_length))
        print('Minimum track length: {}'.format(min_track_length))
        print('Mean number of frame links: {}'.format(mean_num_frame_links))

    @staticmethod
    def get_reminder_kp(taken_kp_idxs, next_frame_supporters_kp):
        # create an array of free indexs of the kps that were not taken, efficiant for large arrays
        # start with all the indexes as a boolean array
        reminder_kp_idxs = np.ones(len(next_frame_supporters_kp[0]), dtype=bool)
        # set the taken indexes to false
        reminder_kp_idxs[taken_kp_idxs] = False
        reminder_kp_idxs = np.where(reminder_kp_idxs)[0]
        # create new tracks for the kps that were not taken
        return next_frame_supporters_kp[0][reminder_kp_idxs], next_frame_supporters_kp[1][reminder_kp_idxs]


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
    reversed_idx = tracks_db.track_ids[::-1]
    # only tracks that have at least 10 frames
    tracks_to_show = [track_id for track_id in reversed_idx if
                      len(tracks_db.tracks[track_id].frame_ids) > TRACK_MIN_LEN]
    for i, track_id in enumerate(tracks_to_show):
        track = tracks_db.tracks[track_id]
        color = cmap(i % 20)
        for frame_id in track.frame_ids:
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
def plot_random_track(tracks_db):
    """
    Randomize track, and display the feature locations on all the relevant
    images. Cut a region of 100x100 pixels (subject to image boundaries)
    around the feature from both left and right images andmark the feature
    as a dot. Present this for all images in the track.
    """
    track = get_rand_track(TRACK_MIN_LEN, tracks_db)
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
    ani.save("track_cut_around.gif", writer="pillow", fps=5)

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
    plt.axhline(y=np.array(inliers_precent).mean(), color='green',
                linestyle='--')
    plt.show()


@utils.measure_time
def run_sequence(start_frame, end_frame):
    db = TracksDB()
    # inliers_precent_lst = []
    for idx in range(start_frame, end_frame):
        left_ext_mat, inliers, inliers_precent =\
            ex3_utils.track_movement_successive([idx, idx + 1])
        if left_ext_mat is not None:
            left0_kp, right0_kp, left1_kp, right1_kp = inliers
            db.extend_tracks(idx, (left0_kp, right0_kp), (left1_kp, right1_kp))
            # inliers_precent_lst.append(inliers_precent)
        print(" -- Step {} -- ".format(idx))
    # frames = [i for i in range(start_frame, end_frame)]
    # plot_inliers_per_frame(inliers_precent_lst, frames)  # q4.5
    db.remove_short_tracks(short=2)
    # db.serialize(DB_PATH)
    return db


def run_ex4():
    """
    Runs all exercise 4 sections.
    """
    np.random.seed(4)
    tracks_db = None
    ## tracks_db = run_sequence(START_FRAME, MOVIE_LENGTH)  # Build the tracks database
    if tracks_db is None:
        tracks_db = TracksDB.deserialize(DB_PATH)

    # q4.2
    # tracks_db.get_statistics()

    # q4.3
    # plot_random_track(tracks_db)

    # q4.4
    # plot_connectivity_graph(tracks_db)

    # q4.6
    # plot_track_length_histogram(tracks_db)

    # q4.7
    # plot_reprojection_error(tracks_db)

    # create_gif(START_FRAME, END_FRAME, tracks_db)


def main():
    run_ex4()


if __name__ == '__main__':
    main()
