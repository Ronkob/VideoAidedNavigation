import os
import pickle
import time

import cv2
import imageio
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
MAX_RANSAC_ITERATIONS = 5000
k, m1, m2 = ex2_utils.read_cameras()


# 4.1 We call a 3D landmark that was matched across multiple pairs of stereo images (frames) a track.
# In the previous exercise we recognized tracks of length 2 (matched over two pairs of images),
# but we hope to extend the tracks by matching features over more pairs.
# Implement a suitable database for the tracks.
# • Every track should have a unique id, we will refer to it as TrackId.
# • Every image stereo pair should have a unique id, we will refer to it as FrameId.
# • Implement a function that returns all the TrackIds that appear on a given FrameId.
# • Implement a function that returns all the FrameIds that are part of a given TrackId.
class Track:
    """
    A class that represents a track.
    A track is a 3D landmark that was matched across multiple pairs of stereo images (frames).
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
        self.kp = kp  # dictionary of tuples of lists of key-points, each tuple is a pair of key-points

    def __str__(self):
        return f"Track ID: {self.track_id}, Frame IDs: {self.frame_ids}, " \
               f"Key-points: {len(self.kp)}"

    def __repr__(self):
        return str(self)

    def get_track_id(self):
        return self.track_id

    def get_frame_ids(self):
        return self.frame_ids

    def add_frame(self, frame_id, curr_kp, next_kp):
        """
        Add a frame to the track.
        :param frame_id: Frame ID.
        :param kp: list of tuples of key points in both images, for each frame.
        """
        kp_to_keep = [kp for kp in self.kp[self.frame_ids[-1]][0] if kp in curr_kp[0]]
        # idx of kp in curr_kp that are in kp_to_keep
        idx_to_keep = [np.where(curr_kp[0] == kp)[0][0] for kp in kp_to_keep]
        self.kp[frame_id] = next_kp[0][idx_to_keep], next_kp[1][idx_to_keep]
        self.frame_ids.append(frame_id)


class TracksDB:
    """
    A class that represents a database for tracks.
    """

    def __init__(self):
        """
        Initialize a tracks database.
        """
        self.tracks = {}  # a dictionary of tracks
        self.frame_ids = []
        self.track_ids = []
        self.track_id = 0

    def __str__(self):
        return f"Tracks: {self.tracks}, Frame IDs: {self.frame_ids}, " \
               f"Track IDs: {self.track_ids}, Track ID: {self.track_id}"

    def __repr__(self):
        return str(self)

    def add_track(self, track):
        """
        Add a track to the database.
        :param track: Track to add.
        """
        self.tracks[self.track_id] = track  # add track to tracks dictionary by track id key
        self.frame_ids += track.frame_ids
        self.track_ids.append(self.track_id)
        self.track_id += 1

    def get_track_ids(self, frame_id):
        """
        Get all the TrackIds that appear on a given FrameId.
        :param frame_id: Frame ID.
        :return: Track IDs.
        """
        return [track_id for track_id in self.track_ids if frame_id in self.tracks[track_id].frame_ids]

    def get_frame_ids(self, track_id):
        """
        Get all the FrameIds that are part of a given TrackId.
        :param track_id: Track ID.
        :return: Frame IDs.
        """
        return self.tracks[track_id].frame_ids

    # a function that for a given (FrameId, TrackId) pair returns:
    #    o Feature locations of track TrackId on both left and right images as a triplet (𝑥𝑙,𝑥𝑟,𝑦) with:
    #       ▪ (𝑥𝑙,𝑦) the feature location on the left image
    #       ▪ (𝑥𝑟,𝑦) the feature location on the right image Note that the 𝑦 index is shared on both images.
    def get_feature_locations(self, frame_id, track_id):
        """
        Get feature locations of track TrackId on both left and right images.
        :param frame_id: Frame ID.
        :param track_id: Track ID.
        :return: Feature locations of track TrackId on both left and right images.
        """
        track = self.tracks[track_id]
        frame_ids = track.frame_ids
        if frame_id not in frame_ids:
            return None
        frame_index = frame_ids.index(frame_id)  # get the index of the frame id in the track
        kp = track.kp[frame_index]
        return kp.pt

    # Implement an ability to extend the database with new tracks on a new frame as we match new stereo pairs to the
    # previous ones.
    def extend_tracks(self, frame_id, curr_frame_supporters_kp, next_frame_supporters_kp):
        """
        get the matches of a new frame, and add the matches that consistent with the previous frames in the tracks
        as a new frame in every track.
        """
        # treats the kps as unique objects
        # get the tracks that include the previous frame_id
        relevant_tracks = [track_id for track_id in self.track_ids if frame_id - 1 in self.tracks[track_id].frame_ids]
        # get the tracks that include the curr_frame_supporters_kp in the previous frame
        relevant_tracks = [track_id for track_id in relevant_tracks if any(
            kp in self.tracks[track_id].kp[frame_id - 1][0] for kp in curr_frame_supporters_kp[0]) and any(
            kp in self.tracks[track_id].kp[frame_id - 1][1] for kp in curr_frame_supporters_kp[1])]

        # add a new frame to every fitting track with the new frame supporters_kp
        for track_id in relevant_tracks:
            track = self.tracks[track_id]
            track.add_frame(frame_id, curr_frame_supporters_kp, next_frame_supporters_kp)

        # get the set of kp in the relevant tracks
        relevant_kp = {}
        for track_id in relevant_tracks:
            relevant_kp.update(self.tracks[track_id].kp)

        # get the matches that are not in the relevant tracks
        new_matches = (left_kp, right_kp) = next_frame_supporters_kp

        # add the new track to the tracks db
        self.add_new_track(Track(self.get_new_id(), [frame_id], {frame_id: new_matches}))

    # Implement an ability to add a new track to the database.
    def add_new_track(self, track):
        """
        Add a new track to the database.
        :param track: Track to add.
        """
        self.tracks[track.track_id] = track
        self.frame_ids += track.frame_ids
        self.track_ids.append(track.track_id)

    def get_new_id(self):
        """
        Get a new track ID.
        :return: New track ID.
        """
        self.track_id += 1
        return self.track_id - 1

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

    # 4.2 Present the following tracking statistics:
    # • Total number of tracks
    # • Number of frames
    # • Mean track length, maximum and minimum track lengths
    # • Mean number of frame links (number of tracks on an average image)
    def get_statistics(self):
        """
        present a plot of the following tracking statistics:
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


# create a gif of some frames of the video
def create_gif(start_frame, end_frame, tracks_db):
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
    for track_id in [tracks_db.track_ids[0]]:
        track = tracks_db.tracks[track_id]
        for i, frame_id in enumerate(track.frame_ids):
            ims[i].append(
                axes.scatter([kp[0] for kp in track.kp[frame_id][0]], [kp[1] for kp in track.kp[frame_id][0]],
                            color='red'))

    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=3000, blit=True)
    ani.save('run.gif', writer='imagemagick')


def run_sequence(start_frame, end_frame):
    db = TracksDB()
    for idx in range(start_frame, end_frame):
        left_ext_mat, inliers = ex3_utils.track_movement_successive([idx, idx + 1])
        if left_ext_mat is not None:
            left0_kp, right0_kp, left1_kp, right1_kp = inliers
            db.extend_tracks(idx, (left0_kp, right0_kp), (left1_kp, right1_kp))

        print(" -- step {} -- ".format(idx))

    db.get_statistics()
    return db


def run_ex4():
    np.random.seed(1)
    """
    Runs all exercise 4 sections.
    """
    tracks_db = run_sequence(0, 35)
    create_gif(0, 35, tracks_db)


def main():
    run_ex4()


if __name__ == '__main__':
    main()
