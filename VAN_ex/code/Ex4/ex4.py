import os
import pickle
import time

import cv2
import numpy as np
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

    def __init__(self, track_id, frame_ids, kp, desc):
        """
        Initialize a track.
        :param track_id: Track ID.
        :param frame_ids: Frame IDs.
        :param kp: Key-points.
        :param desc: Descriptors.
        """
        self.track_id = track_id
        self.frame_ids = frame_ids
        self.kp = kp
        self.desc = desc

    def __str__(self):
        return f"Track ID: {self.track_id}, Frame IDs: {self.frame_ids}, " \
               f"Key-points: {len(self.kp)}, Descriptors: {len(self.desc)}"

    def __repr__(self):
        return str(self)

    def get_track_id(self):
        return self.track_id

    def get_frame_ids(self):
        return self.frame_ids


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
    def extend_tracks(self, frame_id, kp, desc):
        # todo: implement
        pass

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

def run_ex4():
    np.random.seed(1)
    """
    Runs all exercise 4 sections.
    """


def main():
    run_ex4()


if __name__ == '__main__':
    main()
