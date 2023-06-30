import pickle

import numpy as np

from VAN_ex.code.DataBase.Track import Track
from VAN_ex.code.PreCalcData.paths_to_data import DB_PATH


def save_tracks_db(tracks_db, path):
    """
    Save a tracks database to a pickle file.
    :param tracks_db: Tracks database.
    :param path: Path to save the pickle file to.
    """
    with open(path, 'wb') as f:
        pickle.dump(tracks_db, f)


def load_tracks_db(path=DB_PATH):
    """
    Load a tracks database from a pickle file.
    :param path: Path to load the pickle file from.
    :return: Tracks database.
    """
    with open(path, 'rb') as f:
        tracks_db = pickle.load(f)
    return tracks_db


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
    def extend_tracks(self, curr_frame_idx, curr_frame_supporters_kp, next_frame_supporters_kp):
        """
        Get the matches of a new frame, and add the matches that consistent
         with the previous frames in the tracks as a new frame in every track.
        """
        next_frame_idx = curr_frame_idx + 1

        # treats the kps as unique objects
        # get the tracks that include the previous frame_id
        relevant_tracks = {track_id for track_id in self.track_ids if curr_frame_idx in self.tracks[track_id].frame_ids}

        taken_kp_idxs = []

        left_kp, right_kp = curr_frame_supporters_kp
        for i in range(len(left_kp)):
            for track_id in relevant_tracks:
                track = self.tracks[track_id]
                if left_kp[i] in track.kp[curr_frame_idx][0] and right_kp[i] in track.kp[curr_frame_idx][1]:
                    track.add_frame(next_frame_idx, (next_frame_supporters_kp[0][i], next_frame_supporters_kp[1][i]))
                    taken_kp_idxs.append(i)
                    break  # advance to the next kp

        # Create new tracks for the kps that were not taken
        reminder_left_kp_curr, reminder_right_kp_curr = self.get_reminder_kp(taken_kp_idxs, curr_frame_supporters_kp)
        reminder_left_kp_next, reminder_right_kp_next = self.get_reminder_kp(taken_kp_idxs, next_frame_supporters_kp)
        self.create_new_tracks(frame_id=(curr_frame_idx, next_frame_idx),
                               curr_kp=(reminder_left_kp_curr, reminder_right_kp_curr),
                               next_kp=(reminder_left_kp_next, reminder_right_kp_next))

    def create_new_tracks(self, frame_id, curr_kp, next_kp):
        """
        Creates new tracks, one for each kp in the new frame.
        Get the matches of a new frame, and add the matches that consistent
         with the previous frames in the tracks as a new frame in every track.
        """
        curr_frame_idx, next_frame_idx = frame_id
        left_kp_curr, right_kp_curr = curr_kp
        left_kp_next, right_kp_next = next_kp

        # add the new track to the tracks db
        for i in range(len(left_kp_curr)):
            track = Track(self.get_new_id(), [curr_frame_idx], {curr_frame_idx: (left_kp_curr[i], right_kp_curr[i])})
            track.add_frame(next_frame_idx, (left_kp_next[i], right_kp_next[i]))
            self.add_new_track(track)

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
        id = self.track_id
        self.track_id += 1
        return id

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


def create_loop_tracks(left0_inliers, left1_inliers, i, n):
    loop_tracks = []

    for j in range(len(left0_inliers)):
        left0_inlier, left1_inlier = left0_inliers[i], left1_inliers[i]
        cur_track = Track(j, [i, n], [left0_inlier, left1_inlier])
        loop_tracks.append(cur_track)

    return loop_tracks
