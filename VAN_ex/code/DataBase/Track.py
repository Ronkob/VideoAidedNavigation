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
