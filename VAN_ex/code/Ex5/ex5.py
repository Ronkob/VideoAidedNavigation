import os
import pickle
import gtsam
import numpy as np
import matplotlib.pyplot as plt

import VAN_ex.code.Ex3.ex3 as ex3_utils
import VAN_ex.code.Ex4.ex4 as ex4_utils

DB_PATH = os.path.join('..', 'Ex4', 'tracks_db.pkl')


def q5_1(track_db):
    track = ex4_utils.get_rand_track(10, track_db)
    stereo_cameras = define_stereo_cameras(track)
    triangulate_and_project(track, stereo_cameras)


def define_stereo_cameras(track):
    """
    For all the frames participating in this track, define a gtsam.StereoCamera
    using the global camera matrices calculated in exercise 3 (PnP).
    """
    track_frames = track.kp


def triangulate_and_project(track, stereo_camera):
    """
    Using methods in StereoCamera, triangulate a 3d point in global coordinates
    from the last frame of the track, and project this point to all the frames
    of the track (both left and right cameras).
    • Present a graph of the reprojection error size (L2 norm) over the track’s images.
    • Create a factor for each frame projection and present a graph of the
      factor error over the track’s frames.
    """
    pass


def plot_reprojection_error():
    """
    Present a graph of the reprojection error size (L2 norm) over the track’s images.
    """
    plt.title("Reprojection error over track's images")
    plt.ylabel('Error')
    plt.xlabel('Frames')
    plt.scatter()
    plt.show()


def run_ex5():
    """
    Runs all exercise 5 sections.
    """
    ex4_utils.TracksDB()
    tracks_db = ex4_utils.TracksDB.deserialize(DB_PATH)

    q5_1(tracks_db)


def main():
    run_ex5()


if __name__ == '__main__':
    main()
