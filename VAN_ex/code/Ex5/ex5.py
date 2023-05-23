import os
import pickle
import gtsam
import numpy as np
import matplotlib.pyplot as plt

import VAN_ex.code.Ex3.ex3 as ex3_utils
import VAN_ex.code.Ex4.ex4 as ex4_utils

DB_PATH = os.path.join('..', 'Ex4', 'tracks_db.pkl')
T_ARR_PATH = os.path.join('..', 'Ex3', 'T_arr.npy')
old_k, m1, m2 = ex4_utils.k, ex4_utils.m1, ex4_utils.m2


def q5_1(track_db):
    track = ex4_utils.get_rand_track(10, track_db)
    triangulate_and_project(track)


def triangulate_and_project(track):
    """
    For all the frames participating in this track, define a gtsam.StereoCamera
    using the global camera matrices calculated in exercise 3 (PnP).
    Using methods in StereoCamera, triangulate a 3d point in global coordinates
    from the last frame of the track, and project this point to all the frames
    of the track (both left and right cameras).
    • Present a graph of the reprojection error size (L2 norm) over the track’s images.
    • Create a factor for each frame projection and present a graph of the
      factor error over the track’s frames.
    """
    T_arr = np.load("T_arr.npy")
    track_frames = track.kp
    values = gtsam.Values()
    last_frame_id = track.frames[-1]
    last_frame_pt = track.kp[-1]
    base_cam = T_arr[0]
    fx, fy, skew = old_k[0, 0], old_k[1, 1], old_k[0, 1]
    cx, cy, baseline = old_k[0, 2], old_k[1, 2], m2[0, 3]
    K = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -baseline)
    for i, frame in enumerate(track_frames):
        ext_mat = T_arr[i]
        pose = ex3_utils.composite_transformations(base_cam, T_arr[i])
        pose = gtsam.Pose3(pose)
        symbol = gtsam.symbol('c', track.get_frame_ids()[i])
        values.insert(symbol, pose)


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
