import os
import gtsam
import pickle
import numpy as np

from gtsam.utils import plot
import matplotlib.pyplot as plt

from VAN_ex.code.Ex3.ex3 import calculate_relative_transformations
from VAN_ex.code.Ex4.ex4 import TracksDB, Track
from VAN_ex.code.Ex5.ex5 import plot_scene_3d, plot_scene_from_above
from VAN_ex.code.BundleAdjustment import BundleWindow
from VAN_ex.code.PoseGraph.PoseGraph import PoseGraph
from VAN_ex.code.utils import projection_utils, auxilery_plot_utils

DB_PATH = os.path.join('..', 'Ex4', 'tracks_db.pkl')
T_ARR_PATH = os.path.join('..', 'Ex3', 'T_arr.npy')


def q7_1(pose_graph: PoseGraph, idx: int):
    """
    Detect Loop Closure Candidates.
    a. Relative Covariance - find the shortest path from c_n to c_i and sum the covariances along the path to get an
     estimate of the relative covariance.
    b. Detect Possible Candidates - choose the most likely candidate to be close to the pose c_n by applying a
    Mahalanobis distance test with 𝑐_n,i - the relative pose between 𝑐_n and 𝑐_i.
    Choose a threshold to determine if the candidate advances to the next (expensive) stage.
    """
    candidate_frames = []
    cn_pose = pose_graph.result.atPose3(gtsam.symbol('c', idx))

    for i in range(idx):
        ci_pose = pose_graph.result.atPose3(gtsam.symbol('c', i))




def q7_2():
    """
    Consensus Matching - perform consensus match between the two candidate frames. (See exercise 3)
    Set a threshold for the number of inliers that indicates a successful match. Note that this is
    typically a more difficult match than that of two consecutive frames
    """
    pass


def q7_3():
    """
    Relative Pose Estimation - using the inlier matches perform a small Bundle optimization to extract the relative
     pose of the two frames as well as its covariance.
    """
    pass


def q7_4():
    """
    Update the Pose Graph - add the resulting synthetic measurement to the pose graph and optimize it to update the
    trajectory estimate.
    """
    pass


def q7_5():
    """
    Display plots.
    """
    pass


def run_ex7():
    """
    Runs all exercise 7 sections.
    """
    np.random.seed(1)
    # Load tracks DB
    tracks_db = TracksDB.deserialize(DB_PATH)
    T_arr = np.load(T_ARR_PATH)
    rel_t_arr = calculate_relative_transformations(T_arr)

    pose_graph = PoseGraph(tracks_db, T_arr)

    # For each key frame 𝑐𝑛 in the pose graph loop over previous keyframes 𝑐𝑖, 𝑖 < 𝑛, and perform
    # steps 7.1-7.4:
    for i in range(1, len(pose_graph.keyframes)):
        # Detect Loop Closure Candidates
        q7_1(pose_graph, i)

        # 7.2 Consensus Matching
        # q7_2()

        # 7.3 Relative Pose Estimation
        # q7_3()

        # 7.4 Update the Pose Graph
        # q7_4()

    # q7_5()


def main():
    run_ex7()


if __name__ == '__main__':
    main()
