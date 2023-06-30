from numpy import load as np_load, save as np_save

from VAN_ex.code.BundleAdjustment.BundleAdjustment import BundleAdjustment, save_ba, load_ba
from VAN_ex.code.DataBase.TracksDB import TracksDB, save_tracks_db, load_tracks_db
from VAN_ex.code.PoseGraph.PoseGraph import PoseGraph, save_pos_graph, load_pos_graph
from VAN_ex.code.Ex3 import ex3
from VAN_ex.code.Ex4 import ex4
from VAN_ex.code.PreCalcData.paths_to_data import BA_PATH, DB_PATH, T_ARR_PATH, PG_PATH


class Data:
    """a singletons data class that holds all the data needed for the project"""

    def __init__(self):
        self.tracks_db = None
        self.T_arr = None
        self.inliers_lst = None
        self.percents_lst = None
        self.ba = None
        self.pose_graph = None

    def load_data(self, ba: bool = True, tracks_db: bool = True, T_arr: bool = True, pose_graph: bool = True):
        self.T_arr = self.get_T_arr()
        self.tracks_db = self.get_tracks_db()
        self.ba = self.get_ba()
        # self.pose_graph = self.get_pose_graph()

    def get_T_arr(self):
        if self.T_arr is None or self.inliers_lst is None or self.percents_lst is None:
            try:
                self.T_arr = np_load(T_ARR_PATH)

            except FileNotFoundError:
                self.T_arr, self.inliers_lst, self.percents_lst = create_T_arr()
                np_save(T_ARR_PATH, self.T_arr)

        return self.T_arr

    def get_tracks_db(self):
        if self.tracks_db is None:
            try:
                self.tracks_db = load_tracks_db(DB_PATH)
            except FileNotFoundError:
                self.T_arr = self.get_T_arr()
                self.tracks_db = create_tracks_db(self.T_arr, self.inliers_lst, self.percents_lst)
                save_tracks_db(self.tracks_db, DB_PATH)
        return self.tracks_db

    def get_ba(self):
        if self.ba is None:
            try:
                self.ba = load_ba(BA_PATH)
            except FileNotFoundError:
                self.ba = create_ba(self.get_tracks_db(), self.get_T_arr())
                save_ba(self.ba, BA_PATH)
        return self.ba

    def get_pose_graph(self):
        if self.pose_graph is None:
            try:
                self.pose_graph = load_pos_graph(PG_PATH)
            except FileNotFoundError:
                self.pose_graph = create_pose_graph(self.ba)
                save_pos_graph(self.pose_graph, PG_PATH)

        return self.pose_graph




def create_T_arr():
    print("creating T_arr...")
    t_arr, inliers_lst, percents_lst = ex3.track_movement_all_movie()
    return t_arr, inliers_lst, percents_lst


def create_tracks_db(t_arr, inliers_lst, percents_lst):
    print("creating tracks db...")
    db = TracksDB()
    start_frame = 0
    end_frame = 2559

    for idx in range(start_frame, end_frame):
        left_ext_mat, inliers, inliers_precent = t_arr[idx], inliers_lst[idx], percents_lst[idx]
        if left_ext_mat is not None:
            left0_kp, right0_kp, left1_kp, right1_kp = inliers
            db.extend_tracks(idx, (left0_kp, right0_kp), (left1_kp, right1_kp))
        else:
            print("something went wrong, no left_ext_mat")
        print(" -- Step {} -- ".format(idx))
    return db


def create_ba(tracks_db, T_arr):
    print("creating ba...")
    ba = BundleAdjustment(tracks_db, T_arr)
    ba.choose_keyframes('end_frame')
    ba.solve()
    return ba


def create_pose_graph(tracks_db: TracksDB, T_arr):
    print("creating pose graph...")
    assert (tracks_db.frame_ids == T_arr.shape[0]), "tracks_db and T_arr must have the same number of frames"
    pg = PoseGraph(tracks_db, T_arr)
    pg.solve()
    return pg
