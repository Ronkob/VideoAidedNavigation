from VAN_ex.code.PreCalcData.PreCalced import Data
from VAN_ex.code.DataBase.TracksDB import TracksDB
from VAN_ex.code.DataBase.Track import Track
from VAN_ex.code.PoseGraph.PoseGraph import PoseGraph, save_pg
from VAN_ex.code.BundleAdjustment.BundleAdjustment import BundleAdjustment, save_ba
from VAN_ex.code.BundleAdjustment.BundleWindow import Bundle
from VAN_ex.code.utils import auxilery_plot_utils


def main():
    data = Data()
    # data.load_data(ba=True, tracks_db=True, T_arr=True, pose_graph=False)
    tracks_db = data.get_tracks_db()
    # ba = data.get_ba()
    # print("ba loaded")
    # pg = data.get_pose_graph()
    # print("pg loaded")
    print("")

if __name__ == '__main__':
    main()
