from VAN_ex.code.PreCalcData.PreCalced import Data
from VAN_ex.code.DataBase.TracksDB import TracksDB
from VAN_ex.code.DataBase.Track import Track
from VAN_ex.code.PoseGraph.PoseGraph import PoseGraph
from VAN_ex.code.BundleAdjustment.BundleAdjustment import BundleAdjustment, save_ba
from VAN_ex.code.BundleAdjustment.BundleWindow import Bundle
from VAN_ex.code.utils import auxilery_plot_utils


def main():
    data = Data()
    # data.load_data(ba=True, tracks_db=True, T_arr=True, pose_graph=False)
    ba = data.get_ba()
    print("ba loaded")
    # pg = data.get_pose_graph()
    # auxilery_plot_utils.plot_scene_from_above(pg.result, question='projectTesting', marginals=pg.get_marginals())
    save_ba(ba, 'ba')
    print("ba saved")


if __name__ == '__main__':
    main()
