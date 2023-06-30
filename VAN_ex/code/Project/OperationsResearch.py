from VAN_ex.code.PreCalcData.PreCalced import Data
from VAN_ex.code.DataBase.TracksDB import TracksDB
from VAN_ex.code.DataBase.Track import Track
from VAN_ex.code.PoseGraph.PoseGraph import PoseGraph
from VAN_ex.code.BundleAdjustment.BundleAdjustment import BundleAdjustment
from VAN_ex.code.BundleAdjustment.BundleWindow import Bundle

def main():
    data = Data()
    data.load_data(ba=False, tracks_db=True, T_arr=True, pose_graph=True)
    print(data.T_arr)


if __name__ == '__main__':
    main()
