import matplotlib.pyplot as plt

from VAN_ex.code.BundleAdjustment.BundleWindow import Bundle
from VAN_ex.code.BundleAdjustment.BundleAdjustment import BundleAdjustment
from VAN_ex.code.DataBase.TracksDB import TracksDB
from VAN_ex.code.DataBase.Track import Track
from VAN_ex.code.PoseGraph.PoseGraph import PoseGraph
from VAN_ex.code.PreCalcData.PreCalced import Data


def proj_error_pnp():
    """
    Median (or any other meaningful statistic) projection error of the different track links as
    a function of distance from the reference frame (triangulation frame)
    """
    pass


def proj_error_ba():
    """
    Median (or any other meaningful statistic) projection error of the different track links as
    a function of distance from the reference frame (1st frame for Bundle)
    """
    pass


def factor_error_pnp():
    """
    Median (or any other meaningful statistic) factor error of the different track links as a
    function of distance from the reference frame.
    """
    pass


def factor_error_ba():
    """
    Same as above for Bundle Adjustment.
    """
    pass


def abs_pnp_est_error():
    """
    Absolute PnP estimation error: X axis error, Y axis error, Z axis error,
    Total error norm, Angle error.
    """
    pass


def abs_pose_est_error(loop=False):
    """
    Absolute Pose Graph estimation error: X axis error, Y axis error, Z axis error,
    Total error norm, Angle error (With or Without loop closure).
    """
    pass


def rel_pnp_est_error():
    """
    The error of the relative pose estimation compared to the ground truth relative pose,
    evaluated on sequence lengths of (100, 300, 800).
    o X axis, Y axis, Z axis, Total error norm (measure as error%: m/m)
    o Angle error (measure as deg/m)
    o For each graph calculate the average error of all the sequences for total norm
    and angle error (a single number for each).
    """
    pass


def rel_bundle_est_error():
    """
    Same as above for Bundle Adjustment.
    """
    pass


def uncertainty_vs_kf(loop=False):
    """
    Uncertainty size vs keyframe – pose graph without loop closure:
    o Location Uncertainty
    o Angle Uncertainty
    """
    # How did you measure uncertainty size?
    # How did you isolate the different parts of the uncertainty?
    pass


def make_plots():
    """
    Make plots needed for final project submission.
    """
    proj_error_pnp()
    proj_error_ba()
    factor_error_pnp()
    factor_error_ba()
    abs_pnp_est_error()
    abs_pose_est_error()
    rel_pnp_est_error()
    rel_bundle_est_error()
    # Number of matches per successful loop closure frame
    # Inlier percentage per successful loop closure frame
    uncertainty_vs_kf()

def main():
    make_plots()


if __name__ == '__main__':
    main()
