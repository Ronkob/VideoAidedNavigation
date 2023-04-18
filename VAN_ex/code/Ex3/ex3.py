import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import VAN_ex.utils as utils

DATA_PATH = os.path.join('..', '..', 'dataset', 'sequences', '05')
N_FEATURES = 500


def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def create_point_cloud(idx):
    left_image, right_image = utils.read_images(idx)
    matches, left_image_kp, right_image_kp, left_image, right_image = utils.get_matches(left_image, right_image)
    left_inliers, left_outliers, right_inliers, right_outliers = \
        utils.reject_matches_pattern(matches, left_image_kp, right_image_kp)
    k, m1, m2 = utils.read_cameras()
    p3d = utils.triangulate_points(k @ m1, k @ m2, left_inliers, right_inliers)
    return p3d


def run_ex3():
    # Section 3.1 - Create two point clouds - for pair0 and pair1
    cloud0 = create_point_cloud(0)
    cloud1 = create_point_cloud(1)

    # Section 3.2 - Match features between the two left images (left0 and left1)
    left0_image, right0_image = utils.read_images(0)
    matches0, left0_image_kp, right0_image_kp = utils.get_matches(left0_image, right0_image)
    left0_inliers, right0_inliers = utils.reject_matches_pattern(matches0, left0_image_kp, right0_image_kp)

    left1_image, right1_image = utils.read_images(1)
    matches1, left1_image_kp, right1_image_kp, left1_image, right1_image = utils.get_matches(left1_image, right1_image)
    left1_inliers, right1_inliers = utils.reject_matches_pattern(matches1, left1_image_kp, right1_image_kp)

    left_matches = utils.get_matches(left0_image, right1_image)

    # Section 3.3 - Choose 4 key-points that were matched on all four images. This means we have both 3D
    # locations from the triangulation of pair 0 and matching pixel locations on pair 1. Apply the PNP
    # algorithm between the point cloud and the matching pixel locations on left1 to calculate the
    # extrinsic camera matrix [𝑅|𝑡] of left1, with 𝑅 a 3 × 3 rotation matrix and 𝑡 a 3 × 1 translation
    # vector.



def main():
    run_ex3()


if __name__ == '__main__':
    main()
