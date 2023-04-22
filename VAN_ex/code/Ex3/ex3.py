import os
import cv2
import numpy as np
import VAN_ex.code.utils as utils
import matplotlib.pyplot as plt
import VAN_ex.code.Ex1.ex1 as ex1_utils
import VAN_ex.code.Ex2.ex2 as ex2_utils

N_FEATURES = 500
PNP_POINTS = 4
DIST_THRESH = 2


def rodriguez_to_mat(rvec, tvec):
    """
    Uses a function in OpenCV that converts a rotation vector to a rotation
     matrix. The function takes a 3 × 1 rotation vector as input and returns
      a 3 × 3 rotation matrix.
    """
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def find_matching_kps(left_matches, matches0, matches1):
    """
    Receives matches between pair0, pair1 and left0_left1, and returns all
    matches of pair0 and pair1 which have a match in the other pair.
    """
    pairs0, pairs1 = list(), list()
    for match in left_matches:
        left0_kp_idx, left1_kp_idx = match[0].queryIdx, match[0].trainIdx
        if left0_kp_idx in matches0 and left1_kp_idx in matches1:
            pairs0.append(matches0[left0_kp_idx])
            pairs1.append(matches1[left1_kp_idx])
    return np.array(pairs0), np.array(pairs1)


def create_point_cloud(idx):
    """
    Create a point cloud for the next stereo pair.
    """
    left_image, right_image = ex1_utils.read_images(idx)
    matches, left_image_kp, right_image_kp = utils.get_matches(left_image, right_image)
    left_inliers, right_inliers = \
        utils.reject_matches_pattern(matches, left_image_kp, right_image_kp)
    k, m1, m2 = ex2_utils.read_cameras()
    p3d = utils.triangulate_points(k @ m1, k @ m2, left_inliers, right_inliers)
    return p3d


def calculate_pnp(point_cloud, left1_matching_loc, k):
    """
    Calculate the PnP algorithm.
    """
    _, rotation_vector, translation_vector = cv2.solvePnP(
        point_cloud, left1_matching_loc, k, None, flags=cv2.SOLVEPNP_AP3P)
    ex_cam_mat = rodriguez_to_mat(rotation_vector, translation_vector)
    return ex_cam_mat


def calc_relative_camera_pos(ext_camera_mat):
    """
    Returns the relative position of a camera according to its extrinsic
     matrix.
    """
    R, t = ext_camera_mat[:, :3], ext_camera_mat[:, 3]
    return -t


def recognize_supporters(pair0_p3d,
                         left1_mat, left1_inliers,
                         right1_mat, right1_inliers):
    """
    We consider a point x that projects close to the matched pixel locations in
     all four images a supporter of transformation 𝑇. We use a distance
     threshold, recognize the supporters and return their indexes.
    """
    left1_proj = project_and_measure(pair0_p3d, left1_mat, left1_inliers)
    right1_proj = project_and_measure(pair0_p3d, right1_mat, right1_inliers)
    supporters = left1_proj & right1_proj
    return np.where(supporters == 1)[0]


def project_and_measure(p3d_pts, camera_mat, inliers):
    """
    Projects the given p3d points using the camera matrix in order
    to recognize the supporters. Return whether each inlier is within the range
    of the distance threshold.
    """
    R, t = camera_mat[:, :3], camera_mat[:, 3]
    proj = p3d_pts @ R.T + t
    proj = proj[:, :2] / proj[:, [2]]
    sum_of_squared_diffs = np.sum(np.square(proj - inliers), axis=1)
    return sum_of_squared_diffs <= DIST_THRESH**2


def plot_relative_pos(left0_camera, right0_camera, left1_camera, right1_camera):
    """
    Plot the relative position of the four cameras.
    """
    fig, ax = plt.subplots()
    ax.set_title('Relative position of the four cameras')
    ax.scatter(left0_camera[0], left0_camera[2], color='red', label='pair0')
    ax.scatter(right0_camera[0], right0_camera[2], color='red')
    ax.scatter(left1_camera[0], left1_camera[2], color='orange', label='pair1')
    ax.scatter(right1_camera[0], right1_camera[2], color='orange')
    plt.xlabel("x")
    plt.ylabel("z")
    ax.legend()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    plt.show()


def plot_matches_and_supporters(left0, left1, left0_inliers, left1_inliers, supporters_idx):
    """
    Plot on images left0 and left1 the matches, with supporters in
    different color.
    """
    fig = plt.figure()
    fig.suptitle('Left0 and Left1 matches & supporters')
    # externals = [10, 19, 39, 48, 51, 52, 53, 59]

    fig.add_subplot(2, 1, 1)
    plt.imshow(left0, cmap='gray')
    plt.title("Left0")
    plt.scatter(left0_inliers[:, 0], left0_inliers[:, 1],
                s=3, color='blue', label='inliers')
    # plt.scatter(left0_inliers[externals][:, 0],
    #             left0_inliers[externals][:, 1],
    #             s=3, color='pink', label='externals')
    plt.scatter(left0_inliers[supporters_idx][:, 0],
                left0_inliers[supporters_idx][:, 1],
                s=3, color='cyan', label='supporters')
    plt.legend()

    fig.add_subplot(2, 1, 2)
    plt.imshow(left1, cmap='gray')
    plt.title("Left1")
    plt.scatter(left1_inliers[:, 0], left1_inliers[:, 1], s=3, color='blue')
    # plt.scatter(left1_inliers[externals][:, 0],
    #             left1_inliers[externals][:, 1],
    #             s=3, color='pink')
    plt.scatter(left1_inliers[supporters_idx][:, 0],
                left1_inliers[supporters_idx][:, 1], s=3, color='cyan')

    plt.show()


def get_pair_matches(idx):
    """
    Calculates all matches and other needed values for a given pair index.
    """
    left_image, right_image = ex1_utils.read_images(idx)
    matches, left_image_kp, right_image_kp = utils.get_matches(left_image,
                                                                  right_image)
    inliers_idxs = utils.stereo_reject(matches, left_image_kp,
                                        right_image_kp)
    pair0_dict = {matches[idx][0].queryIdx: idx for idx in inliers_idxs}
    return left_image, left_image_kp, matches, pair0_dict, right_image_kp


def run_ex3():
    """
    Runs all exercise 3 sections.
    """
    # Section 3.1 - Create two point clouds - for pair0 and pair1
    # cloud0 = create_point_cloud(0)
    # cloud1 = create_point_cloud(1)

    # Section 3.2 - Match features between the two left images (left0 and left1)
    left0_image, left0_image_kp, matches0, pair0_dict, right0_image_kp = get_pair_matches(0)
    left1_image, left1_image_kp, matches1, pair1_dict, right1_image_kp = get_pair_matches(1)
    left_matches, _, _ = utils.get_matches(left0_image, left1_image)


    # Section 3.3 - Choose 4 key-points that were matched on all four images, apply the PnP
    # algorithm (between the point cloud and the matching pixel locations on left1) to calculate the extrinsic camera
    # matrix [R|t] of left1.
    k, m1, m2 = ex2_utils.read_cameras()
    pairs0, pairs1 = find_matching_kps(left_matches, pair0_dict, pair1_dict)
    pair0_matches, pair1_matches = matches0[pairs0], matches1[pairs1]
    left0_inliers, right0_inliers = utils.matches_to_pts(pair0_matches, left0_image_kp, right0_image_kp)
    left1_inliers, right1_inliers = utils.matches_to_pts(pair1_matches, left1_image_kp, right1_image_kp)

    pair0_p3d = utils.triangulate_points(k @ m1, k @ m2, left0_inliers, right0_inliers)
    pnp_3d_pts = pair0_p3d[np.array([i for i in range(PNP_POINTS)])]  # Choose 4 key-points
    pnp_left1_pts = left1_inliers[np.array([i for i in range(PNP_POINTS)])]  # Choose matching 3D locations
    left1_ext_mat = calculate_pnp(pnp_3d_pts, pnp_left1_pts, k)  # Estimate the extrinsic camera matrix [R|t] of left1.

    # Plot the relative position of the four cameras
    left0_cam, right0_cam = calc_relative_camera_pos(m1), calc_relative_camera_pos(m2)
    left1_cam = calc_relative_camera_pos(left1_ext_mat)
    right1_ext_mat = m2 @ np.vstack((left1_ext_mat, [np.array([0, 0, 0, 1])]))
    right1_cam = calc_relative_camera_pos(right1_ext_mat)   # left0 -> right0 -> right1
    plot_relative_pos(left0_cam, right0_cam, left1_cam, right1_cam)


    # Section 3.4 - Recognize supporters, plot on images left0 and left1 the
    # matches, with supporters in different color.
    left1_mat, right1_mat = k @ left1_ext_mat, k @ right1_ext_mat
    supporters_idx = recognize_supporters(pair0_p3d,
                                          left1_mat, left1_inliers,
                                          right1_mat, right1_inliers)
    plot_matches_and_supporters(left0_image, left1_image,
                                left0_inliers, left1_inliers,
                                supporters_idx)


    # Section 3.5 - Use a RANSAC framework, with PNP as the inner model,
    # to find the 4 points that maximize the number of supporters.
    # TODO - Implement


    # Section 3.6 - Repeat steps 2.1-2.5 for the whole movie for all the images.
    # TODO - Implement


def main():
    run_ex3()


if __name__ == '__main__':
    main()
