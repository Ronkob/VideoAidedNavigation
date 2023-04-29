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
    left_inliers, right_inliers = utils.reject_matches_pattern(matches, left_image_kp, right_image_kp)
    k, m1, m2 = ex2_utils.read_cameras()
    p3d = utils.triangulate_points(k @ m1, k @ m2, left_inliers, right_inliers)
    return p3d


def calculate_pnp(point_cloud, left1_matching_loc, k):
    """
    Calculate the PnP algorithm.
    """
    _, rotation_vector, translation_vector = cv2.solvePnP(point_cloud, left1_matching_loc, k, None,
                                                          flags=cv2.SOLVEPNP_AP3P)

    if rotation_vector is None:
        print("aha! spotted a hidden outlier")
        return None
    ex_cam_mat = rodriguez_to_mat(rotation_vector, translation_vector)
    return ex_cam_mat


def calc_relative_camera_pos(ext_camera_mat):
    """
    Returns the relative position of a camera according to its extrinsic
     matrix.
    """
    R, t = ext_camera_mat[:, :3], ext_camera_mat[:, 3]
    return -t


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
    return sum_of_squared_diffs <= DIST_THRESH ** 2


def recognize_supporters(pair0_p3d, left1_mat, left1_inliers, right1_mat, right1_inliers):
    """
    We consider a point x that projects close to the matched pixel locations in
     all four images a supporter of transformation 𝑇. We use a distance
     threshold, recognize the supporters and return their indexes.
    """
    left1_proj = project_and_measure(pair0_p3d, left1_mat, left1_inliers)
    right1_proj = project_and_measure(pair0_p3d, right1_mat, right1_inliers)
    supporters = left1_proj & right1_proj
    return np.where(supporters == True)[0]


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
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-10, 20)
    plt.show()


def plot_matches_and_supporters(left0, left1, left0_inliers, left1_inliers, supporters_idx):
    """
    Plot on images left0 and left1 the matches, with supporters in
    different color.
    """
    fig = plt.figure()
    fig.suptitle(f"Left0 and Left1 matches & supporters \n"
                 f"Number of supporters: {len(supporters_idx)} {len(supporters_idx) / len(left0_inliers) * 100:.2f}%")

    fig.add_subplot(2, 1, 1)
    plt.imshow(left0, cmap='gray')
    plt.title("Left0")
    plt.scatter(left0_inliers[:, 0], left0_inliers[:, 1], s=3, color='blue', label='inliers')
    plt.scatter(left0_inliers[supporters_idx][:, 0], left0_inliers[supporters_idx][:, 1], s=3, color='cyan',
                label='supporters')
    plt.legend()
    plt.tight_layout(h_pad=3, pad=2.5)
    fig.add_subplot(2, 1, 2)
    plt.imshow(left1, cmap='gray')
    plt.title("Left1")
    plt.scatter(left1_inliers[:, 0], left1_inliers[:, 1], s=3, color='blue')
    plt.scatter(left1_inliers[supporters_idx][:, 0], left1_inliers[supporters_idx][:, 1], s=3, color='cyan')

    plt.show()


def get_pair_matches(left_image, right_image):
    """
    Calculates all matches and other needed values for a given pair index.
    """
    matches, left_image_kp, right_image_kp = utils.get_matches(left_image, right_image)
    inliers_idxs = utils.stereo_reject(matches, left_image_kp, right_image_kp)
    pair0_dict = {matches[idx][0].queryIdx: idx for idx in inliers_idxs}
    return left_image_kp, matches, pair0_dict, right_image_kp


def plot_relative_camera_positions(left1_ext_mat, m1, m2):
    left0_cam, right0_cam = calc_relative_camera_pos(m1), calc_relative_camera_pos(m2)
    left1_cam = calc_relative_camera_pos(left1_ext_mat)
    right1_ext_mat = m2 @ np.vstack((left1_ext_mat, [np.array([0, 0, 0, 1])]))
    right1_cam = calc_relative_camera_pos(right1_ext_mat)  # left0 -> right0 -> right1
    plot_relative_pos(left0_cam, right0_cam, left1_cam, right1_cam)


def find_matching_features_successive(left0_image, right0_image, left1_image, right1_image):
    # Section 3.2 - Match features between the two left images (left0 and left1)
    left0_image_kp, matches0, pair0_dict, right0_image_kp = get_pair_matches(left0_image, right0_image)
    left1_image_kp, matches1, pair1_dict, right1_image_kp = get_pair_matches(left1_image, right1_image)
    left_successive_matches, _, _ = utils.get_matches(left0_image, left1_image)

    # Section 3.3 - calculate the extrinsic camera matrix [R|t] of left1.
    pairs0, pairs1 = find_matching_kps(left_successive_matches, pair0_dict, pair1_dict)
    pair0_matches, pair1_matches = matches0[pairs0], matches1[pairs1]
    left0_inliers, right0_inliers = utils.matches_to_pts(pair0_matches, left0_image_kp, right0_image_kp)
    left1_inliers, right1_inliers = utils.matches_to_pts(pair1_matches, left1_image_kp, right1_image_kp)
    pictures = (left0_image, left1_image)
    inliers = (left0_inliers, right0_inliers, left1_inliers, right1_inliers)
    return pictures, inliers


def calc_ext_mat_from_sample_idxs(sample_idxs, left1_inliers, pair0_p3d, k):
    """ find pnp from 4 sample points """
    pnp_3d_pts = pair0_p3d[sample_idxs]  # Choose 4 key-points
    pnp_left1_pts = left1_inliers[sample_idxs]  # Choose matching 3D locations
    return calculate_pnp(pnp_3d_pts, pnp_left1_pts, k)  # Estimate the extrinsic camera matrix [R|t] of left1.


def calculate_T_and_right_T(left1_ext_mat, k, m2):
    right1_ext_mat = m2 @ np.vstack((left1_ext_mat, [np.array([0, 0, 0, 1])]))
    return k @ left1_ext_mat, k @ right1_ext_mat


def ransac_pnp(pair0_p3d, left1_inliers, right1_inliers, k, m2):
    """
    calculates findpnp with ransac
    :param pair0_p3d: first pair points in 3d space in poir0 coordinate space
    :param left1_mat: left pair1 camera extrinsic matrix
    :param left1_inliers: left pair1 inliers
    :param right1_mat: right pair1 camera extrinsic matrix
    :param right1_inliers: right pair1 inliers
    :return: best_T, supporters_idx
    """
    best_num_supporters = 0
    best_idx = np.zeros(PNP_POINTS)
    best_ext_mat = None
    best_supporters = None

    for _ in range(300):
        # random sample 4 points
        random_idx = np.random.choice(len(pair0_p3d), size=PNP_POINTS, replace=False)
        left1_ext_mat = calc_ext_mat_from_sample_idxs(random_idx, left1_inliers, pair0_p3d, k)  # find pnp
        if left1_ext_mat is None:
            continue
        T, right_T = calculate_T_and_right_T(left1_ext_mat, k, m2)
        supporters_idx = recognize_supporters(pair0_p3d, T, left1_inliers, right_T, right1_inliers)

        if len(supporters_idx) > best_num_supporters:
            best_idx = random_idx
            best_num_supporters = len(supporters_idx)
            best_ext_mat = left1_ext_mat
            best_supporters = supporters_idx

    return best_ext_mat, best_supporters


def track_movement_successive(idxs):
    """sections 3.2 - 3.5"""

    # section 3.2 - match features between two left images
    left0_image, right0_image = ex1_utils.read_images(idxs[0])
    left1_image, right1_image = ex1_utils.read_images(idxs[1])
    photos, inliers = find_matching_features_successive(left0_image, right0_image, left1_image, right1_image)
    left0_inliers, right0_inliers, left1_inliers, right1_inliers = inliers

    # section 3.3.1 - find the transformation 𝑇 that transforms from left0 coordinates to left1 coordinates.
    k, m1, m2 = ex2_utils.read_cameras()
    pair0_p3d = utils.triangulate_points(k @ m1, k @ m2, left0_inliers, right0_inliers)
    pair1_p3d = utils.triangulate_points(k @ m1, k @ m2, left1_inliers, right1_inliers)

    left1_ext_mat = calc_ext_mat_from_sample_idxs(np.arange(PNP_POINTS), left1_inliers, pair0_p3d, k)

    # section 3.3.2 - plot the relative position of the four cameras
    plot_relative_camera_positions(left1_ext_mat, m1, m2)

    # Section 3.4 - Recognize supporters for the ext_mat
    T, right_T = calculate_T_and_right_T(left1_ext_mat, k, m2)
    supporters_idx = recognize_supporters(pair0_p3d, T, left1_inliers, right_T, right1_inliers)
    plot_matches_and_supporters(left0_image, left1_image, left0_inliers, left1_inliers, supporters_idx)

    # Section 3.5 - Use a RANSAC framework, with PNP as the inner model,
    # to find the 4 points that maximize the number of supporters.
    best_ext_mat, supporters_idx = ransac_pnp(pair0_p3d, left1_inliers, right1_inliers, k, m2)
    plot_matches_and_supporters(left0_image, left1_image, left0_inliers, left1_inliers, supporters_idx)

    utils.display_2_point_clouds(pair0_p3d, pair1_p3d, "second clouds")
    proj_no_ransac = pair0_p3d @ left1_ext_mat
    proj_ransac = pair0_p3d @ best_ext_mat
    utils.display_point_cloud(proj_no_ransac, pair1_p3d,
                              ["third cloud", "transformed pair 0 points, no ransac", "pair 1 points"])
    utils.display_point_cloud(proj_ransac, pair1_p3d,
                              ["forth cloud", "transformed pair 0 points, ransac", "pair 1 points"])


def track_movement_all_movie():
    pass


def run_ex3():
    np.random.seed(2)
    """
    Runs all exercise 3 sections.
    """

    idxs = [0, 10]
    # Section 3.1 - Create two point clouds - for pair0 and pair1
    cloud0 = create_point_cloud(idxs[0])
    cloud1 = create_point_cloud(idxs[1])
    utils.display_2_point_clouds(cloud0, cloud1, "first clouds")

    # sections 3.2 - 3.5
    track_movement_successive(idxs)

    # Section 3.6 - Repeat steps 2.1-2.5 for the whole movie for all the images.
    track_movement_all_movie()


def main():
    run_ex3()


if __name__ == '__main__':
    main()
