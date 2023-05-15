import os.path
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = os.path.join('..', 'dataset', 'sequences', '05')
N_FEATURES = 500
RATIO = 0.6
DIFF = 2
ALGORITHM = cv2.AKAZE_create()


# a decorator to measure the time of a function
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        # prints the time in minutes and seconds and to the 3rd digit after the dot
        print("Execution time: ", round((end_time - start_time) / 60, 0), " minutes and ",
              round((end_time - start_time) % 60, 3), " seconds")

        return ret

    return wrapper # returns the decorated function


def read_images(idx):
    """
    Read the stereo pair (both right and left images) of the given index.
    :param idx: index of pair to read.
    :return: Two Grayscale Images after cv.imread.
    """
    img_name = '{:06d}.png'.format(idx)
    # print(os.path.join(os.path.dirname(__file__), DATA_PATH) + '\\image_1\\' + img_name)
    # print(os.path.exists(os.path.join(os.path.dirname(__file__), DATA_PATH)))

    left_image = cv2.imread(DATA_PATH + '\\image_0\\' + img_name, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(DATA_PATH + '\\image_1\\' + img_name, cv2.IMREAD_GRAYSCALE)

    if left_image is None or right_image is None:
        raise RuntimeWarning("not a valid path, images are null")

    return left_image, right_image


def detect_and_extract(algorithm, left_image, right_image):
    """
    Detect and extract key-points from the given stereo pairs, and calculate
     the feature-descriptors for each key-point in both key-points lists.
    :param algorithm:
    :param left_image: Image1 object.
    :param right_image: Image2 object.
    :return: Key-points and descriptors of both images.
    """
    kp1, desc1 = algorithm.detectAndCompute(left_image, mask=None)
    kp2, desc2 = algorithm.detectAndCompute(right_image, mask=None)
    # print(f"Detected {len(kp1)} keypoints in left image, and {len(kp2)} "
    #       f"keypoints in right image")
    return kp1, desc1, kp2, desc2


def significance_test(matches, ratio):
    """
    Use significance test to reject matches. generate an output with 20 of the
     resulting matches and present a correct match that failed the
     significance test.
    """
    accepted_matches, rejected_matches = [], []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            accepted_matches.append([m, n])
        else:
            rejected_matches.append([m, n])
    return accepted_matches, rejected_matches


def match(desc1, desc2):
    """
    Match the two descriptors list.
    :param desc1: List of descriptors from Image1.
    :param desc2: List of descriptors from Image2.
    :return: List of matches found.
    """
    brute_force = cv2.BFMatcher()
    matches = brute_force.knnMatch(desc1, desc2, k=2)
    matches, _ = significance_test(matches, RATIO)
    return np.array(matches)


def get_matches(img1, img2):
    """
    Returns matches from 2 images.
    """
    algorithm = ALGORITHM
    left_image_kp, left_image_desc, right_image_kp, right_image_desc = detect_and_extract(algorithm, img1, img2)
    return match(left_image_desc, right_image_desc), left_image_kp, right_image_kp


def reject_matches_pattern(matches, left_image_kp, right_image_kp):
    """
    Use the rectified stereo pattern to reject matches, return all inliers points.
    """
    left_inliers, left_outliers = list(), list()
    right_inliers, right_outliers = list(), list()
    for i, match in enumerate(matches):
        img1_idx, img2_idx = match[0].queryIdx, match[0].trainIdx
        x1, y1 = left_image_kp[img1_idx].pt
        x2, y2 = right_image_kp[img2_idx].pt
        if np.abs(y2 - y1) > DIFF:
            left_outliers.append(left_image_kp[img1_idx].pt)
            right_outliers.append(right_image_kp[img2_idx].pt)
        else:
            left_inliers.append(left_image_kp[img1_idx].pt)
            right_inliers.append(right_image_kp[img2_idx].pt)

    return np.array(left_inliers), np.array(right_inliers)


def stereo_reject(matches, left_image_kp, right_image_kp):
    """
    Use the rectified stereo pattern to reject matches, return the indexes.
    """
    inliers_idx = list()
    for i, match in enumerate(matches):
        img1_idx, img2_idx = match[0].queryIdx, match[0].trainIdx
        x1, y1 = left_image_kp[img1_idx].pt
        x2, y2 = right_image_kp[img2_idx].pt
        if np.abs(y2 - y1) < DIFF:
            inliers_idx.append(i)
    return inliers_idx


def read_cameras():
    """
    Read the relative camera matrices of the stereo cameras from ‘calib.txt’.
    """
    with open(DATA_PATH + '\calib.txt') as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2


def least_squares_algorithm(P, Q, left_point, right_point):
    """
    Calculates the least square algorithm solution for P and Q matrices over
    two points.
    """
    p_x, p_y = left_point
    q_x, q_y = right_point
    A = np.array([P[2] * p_x - P[0], P[2] * p_y - P[1], Q[2] * q_x - Q[0], Q[2] * q_y - Q[1]])
    u, s, vt = np.linalg.svd(A)
    return vt[-1]


def triangulate_points(left_mat, right_mat, left_points, right_points):
    """
    Use the matches and the camera matrices to define and solve a linear least
     squares triangulation problem.
    :return: nparray with 3D points of our triangulation.
    """
    p3d_lst = []
    for i in range(len(right_points)):
        p4d = least_squares_algorithm(left_mat, right_mat, left_points[i], right_points[i])
        p3d = p4d[:3] / p4d[3]
        p3d_lst.append(p3d)
    return np.array(p3d_lst)


def matches_to_pts(matches, left_kps, right_kps):
    """
    Takes matches OpenCV objects and returns their pixels in the images.
    """
    left_inliers, right_inliers = list(), list()
    for match in matches:
        left_inliers.append(left_kps[match[0].queryIdx].pt)
        right_inliers.append(right_kps[match[0].trainIdx].pt)
    return np.array(left_inliers), np.array(right_inliers)


def display_point_cloud(first_cloud, second_claud, txt, elev=60, azim=10):
    """
    - Present a 3D plot of the calculated 3D points of our triangulation.
    - Display the point cloud obtained from opencv and
    - Compare the results: print the median distance between the corresponding
      3d points.
    """
    # Main figure
    rows, cols = 1, 1
    fig = plt.figure()
    fig.suptitle(txt[0])

    # first point cloud
    axes = fig.add_subplot(rows, cols, 1, projection='3d')
    axes.scatter3D(0, 0, 0, c='red', s=60, marker='^')  # Camera
    axes.scatter3D(first_cloud[:, 0], first_cloud[:, 1], first_cloud[:, 2], marker='^', alpha=0.5, color='orange',
                   label=txt[1])

    # second point cloud
    axes.scatter(second_claud[:, 0], second_claud[:, 1], second_claud[:, 2], marker='o', alpha=0.5, color='cyan',
                 label=txt[2])
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_xlim3d(-20, 20)
    axes.set_ylim3d(-20, 20)
    axes.set_zlim3d(-10, 300)
    axes.invert_yaxis()
    axes.invert_zaxis()
    axes.view_init(elev=elev, azim=azim, vertical_axis='y')
    # view legend in a meaningful location
    axes.legend(loc='upper left')
    # plt.legend()
    plt.show()


def display_2_point_clouds(first_cloud, second_claud, txt, elev=60, azim=10):
    """
    - Present a 3D plot of the calculated 3D points of our triangulation.
    - Display the point cloud obtained from opencv and
    - Compare the results: print the median distance between the corresponding
      3d points.
    """
    # Main figure
    rows, cols = 1, 2
    fig = plt.figure()
    fig.suptitle(txt)

    # our triangulation
    axes = fig.add_subplot(rows, cols, 1, projection='3d')
    axes.set_title("First Point Cloud")
    axes.scatter3D(0, 0, 0, c='red', s=60, marker='^')  # Camera
    axes.scatter3D(first_cloud[:, 0], first_cloud[:, 1], first_cloud[:, 2])

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_xlim3d(-20, 20)
    axes.set_ylim3d(-20, 20)
    axes.set_zlim3d(-10, 300)
    axes.invert_yaxis()
    axes.invert_zaxis()
    axes.view_init(elev=elev, azim=azim, vertical_axis='y')

    # cv triangulation
    axes = fig.add_subplot(rows, cols, 2, projection='3d')
    axes.set_title("Second Point Cloud")
    axes.scatter(second_claud[:, 0], second_claud[:, 1], second_claud[:, 2])
    axes.scatter3D(0, 0, 0, c='red', s=60, marker='^')  # Camera

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_xlim3d(-20, 20)
    axes.set_ylim3d(-20, 20)
    axes.set_zlim3d(-10, 300)
    axes.invert_yaxis()
    axes.invert_zaxis()
    axes.view_init(elev=elev, azim=azim, vertical_axis='y')

    plt.show()


def project(p3d_pts, camera_mat):
    """
    Projects the given p3d points using the camera matrix.
    """
    R, t = camera_mat[:, :3], camera_mat[:, 3]
    proj = p3d_pts @ R.T + t.T
    return proj[:, :2] / proj[:, [2]]
