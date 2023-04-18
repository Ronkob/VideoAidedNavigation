import os.path
import cv2
import numpy as np

DATA_PATH = os.path.join('..', '..', 'dataset', 'sequences', '05')
N_FEATURES = 500
RATIO = 0.6
DIFF = 2


def read_images(idx):
    """
    Read the stereo pair (both right and left images) of the given index.
    :param idx: index of pair to read.
    :return: Two Grayscale Images after cv.imread.
    """
    img_name = '{:06d}.png'.format(idx)
    print(os.path.join(os.path.dirname(__file__), DATA_PATH)+'\\image_1\\'+img_name)
    print(os.path.exists(os.path.join(os.path.dirname(__file__), DATA_PATH)))

    left_image = cv2.imread(DATA_PATH+'\\image_0\\'+img_name, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(DATA_PATH+'\\image_1\\'+img_name, cv2.IMREAD_GRAYSCALE)

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
    print(f"Detected {len(kp1)} keypoints in left image, and {len(kp2)} "
          f"keypoints in right image")
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
    return matches


def get_matches(img1, img2):
    """
    Returns matches from 2 images.
    """
    algorithm = cv2.SIFT_create(nfeatures=N_FEATURES)
    left_image_kp, left_image_desc, right_image_kp, right_image_desc = detect_and_extract(algorithm, img1, img2)
    return match(left_image_desc, right_image_desc), left_image_kp, right_image_kp


def reject_matches_pattern(matches, left_image_kp, right_image_kp):
    """
    Use the rectified stereo pattern to reject matches. Present all the
     resulting matches as dots on the image pair. Accepted matches (inliers)
     in orange and rejected matches (outliers) in cyan.
    """
    left_inliers, left_outliers = list(), list()
    right_inliers, right_outliers = list(), list()
    for match in matches:
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
    p_x, p_y = left_point
    q_x, q_y = right_point
    A = np.array([P[2] * p_x - P[0],
                  P[2] * p_y - P[1],
                  Q[2] * q_x - Q[0],
                  Q[2] * q_y - Q[1]])
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
        p4d = least_squares_algorithm(left_mat, right_mat,
                                      left_points[i], right_points[i])
        p3d = p4d[:3] / p4d[3]
        p3d_lst.append(p3d)
    return np.array(p3d_lst)
