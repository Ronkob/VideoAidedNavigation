import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import VAN_ex.code.Ex1.ex1 as ex1_utils

DATA_PATH = os.path.join('..', '..', 'dataset', 'sequences', '05')
FIRST_PAIR_IDX = 1
N_FEATURES = 500
BINS = 50
DIFF = 1


def calculate_deviations(matches, left_image_kp, right_image_kp):
    """
    Find all deviations from the special pattern of correct matches
     for all the matches.
    """
    dev_lst = []
    for match in matches:
        img1_idx, img2_idx = match[0].queryIdx, match[0].trainIdx
        x1, y1 = left_image_kp[img1_idx].pt
        x2, y2 = right_image_kp[img2_idx].pt
        dev_lst.append(abs(y2 - y1))
    return dev_lst


def create_dev_hist(matches, left_image_kp, right_image_kp):
    """
    Create a histogram of the deviations from this pattern for all the matches,
    and print the percentage of matches that deviate by more than 2 pixels.
    """
    deviations = calculate_deviations(matches, left_image_kp, right_image_kp)

    plt.title('Deviations From Pattern')
    plt.ylabel('Number of matches')
    plt.xlabel('Deviation from rectified stereo pattern')
    plt.hist(deviations, bins=BINS)
    plt.show()

    deviated = sum(map(lambda x: x > 2, deviations))
    print("Percentage of matches that deviate by more than 2 pixels is {}%".
          format(deviated / len(matches) * 100))


def use_rs_pattern(left_image, right_image, matches, left_image_kp, right_image_kp):
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
        if abs(y2 - y1) > DIFF:
            left_outliers.append(left_image_kp[img1_idx].pt)
            right_outliers.append(right_image_kp[img2_idx].pt)
        else:
            left_inliers.append(left_image_kp[img1_idx].pt)
            right_inliers.append(right_image_kp[img2_idx].pt)
    print("Number of matches that were discarded is {}".format(len(left_outliers)))

    present_matches(left_image, left_inliers, left_outliers, "Left Image")
    present_matches(right_image, right_inliers, right_outliers, "Right Image")

    return np.array(left_inliers), np.array(right_inliers)


def present_matches(image, inliers, outliers, text):
    # Present all matches on both images
    plt.imshow(image, cmap='gray')
    plt.title(text)
    plt.scatter([i[0] for i in inliers], [i[1] for i in inliers], s=1,
                color='orange')
    # plt.scatter([i[0] for i in outliers], [i[1] for i in outliers], s=1,
    #             color='cyan')
    plt.show()


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


def display_and_compare(p3d, cv_p3d):
    """
    - Present a 3D plot of the calculated 3D points of our triangulation.
    - Display the point cloud obtained from opencv and
    - Compare the results: print the median distance between the corresponding
      3d points.
    """
    # Main figure
    rows, cols = 1, 2
    elev, azim = 10, 10

    fig = plt.figure()
    fig.suptitle(f"Our vs. Open-CV triangulation\n"
                 f"Median distance between corresponding points = "
                 f"{np.median(np.linalg.norm(p3d - cv_p3d, axis=1))}\n")

    # our triangulation
    axes = fig.add_subplot(rows, cols, 1, projection='3d')
    axes.set_title("Our Triangulation")
    axes.scatter3D(0, 0, 0, c='red', s=60, marker='^')  # Camera
    axes.scatter3D(p3d[:, 0], p3d[:, 1], p3d[:, 2])

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_xlim3d(-20, 10)
    axes.set_ylim3d(10, -20)
    axes.set_zlim3d(600, 0)
    axes.view_init(elev=elev, azim=azim, vertical_axis='y')

    # cv triangulation
    axes = fig.add_subplot(rows, cols, 2, projection='3d')
    axes.set_title("CV Triangulation")
    axes.scatter(cv_p3d[:, 0], cv_p3d[:, 1], cv_p3d[:, 2])
    axes.scatter3D(0, 0, 0, c='red', s=60, marker='^')  # Camera

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_xlim3d(-20, 10)
    axes.set_ylim3d(10, -20)
    axes.set_zlim3d(600, 0)
    axes.view_init(elev=elev, azim=azim, vertical_axis='y')

    plt.show()


def run_few_images(num_images):
    """
    Run this process (matching and triangulation) over a few pairs of images.
    """
    for idx in np.arange(FIRST_PAIR_IDX + 1, FIRST_PAIR_IDX + num_images * 20, 20):
        left_image, right_image = ex1_utils.read_images(idx)
        algorithm = cv2.SIFT_create(nfeatures=N_FEATURES)
        left_image_kp, left_image_desc, right_image_kp, right_image_desc = \
            ex1_utils.detect_and_extract(algorithm, left_image, right_image)
        matches = ex1_utils.match(left_image_desc, right_image_desc)
        run_ex2(left_image, right_image, left_image_kp, right_image_kp, matches)


def run_ex2(left_image, right_image, left_image_kp, right_image_kp, matches):
    # Section 2.1
    create_dev_hist(matches, left_image_kp, right_image_kp)

    # Section 2.2
    left_inliers, right_inliers = \
        use_rs_pattern(left_image, right_image, matches, left_image_kp, right_image_kp)

    # Section 2.3
    k, m1, m2 = read_cameras()
    p3d = triangulate_points(k @ m1, k @ m2, left_inliers, right_inliers)
    cv_p4d = cv2.triangulatePoints(
        k @ m1, k @ m2, left_inliers.T, right_inliers.T).T
    cv_p3d = np.squeeze(cv2.convertPointsFromHomogeneous(cv_p4d))
    display_and_compare(p3d, cv_p3d)


def main():
    """
    :return:
    """
    # single run on first image
    run_few_images(1)

    # consecutive run over a few images
    run_few_images(5)


if __name__ == '__main__':
    main()
