import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import VAN_ex.code.Ex1.ex1 as ex1_utils

DATA_PATH = os.path.join('..', '..', 'dataset', 'sequences', '05')
FIRST_PAIR_IDX = 1
N_FEATURES = 550
BINS = 50


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


    def read_cameras():
        with open(DATA_PATH+'calib.txt') as f:
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


def run_ex2(left_image_kp, right_image_kp, matches):
    # Section 2.1
    create_dev_hist(matches, left_image_kp, right_image_kp)


def main():
    left_image, right_image = ex1_utils.read_images(FIRST_PAIR_IDX)
    algorithm = cv2.SIFT_create(nfeatures=N_FEATURES)
    left_image_kp, left_image_desc, right_image_kp, right_image_desc =\
        ex1_utils.detect_and_extract(algorithm, left_image, right_image)
    matches = ex1_utils.match(left_image_desc, right_image_desc)
    run_ex2(left_image_kp, right_image_kp, matches)


if __name__ == '__main__':
    main()

