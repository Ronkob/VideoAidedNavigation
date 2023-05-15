import os.path
import cv2
import random

DATA_PATH = os.path.join('..', '..', 'dataset', 'sequences', '05')
FIRST_PAIR_IDX = 1
NUM_MATCHES = 20
RATIO = 0.8
N_FEATURES = 550


def read_images(idx):
    """
    Read the stereo pair (both right and left images) of the given index.
    :param idx: index of pair to read.
    :return: Two Grayscale Images after cv.imread.
    """
    img_name = '{:06d}.png'.format(idx)
    # print(os.path.join(os.path.dirname(__file__), DATA_PATH)+'\\image_1\\'+img_name)
    # print(os.path.exists(os.path.join(os.path.dirname(__file__), DATA_PATH)))

    left_image = cv2.imread(os.path.join(DATA_PATH,'image_0',img_name), cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(os.path.join(DATA_PATH,'image_1',img_name), cv2.IMREAD_GRAYSCALE)

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


def present_kp_locations(left_image, kp1, right_image, kp2):
    """
    Present the key-points pixel locations on both images.
    :param left_image: Image1 to present key-points on.
    :param kp1: Key-points detected on left_image.
    :param right_image: Image2 to present key-points on.
    :param kp2: Key-points detected on right_image.
    """
    output_image1 = cv2.drawKeypoints(left_image, kp1, 0, (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    output_image2 = cv2.drawKeypoints(right_image, kp2, 0, (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("1.1 - Left image keypoints", output_image1)
    cv2.imshow("1.1 - Right image keypoints", output_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_descriptors(desc1, desc2):
    """
    Print the descriptors of the two first features.
    :param desc1: First descriptor of Image1.
    :param desc2: First descriptor of Image2.
    """
    print("section 1.2:\n")
    print("Left image first two feature's descriptor:\n", desc1[0:2])
    print("\nRight image first two feature's descriptor:\n", desc2[0:2])


def match(desc1, desc2):
    """
    Match the two descriptors list.
    :param desc1: List of descriptors from Image1.
    :param desc2: List of descriptors from Image2.
    :return: List of matches found.
    """
    brute_force = cv2.BFMatcher()
    matches = brute_force.knnMatch(desc1, desc2, k=2)
    return matches


def present_matches(left_image, kp1, right_image, kp2, matches, window_text):
    """
    Present num random matches as lines connecting the key-point pixel
     location on the images pair.
    :param left_image: Image1 to present matches on.
    :param kp1: Key-points detected on left_image.
    :param right_image: Image2 to present matches on.
    :param kp2: Key-points detected on right_image.
    :param matches: Matches found between the given images.
    """
    output_image = cv2.drawMatchesKnn(left_image, kp1, right_image, kp2, matches,
                                      None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                                      matchColor=(0, 255, 255), matchesMask=[[1, 0] for _ in matches])
    cv2.imshow(window_text, output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def present_failed_sig_test(left_image, left_image_kp, right_image, right_image_kp,
                            rejected_matches, window_text):
    """
    present the best match that was rejected
    """
    best_match = sorted(rejected_matches, key=lambda x: x[0].distance / x[1].distance)[0][0]
    left_kp = [left_image_kp[best_match.queryIdx]]
    right_kp = [right_image_kp[best_match.trainIdx]]
    left_point = cv2.drawKeypoints(left_image, left_kp, None, color=(0, 255, 0))
    right_point = cv2.drawKeypoints(right_image, right_kp, None, color=(0, 255, 0))
    cv2.imshow(window_text + " - left", left_point)
    cv2.imshow(window_text + " - right", right_point)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ex1_run(algorithm, left_image, right_image):
    # Section 1.1
    left_image_kp, left_image_desc, right_image_kp, right_image_desc = \
        detect_and_extract(algorithm, left_image, right_image)
    present_kp_locations(left_image, left_image_kp, right_image, right_image_kp)

    # Section 1.2
    print_descriptors(left_image_desc, right_image_desc)

    # Section 1.3
    matches = match(left_image_desc, right_image_desc)
    sampled_matches = random.choices(matches, k=NUM_MATCHES)
    present_matches(left_image, left_image_kp, right_image, right_image_kp, sampled_matches,
                    window_text=f"1.3 - Random {NUM_MATCHES} matches")

    # Section 1.4
    accepted_matches, rejected_matches = significance_test(matches, RATIO)
    sampled_matches = random.choices(accepted_matches, k=NUM_MATCHES)
    present_matches(left_image, left_image_kp, right_image, right_image_kp,
                    sampled_matches,
                    window_text=f"1.4.1 - Random {NUM_MATCHES} accepted matches")
    print(f"For a ratio of {RATIO} used, we got "
          f"{len(rejected_matches)}/{len(matches)} rejected matches")
    present_failed_sig_test(left_image, left_image_kp,
                            right_image, right_image_kp, rejected_matches,
                            window_text=f'1.4.2 - A correct match that didn\'t pass the significance test')


def main():
    random.seed(0)
    left_image, right_image = read_images(FIRST_PAIR_IDX)
    algorithm = cv2.SIFT_create(nfeatures=N_FEATURES)
    ex1_run(algorithm, left_image, right_image)


if __name__ == '__main__':
    main()
