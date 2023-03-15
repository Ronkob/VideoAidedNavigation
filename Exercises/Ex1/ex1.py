import cv2
import matplotlib.pyplot as plt
import random


DATA_PATH = r'..\..\VAN_ex\dataset\sequences\05\\'
PAIR = 1
NUM_MATCHES = 20


def read_images(idx):
    """
    Read the stereo pair (both right and left images) of the given index.
    :param idx: index of pair to read.
    :return: Two Images after cv.imread.
    """
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + 'image_0\\' + img_name, 0)
    img2 = cv2.imread(DATA_PATH + 'image_1\\' + img_name, 0)
    return img1, img2


def detect_and_extract(algorithm, img1, img2):
    """
    Detect and extract key-points from the given stereo pairs, and calculate
     the feature-descriptors for each key-point in both key-points lists.
    :param algorithm:
    :param img1: Image1 object.
    :param img2: Image2 object.
    :return: Key-points and descriptors of both images.
    """
    kp1, desc1 = algorithm.detectAndCompute(img1, mask=None)
    kp2, desc2 = algorithm.detectAndCompute(img2, mask=None)
    return kp1, desc1, kp2, desc2


def present_kp_locations(img1, kp1, img2, kp2):
    """
    Present the key-points pixel locations on both images.
    :param img1: Image1 to present key-points on.
    :param kp1: Key-points detected on img1.
    :param img2: Image2 to present key-points on.
    :param kp2: Key-points detected on img2.
    """
    output_image1 = cv2.drawKeypoints(img1, kp1, 0, (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    output_image2 = cv2.drawKeypoints(img2, kp2, 0, (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(output_image1)
    plt.show()
    plt.imshow(output_image2)
    plt.show()


def print_descriptors(desc1, desc2):
    """
    Print the descriptors of the two first features.
    :param desc1: First descriptor of Image1.
    :param desc2: First descriptor of Image2.
    """
    print("Image 1 first feature's descriptor:\n", desc1)
    print("\nImage 2 first feature's descriptor:\n", desc2)


def match(desc1, desc2):
    """
    Match the two descriptors list.
    :param desc1: List of descriptors from Image1.
    :param desc2: List of descriptors from Image2.
    :return: List of matches found.
    """
    brute_force = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = brute_force.match(desc1, desc2)
    return matches


def present_matches(img1, kp1, img2, kp2, matches):
    """
    Present num random matches as lines connecting the key-point pixel
     location on the images pair.
    :param img1: Image1 to present matches on.
    :param kp1: Key-points detected on img1.
    :param img2: Image2 to present matches on.
    :param kp2: Key-points detected on img2.
    :param matches: Matches found between the given images.
    """
    output_image = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2, matchColor=(0, 255, 0))
    plt.imshow(output_image)
    plt.show()


def significance_test():
    """
    Use significance test to reject matches
    • generate an output with 20 of the resulting matches (as in sections 1.3).
    • What ratio value did you use?
    • How many matches were discarded?
    • Present a correct match (as a dot on each image) that failed the significance test. (If you
    cannot find such match, strengthen the significance test (how?) until you find one.)
    :return:
    """
    pass


def main():
    img1, img2 = read_images(PAIR)
    algorithm = cv2.KAZE_create()

    img1_kp, img1_desc, img2_kp, img2_desc = \
        detect_and_extract(algorithm, img1, img2)
    # present_kp_locations(img1, img1_kp, img2, img2_kp)
    # print_descriptors(img1_desc, img2_desc)

    matches = match(img1_desc, img2_desc)
    selected_matches = random.choices(matches, k=NUM_MATCHES)
    present_matches(img1, img1_kp, img2, img2_kp, selected_matches)


if __name__ == '__main__':
    main()