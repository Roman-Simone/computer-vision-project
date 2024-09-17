import os
import cv2
import numpy as np
from utils import load_pickle, save_pickle, undistortedAndCrop
import copy

from matplotlib import pyplot as plt

def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def find_common_area(img1, img2, kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp image 1 to image 2 space
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    img1_warped = cv2.warpPerspective(img1, H, (w2, h2))

    # Compute the inverse homography to warp image 2 to image 1 space
    H_inv = np.linalg.inv(H)
    img2_warped = cv2.warpPerspective(img2, H_inv, (w1, h1))

    # Calculate intersection
    intersection1 = cv2.bitwise_and(img1_warped, img2)
    intersection2 = cv2.bitwise_and(img1, img2_warped)

    return intersection1, intersection2, H, H_inv, img1_warped, img2_warped

def sift_and_find_fundamental(img1, img2):
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Find common area using homography
    intersection1, intersection2, H, H_inv, img1_warped, img2_warped = find_common_area(img1, img2, kp1, kp2, good_matches)

    # Visualize the warped images and intersections
    plt.figure(figsize=(20, 10))
    plt.subplot(231), plt.imshow(img1, cmap='gray'), plt.title('Image 1')
    plt.subplot(232), plt.imshow(img1_warped, cmap='gray'), plt.title('Warped Image 1')
    plt.subplot(233), plt.imshow(intersection1, cmap='gray'), plt.title('Intersection 1')

    plt.subplot(234), plt.imshow(img2, cmap='gray'), plt.title('Image 2')
    plt.subplot(235), plt.imshow(img2_warped, cmap='gray'), plt.title('Warped Image 2')
    plt.subplot(236), plt.imshow(intersection2, cmap='gray'), plt.title('Intersection 2')

    # plt.show()

    # Find keypoints and descriptors in the intersection area
    kp1_inter, des1_inter = sift.detectAndCompute(img1_warped, None)
    kp2_inter, des2_inter = sift.detectAndCompute(img2_warped, None)

    # Match descriptors again
    matches_inter = flann.knnMatch(des1_inter, des2_inter, k=2)

    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches_inter):
        if m.distance < 0.7 * n.distance:
            pts2.append(kp2_inter[m.trainIdx].pt)
            pts1.append(kp1_inter[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # for point1, point2 in zip(pts1, pts2):
    #     img1 = copy.deepcopy(intersection1)
    #     img2 = copy.deepcopy(intersection2)
    #     #convert to BGR
    #     img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    #     img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    #     cv2.circle(img1, tuple(point1), 25, (0, 0, 255), -1)
    #     cv2.circle(img2, tuple(point2), 25, (0, 0, 255), -1)
    #     plt.figure(figsize=(30, 10))
    #     plt.subplot(121), plt.imshow(img1)
    #     plt.subplot(122), plt.imshow(img2)
    #     plt.show()

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1_warped, img2_warped, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 2, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2_warped, img1_warped, lines2, pts2, pts1)

    plt.figure(figsize=(30, 10))
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()

    return F, H, intersection1, intersection2

def calculate_F_matrix():

    current_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.join(current_path, os.pardir)
    parent_path = os.path.abspath(parent_path)

    calibration_file = os.path.join(parent_path, "data/calibrationMatrix/calibration.pkl")
    path_video_1 = os.path.join(parent_path, "data/dataset/video/out1.mp4")
    path_video_2 = os.path.join(parent_path, "data/dataset/video/out2.mp4")

    video_capture_1 = cv2.VideoCapture(path_video_1)
    video_capture_2 = cv2.VideoCapture(path_video_2)
    ret1, frame1 = video_capture_1.read()
    ret2, frame2 = video_capture_2.read()

    camera_infos = load_pickle(calibration_file)
    camera_number_1 = 1
    camera_number_2 = 2

    camera_info_1 = next((cam for cam in camera_infos if cam.camera_number == camera_number_1), None)
    camera_info_2 = next((cam for cam in camera_infos if cam.camera_number == camera_number_2), None)

    frame1, frame2 = undistortedAndCrop(frame1, frame2, camera_info_1, camera_info_2)

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    F, H = sift_and_find_fundamental(frame1, frame2)

    output_path = os.path.join(parent_path, "data/fundamentalMatrix/provaF.pkl")
    save_pickle(F, output_path)

    output_path = os.path.join(parent_path, "data/fundamentalMatrix/provaH.pkl")
    save_pickle(H, output_path)

if __name__ == '__main__':
    calculate_F_matrix()
