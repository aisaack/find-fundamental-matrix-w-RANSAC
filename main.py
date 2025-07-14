import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_images, filter_good_points, get_homogenous, verify_matrix_equivalence, compute_iter_number
from ransac import Ransac


class ORB:
    def __init__(self, n_feat):
        self.orb = cv2.ORB_create(n_feat)

    def kp_det(self, im):
        return self.orb.detect(im)

    def get_desc(self, im, kps):
        return self.orb.compute(im, kps)[1]


class BFMatcher:
    def __init__(self, mode='hamming1', cross_check=True):
        self.mode = dict(
                l1=cv2.NORM_L1,
                l2=cv2.NORM_L2,
                hamming1=cv2.NORM_HAMMING,
                hamming2=cv2.NORM_HAMMING2)

        self.bf = cv2.BFMatcher(self.mode[mode], crossCheck=cross_check)

    def match(self, desc1, desc2):
        return self.bf.match(desc1, desc2)

    def knn_match(self, desc1, desc2, k=2):
        return self.bf.knnMatch(desc1, desc2, k)


def run():
    # load images
    im1, im2 = load_images('./imgs')
    num_feats = 1000

    # initialize feature detector, matcher and RANSAC
    det = ORB(num_feats)
    mat = BFMatcher('hamming1', cross_check=False)
    threshold = 1.0
    sample_size = 8
    max_iter = compute_iter_number(sample_size, p=0.99, e=0.1)
    print(f'Number of iteration: {max_iter}\n')
    opt = Ransac(num_points=sample_size, threshold=threshold, max_iter=max_iter)

    kp1 = det.kp_det(im1)
    kp2 = det.kp_det(im2)
    desc1 = det.get_desc(im1, kp1)
    desc2 = det.get_desc(im2, kp2)
    
    # RANSAC finds optimal value in iterative fashion
    # It'd reduce runtime if the points are filtered
    # out before running RANSAC
    matches = mat.knn_match(desc1, desc2)

    pts1, pts2 = filter_good_points(matches, kp1, kp2, r=0.75)      # minimum number of feature points should be greater than 150.
    print(f'{len(pts1)} number of key points remained')

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    myF, _ = opt.run(pts1, pts2)
    cv2F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC)
    
    verify_matrix_equivalence(myF, cv2F, get_homogenous(pts1), get_homogenous(pts2))
    print()
    print(myF)
    print()
    print(cv2F)


if __name__ == '__main__':
    run()

