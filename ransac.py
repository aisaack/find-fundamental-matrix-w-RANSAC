import math
import cv2
import numpy as np
from utils import hartly_norm, get_homogenous, find_fundamental, epipolar_distance, sampson_distance


class Ransac:
    '''RANSAC for optimal fundamental matrix between 2 images'''
    def __init__(
            self,
            model=find_fundamental,
            dist_fn=epipolar_distance,
            max_iter=1000,
            num_points=8,
            threshold=2.5):
        self.model = model
        self.dist_fn= dist_fn
        self.max_iter = max_iter
        self.num_points = num_points
        self.threshold = threshold

    def run(self, point1, point2):
        if point1.ndim == 2:
            pt1_h = get_homogenous(point1)
        if point2.ndim == 2:
            pt2_h = get_homogenous(point2)

        best_inliers = 0
        best_F = None
        best_idx = None
        pts_len = pt1_h.shape[0]
        for iter_i in range(self.max_iter):
            idx = np.random.choice(pts_len, size=self.num_points, replace=False)
            p1 = pt1_h[idx]
            p2 = pt2_h[idx]

            F = self.model(p1, p2)

            dist = sampson_distance(p1, p2, F)
            inliers = dist.squeeze() < self.threshold
            num_inliers = np.sum(inliers)

            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_F = F.copy()
                best_idx = idx

        F = self.model(pt1_h[best_idx], pt2_h[best_idx])

        return F, best_idx
            
