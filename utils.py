import os
import math
from typing import List, Iterable
import cv2
import numpy as np


def load_images(im_dir: str='to/the/img/dir') -> List[np.ndarray]:
    imgs = os.listdir(im_dir)
    im_paths = [os.path.join(im_dir, im) for im in imgs]
    return [cv2.imread(im, cv2.COLOR_BGR2GRAY) for im in im_paths]

def filter_good_points(
        matches: Iterable[cv2.DMatch],
        kp1: Iterable[cv2.KeyPoint],
        kp2: Iterable[cv2.KeyPoint],
        r=0.85):
    # matches = sorted(matches, lambda x: x.disatance)
    pts1, pts2 = [], []
    for pair in matches:
        m1, m2 = pair
        if m1.distance <= r * m2.distance:
            pts1.append(kp1[m1.queryIdx].pt)
            pts2.append(kp2[m1.trainIdx].pt)
    return pts1, pts2

def get_homogenous(pts: np.ndarray):
    '''
    pts (np.ndarray): N x 2 sized feature point coordinate
    '''
    z = np.ones(pts.shape[0])
    pts_h = np.hstack([pts, z[:, None]])
    return pts_h

def compute_iter_number(num_samples, p=0.99, e=0.5):
    num = math.log(1 - p) 
    den = math.log(1 - (1 - e) ** num_samples)
    return math.ceil(num / den)

def hartly_norm(pts):
    if pts.shape[1] == 2:
        pts_h = get_homogenous(pts)
    else:
        pts_h = pts.copy()

    center_pts = np.mean(pts_h[:, :2], axis=0)
    mean_pts = pts_h[:, :2] - center_pts
    dist = np.sqrt(np.sum(mean_pts * mean_pts, axis=1))
    mean_dist = np.mean(dist)
    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0
    pts_norm = mean_pts * scale
    T = np.array(
            [[scale, 0, -scale * center_pts[0]],
             [0, scale, -scale * center_pts[1]],
             [0, 0, 1]])
    return pts_norm, T

def find_fundamental(pts1:np.ndarray, pts2:np.ndarray) -> np.ndarray:
    '''
    pts1 (np.ndarray): N x 3. feature points from left image
    pts2 (np.ndarray): N x 3. feature points from right image
    '''
    if pts1.shape[1]== 2:
        pts1 = get_homogenous(pts1)
    if pts2.shape[1] == 2:
        pts2 = get_homogenous(pts2)

    n = pts1.shape[0]
    A = np.zeros((n, 9))
    ul, vl = pts1[:, 0], pts1[:, 1]
    ur, vr = pts2[:, 0], pts2[:, 1]
    A[:, 0] = ur * ul
    A[:, 1] = ur * vl
    A[:, 2] = ur
    A[:, 3] = vr * ul
    A[:, 4] = vr * vl
    A[:, 5] = vr
    A[:, 6] = ul
    A[:, 7] = vl
    A[:, 8] = np.ones(n)
    U, S, Vt = np.linalg.svd(A)

    # Af = 0
    v = Vt[-1, :].reshape(3, 3)
    u, s, vt = np.linalg.svd(v)
    # s[2] = 0      # Fundamental matrix is not singular in practical setting.
    F = u @ np.diag(s) @ vt

    return F / F[-1, -1] # normalize F.

def epipolar_distance(
        pts1: np.ndarray,
        pts2: np.ndarray,
        F: np.ndarray,
        eps: float=1e-7) -> np.ndarray:
    '''
    pts1 (np.ndarray): N x 3. feature ptss from left image
    pts2 (np.ndarray): N x 3. feature ptss from right image
    F (np.ndarray): 3 x 3. fundamental matrix estimated with pts1, 2
    eps (float32): for preventing to divide by 0
    '''
    line_left = F @ pts1.T
    line_right = F.T @ pts2.T
    num = np.abs(np.einsum('ij,jk,ik->i', pts2, F, pts1))
    den_l = np.sqrt(line_left[0, :] ** 2 + line_left[1, :] ** 2)
    den_r = np.sqrt(line_right[0, :] ** 2 + line_right[1, :] ** 2)
    dist_l = num / (den_l + eps)
    dist_r = num / (den_r + eps)
    symmetric_dist = (dist_l + dist_r) / 2

    return symmetric_dist.reshape(-1, 1)

def sampson_distance(pts1, pts2, F):
    F_left = np.matmul(F, pts1.T)
    F_right= np.matmul(F.T, pts2.T)
    dist = np.sum(pts2 * F_left.T, axis=1)
    num = np.abs(dist)
    den = np.sqrt(F_left[0]**2 + F_left[1]**2 + F_right[0]**2 + F_right[1]**2)
    return num / den


def verify_matrix_equivalence(myF, cv2F, points1, points2) -> None:
    """
    Test if both matrices are mathematically equivalent
    """
    print("=== EQUIVALENCE TEST ===")
    
    errors_yours = []
    errors_cv2 = []
    
    for i in range(min(10, len(points1))):
        p1 = np.append(points1[i][:2], 1)
        p2 = np.append(points2[i][:2], 1)
        
        error_yours = abs(p2.T @ myF @ p1)
        error_cv2 = abs(p2.T @ cv2F @ p1)
        
        errors_yours.append(error_yours)
        errors_cv2.append(error_cv2)
    
    print(f"My F - Mean epipolar error: {np.mean(errors_yours):.6f}")
    print(f"OpenCV F - Mean epipolar error: {np.mean(errors_cv2):.6f}")
    
    dist_yours = epipolar_distance(points1, points2, myF)
    dist_cv2 = epipolar_distance(points1, points2, cv2F)
    
    print(f"\nMy F - Mean distance: {np.mean(dist_yours):.3f}")
    print(f"OpenCV F - Mean distance: {np.mean(dist_cv2):.3f}")
    
    # Check if they're proportional
    ratio = np.median(cv2F.flatten() / (myF.flatten() + 1e-15))
    print(f"\nApproximate scale ratio: {ratio:.2f}")
