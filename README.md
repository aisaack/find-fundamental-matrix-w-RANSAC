# Finding fundamental matrix from scratch
## Objective
- Implement RANSAC algorithm for finding fundamental matrix $F$.

## Entire process
- Detect feature points by ORB detector.
- Compute description of eatch point.
- Match feature points of left image with right image.
- Conduct ratio testing.
- Run RANSAC to find $F$.

## RANSAC
- Sample $N$ points. 
- Estimate $F$ with sampled points.
- Compute error with all poitns and $F$.
- Sampson error is applied for calculating distances between the sampled points in this implementation.
- If the error is smaller than threshold, the samples are considered inlier set. 
- Iterate above process $\frac{\log (1-p)}{\log (1-(1-e)^N)}$ times where $p$, $e$, and $N$ are confidence level, outlier ratio and sample size repectively.

## Fundamental matrix estimation
- Create matrix $\mathbf{A}$ using matched feature points for $\mathbf{Af}=0$.
- Compute SVD on $\mathbf{A}$.
- Since this script use left-right images, rank of fundamental matrix is 2, however, usually it is 3 since it is not singular in practical environment.
- Again, conduct SVD on last column of right singular vector since it is correspond to smallest singular value.


