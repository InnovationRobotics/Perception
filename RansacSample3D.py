import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import LineModelND, ransac
import scipy.linalg


class PlaneModel3D:

    def __init__(self):
        # our parameters for the model are represented in the following order
        # ax + by + c - z = 0
        self.params = None

    def estimate(self, data):
        """Estimate 3d plane model from data.

        This minimizes the sum of shortest (orthogonal) distances
        from the given data points to the estimated plane.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimensionality dim >= 2.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

        self.params = C

        return True

    def residuals(self, data, params=None):
        """Determine residuals of data to model.

        For each point, the shortest (orthogonal) distance to the plane

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimension dim.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """

        # a * x + b * y + 1 * c = z
        # our parameters for the model are represented in the following order
        # ax + by + c - z = 0
        # distance to a plane - find explination here: https://www.geeksforgeeks.org/distance-between-a-point-and-a-plane-in-3-d/
        # https://www.google.com/search?q=calcualte+the+distance+from+a+3d+point+to+a+plane&oq=calcualte+the+distance+from+a+3d+point+to+a+plane+&aqs=chrome..69i57j0l2.10173j0j7&sourceid=chrome&ie=UTF-8#kpvalbx=_95pKXp-eIMT16QSSr5GoCQ34
        planeNorm = np.sqrt( self.params[0] * self.params[0] + self.params[1] * self.params[1] + 1)
        distanceToPlane = abs(data[:, 0] * self.params[0] + data[:, 1] * self.params[1] + self.params[2] - data[:, 2]) / planeNorm

        return distanceToPlane

np.random.seed(seed=1)

# generate coordinates of line
point = np.array([0, 0, 0], dtype='float')
direction = np.array([1, 1, 1], dtype='float') / np.sqrt(3)
xyz = point + 1 * np.arange(-10, 10)[..., np.newaxis] * direction

# add gaussian noise to coordinates
noise = np.random.normal(size=xyz.shape)
xyz += 0.5 * noise
xyz[::2] += 1 * noise[::2]
xyz[::4] += 1 * noise[::4]

# robustly fit line only using inlier data with RANSAC algorithm
model_robust, inliers = ransac(xyz, LineModelND, min_samples=3,
                               residual_threshold=1, max_trials=1000)
outliers = inliers == False

print(model_robust.params)

if False:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[inliers][:, 0], xyz[inliers][:, 1], xyz[inliers][:, 2], c='b',
               marker='o', label='Inlier data')
    ax.scatter(xyz[outliers][:, 0], xyz[outliers][:, 1], xyz[outliers][:, 2], c='r',
               marker='o', label='Outlier data')
    ax.legend(loc='lower left')
    plt.show()

 # Continue - now estiamte a plane using LS - not a line vector
# best-fit linear plane
A = np.c_[xyz[:, 0], xyz[:, 1], np.ones(xyz.shape[0])]
C, _, _, _ = scipy.linalg.lstsq(A, xyz[:, 2])  # coefficients

# evaluate it on grid
# regular grid covering the domain of the data
X, Y = np.meshgrid(np.arange(-10, 10, 0.5), np.arange(-10, 10, 0.5))
Z = C[0] * X + C[1] * Y + C[2]

if False:
    # plot points and fitted surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='r', s=50)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    #ax.axis('equal')
    #ax.axis('tight')
    plt.show()

# Now estimate the 3d plane using ransac algorithm
model_robust, inliers = ransac(xyz, PlaneModel3D, min_samples=3,
                               residual_threshold=1, max_trials=1000)


if False:
    Cransac = model_robust.params
    Zransac = Cransac[0] * X + Cransac[1] * Y + Cransac[2]
    # plot points and fitted surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Zransac, rstride=1, cstride=1, alpha=0.2)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='r', s=50)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('ransac versus LS estimation for a 3d plane')
    #ax.axis('equal')
    #ax.axis('tight')
    plt.show()

# plt.close('all')
print('finished')