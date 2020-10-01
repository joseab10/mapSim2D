import numpy as np


def rotate2d(theta):
    """
    Compute a 2D rotation matrix given an angle.

    :param theta: (float) Angle to rotate about the z axis.

    :return: (ndarray) A 2x2 rotation matrix.
    """

    s, c = np.sin(theta), np.cos(theta)

    return np.array([[c, -s], [s, c]]).reshape((2, 2))


def quaternion_axis_angle(q):
    """
    Convert a rotation expressed as a quaternion into a 3D vector
    representing the axis of rotation and the rotation angle in radians.

    :param q: (list|tuple) A quaternion (x, y, z, w)

    :return: (numpy.array, float) A tuple containting a 3D numpy vector and the angle as float
    """

    w, v = q[3], q[0:2]
    theta = np.arccos(w) * 2
    return v, theta