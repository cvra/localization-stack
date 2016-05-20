import numpy as np
import math


class TransformationModel(object):
    """ TransformationModel

        Modelise an affine transformation composed of a rotation 
        followed by a translation.
    """

    def __call__(self, coords):
        coords = np.array(coords, copy=False, ndmin=2)

        x, y = np.transpose(coords)
        src = np.vstack((x, y, np.ones_like(x)))
        dst = np.dot(src.transpose(), self.params.transpose())

        # rescale to homogeneous coordinates
        dst[:, 0] /= dst[:, 2]
        dst[:, 1] /= dst[:, 2]

        return dst[:, :2]

    def __init__(self, matrix=None, rotation=None, translation=None):
        params = any(param is not None for param in (rotation, translation))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix
        elif params:
            if rotation is None:
                rotation = 0
            if translation is None:
                translation = (0, 0)

            self.params = np.array([
                [math.cos(rotation), math.sin(rotation), 0],
                [math.sin(rotation), math.cos(rotation), 0],
                [                 0,                  0, 1]
            ])
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(3)


    def estimate(self, p, q):
        p_mean = np.sum(p,axis=0) / len(p)
        q_mean = np.sum(q,axis=0) / len(q)

        centered_p = p - p_mean
        centered_q = q - q_mean

        S = np.dot(centered_p.T, centered_q)
        U, Epsilon, VT = np.linalg.svd(S)

        R = np.dot(VT.T,np.linalg.det(np.dot(VT.T,U.T)))
        R = np.dot(R,U.T)

        translation = q_mean - np.dot(R, p_mean)

        self.params = np.zeros((3,3))
        self.params[2,2] = 1
        self.params[0:2,0:2] = R
        self.params[0:2, 2] = translation

    def residuals(self, src, dst):
        return np.sqrt(np.sum((self(src) - dst)**2, axis=1))

    @property
    def rotation(self):
        return math.atan2(self.params[1, 0], self.params[0, 0])

    @property
    def translation(self):
        return self.params[0:2, 2]