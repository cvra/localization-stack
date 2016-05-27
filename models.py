import numpy as np
import math
from cmath import rect, phase


def mean_angle(rad_list):
    return np.array(phase(sum(rect(1, rad) for rad in rad_list)/len(rad_list)))


def diff_angle(rad_list, rad_ref):
    return np.array([math.atan2(math.sin(rad-rad_ref), math.cos(rad_ref-rad)) for rad in rad_list])


def position_residuals(position1, position2):
    if len(position1[0].shape) < 2:
        position1[0] = np.expand_dims(position1[0], axis=0)
        position1[1] = np.expand_dims(position1[1], axis=0)

    err_position = np.sqrt((np.sum((position1[0] - position2[0])**2, axis = 1)))
    err_heading = np.abs(diff_angle(position1[1], position2[1]))

    return err_position, err_heading


class PositionModel(object):
    """ PositionModel

        Modelise the position of the robot using its XY position
        and heading angle.
    """

    def __init__(self, heading=None, translation=None, pos_err_factor=5.5, heading_err_factor=1.0):
        self.pos_err_factor = pos_err_factor
        self.heading_err_factor = heading_err_factor

        params = any(param is not None for param in (heading, translation))

        if params:
            if heading is None:
                heading = 0
            if translation is None:
                translation = (0, 0)
        else:
            heading = 0
            translation = (0, 0)

        self.params = np.hstack([translation, heading])

    def estimate(self, positions, headings):
        if len(positions.shape) < 2:
            positions = positions.reshape(1,2)

        position = np.sum(positions, axis=0) / positions.shape[0]
        heading = mean_angle(headings)

        self.params = np.hstack([position, heading])

    def residuals(self, positions, headings):
        err_positions, err_heading = position_residuals(position1=(positions, headings), 
                                                        position2=(self.position, self.heading))

        return err_positions * self.pos_err_factor + err_heading * self.heading_err_factor

    @property
    def heading(self):
        return self.params[2]

    @property
    def position(self):
        return self.params[0:2]


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

        if len(p) < 2 or len(q) < 2:
            return False

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

    def is_data_valid(cls, data, min_distance):
        distance = np.sqrt(np.sum((data[0]-data[1])**2))
        if distance > min_distance:
            return True
        else:
            return False

    @property
    def rotation(self):
        return math.atan2(self.params[1, 0], self.params[0, 0])

    @property
    def translation(self):
        return self.params[0:2, 2]