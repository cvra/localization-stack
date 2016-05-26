import unittest
import lidar_localization
from math import cos, sin, pi
import numpy as np


class TestLidarLocalization(unittest.TestCase):

    def test_get_robot_position_form_lidar(self):
        lidar = [1, 1, pi/2]
        robot_position = lidar_localization.get_robot_position_from_lidar(lidar)
        self.assertAlmostEqual([1, 1-0.074, pi/2], robot_position)

    def test_get_robot_position_form_lidar(self):
        robot = [1, 1, pi/2]
        lidar_position = lidar_localization.get_lidar_position_from_robot(robot)
        np.testing.assert_almost_equal([1, 1+0.074, pi/2], lidar_position)

        robot_position = lidar_localization.get_robot_position_from_lidar(lidar_position)
        np.testing.assert_almost_equal(robot, robot_position)
