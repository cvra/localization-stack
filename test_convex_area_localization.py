import unittest
from convex_area_localization import filter_positions
from math import cos, sin, pi
import numpy as np


class TestConvexAreaLocalization(unittest.TestCase):

    def test_filter_positions_similar_pos(self):
        last_pos = (np.array([[0.5, 0.5]]), np.array([pi/2]))
        pos1 = (np.array([[1, 1]]), np.array([pi/2]))
        pos2 = (np.array([[1.1, 1.1]]), np.array([pi/2+0.01]))
        
        config = dict()
        config['MAX_ERR_POSITION'] = 1
        config['MAX_ERR_HEADING'] = 1
        config['POS_ERR_FACTOR'] = 5.5
        config['HEADING_ERR_FACTOR'] = 1.0
        config['MAX_POS_DISTANCE'] = 2

        position, heading = filter_positions(last_pos=last_pos,
                                             positions=(pos1, pos2),
                                             options=config)

        pos_out = (np.array([[1.05, 1.05]]), np.array([pi/2+0.005]))
        np.testing.assert_almost_equal(position, pos_out[0])
        np.testing.assert_almost_equal(heading, pos_out[1])


    def test_filter_positions_one_bad(self):
        last_pos = (np.array([[0.5, 0.5]]), np.array([pi/2]))
        pos1 = (np.array([[0.55, 0.56]]), np.array([pi/2+0.01]))
        pos2 = (np.array([[5, 5]]), np.array([pi/2+1]))
        
        config = dict()
        config['MAX_ERR_POSITION'] = 1
        config['MAX_ERR_HEADING'] = 1
        config['POS_ERR_FACTOR'] = 5.5
        config['HEADING_ERR_FACTOR'] = 1.0
        config['MAX_POS_DISTANCE'] = 2

        position, heading = filter_positions(last_pos=last_pos,
                                             positions=(pos1, pos2),
                                             options=config)

        pos_out = (np.array([[0.55, 0.56]]), np.array([pi/2+0.01]))
        np.testing.assert_almost_equal(position, pos_out[0])
        np.testing.assert_almost_equal(heading, pos_out[1])


    def test_filter_positions_all_bad(self):
        last_pos = (np.array([[0.5, 0.5]]), np.array([pi/2]))
        pos1 = (np.array([[10, 10]]), np.array([pi]))
        pos2 = (np.array([[5, 5]]), np.array([pi/2+1]))
        
        config = dict()
        config['MAX_ERR_POSITION'] = 1
        config['MAX_ERR_HEADING'] = 1
        config['POS_ERR_FACTOR'] = 5.5
        config['HEADING_ERR_FACTOR'] = 1.0
        config['MAX_POS_DISTANCE'] = 2

        position, heading = filter_positions(last_pos=last_pos,
                                             positions=(pos1, pos2),
                                             options=config)

        np.testing.assert_equal(position, None)
        np.testing.assert_equal(heading, None)

