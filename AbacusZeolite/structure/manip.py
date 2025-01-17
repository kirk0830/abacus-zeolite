'''structure manipulation functions'''

# built-in modules
import os
import unittest
import logging
import json
import itertools as it
import time
from multiprocessing import Process, Queue

# third-party modules
import numpy as np
from scipy.optimize import minimize

# home-made modules
from AbacusZeolite.test.util import init as test_init
from AbacusZeolite.structure.util import dist

where_am_i = os.path.dirname(os.path.abspath(__file__))

def _so3rep_axis(axis, angle, deg=True):
    '''return the rotation matrix for SO(3) group
    
    Parameters
    ----------
    axis : str
        the axis of rotation, should be one of 'x', 'y', 'z'
    angle : float
        the angle of rotation in degree, default is clockwise
    
    Returns
    -------
    np.ndarray
        the rotation matrix for SO(3) group
    '''
    angle = np.deg2rad(angle) if deg else angle
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        return np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
    else:
        errmsg = 'axis should be one of x, y, z'
        logging.error(errmsg)
        raise ValueError(errmsg)

def _cal_mass_center(elem, coord):
    '''calculate the mass center of the group of atoms
    
    Parameters
    ----------
    elem : list
        the list of atomic symbols
    coord : np.ndarray
        the Cartesian coordinates of the atoms
    
    Returns
    -------
    np.ndarray
        the Cartesian coordinates of the mass center
    '''
    datadir = os.path.join(os.path.dirname(where_am_i), 'data')
    with open(os.path.join(datadir, 'index.json')) as f:
        index = json.load(f)
    with open(os.path.join(datadir, 'mass.json')) as f:
        mass = json.load(f)
    
    # because the mass is stored in a list, we need to get the atomic index to
    # achieve a quick access.
    mass_center = np.zeros(3)
    
    for e, c in zip(elem, coord):
        mass_center += np.array(c) * mass[index[e] - 1] # because the index starts from 1
    mass_center /= np.sum([mass[index[e] - 1] for e in elem])
    return mass_center

def rotate(atoms, angle, **kwargs):
    '''
    rotate the group of atoms with respect to either one axis, or one
    point. For one axis, only one angle is needed, for one point, the
    azimuthal and polar angles are needed.
    
    If the mode is not specified, will infer from angles: if one, will
    rotate with respect to z-axis, if two, will rotate with respect to
    the mass center.
    '''
    if all([key in kwargs for key in ['point', 'axis']]):
        errmsg = 'only one of point and axis should be given'
        logging.error(errmsg)
        raise ValueError(errmsg)

class TestStructureManipulate(unittest.TestCase):
    def test_so3rep_axis(self):
        for _ in range(10): # test for 10 times
            axis = np.random.choice(['x', 'y', 'z'])
            angle = np.random.rand() * 360
            mat = _so3rep_axis(axis, angle)
            matinv = _so3rep_axis(axis, -angle)
            self.assertTrue(np.allclose(np.dot(mat, matinv), np.eye(3)))

if __name__ == '__main__':
    flog = f'abacus-zeolite.structure.manip@{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=flog)
    exit_after_test = test_init()
    unittest.main(exit=exit_after_test)

    logging.shutdown()
