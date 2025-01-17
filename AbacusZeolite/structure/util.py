'''
Functions
---------
- abc_angles_to_vec: convert lattice parameters to vectors
- vec_to_abc_angles: convert vectors to lattice parameters
- dist: calculate the distance under PBC
- rdf: calculate the Radial distribution function (RDF) of the group of atoms

'''

# built-in modules
import os
import unittest
import logging
import time
import json

# third-party modules
import numpy as np

# home-made modules
from AbacusZeolite.test.util import init as test_init

where_am_i = os.path.dirname(os.path.abspath(__file__))

def abc_angles_to_vec(lat: list) -> np.ndarray:
    """convert lattice parameters to vectors
    
    NOTE: the alpha, beta, gamma should be given in degree
    """
    assert len(lat) == 6, f'lat should be a list of 6 floats: {lat}'
    a, b, c, alpha, beta, gamma = lat
    alpha, beta, gamma = np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)
    e11, e12, e13 = a, 0, 0
    e21, e22, e23 = b * np.cos(gamma), b * np.sin(gamma), 0
    e31 = c * np.cos(beta)
    e32 = (b * c * np.cos(alpha) - e21 * e31) / e22
    e33 = np.sqrt(c**2 - e31**2 - e32**2)
    return np.array([[e11, e12, e13], [e21, e22, e23], [e31, e32, e33]])

def vec_to_abc_angles(vec: np.ndarray) -> np.ndarray:
    """convert vectors to lattice parameters
    
    NOTE: the alpha, beta, gamma will be returned in degree
    """
    assert vec.shape == (3, 3), f'vec should be a 3x3 array: {vec}'
    a = np.linalg.norm(vec[0])
    b = np.linalg.norm(vec[1])
    c = np.linalg.norm(vec[2])
    alpha = np.arccos(np.dot(vec[1], vec[2]) / (b * c))
    beta = np.arccos(np.dot(vec[0], vec[2]) / (a * c))
    gamma = np.arccos(np.dot(vec[0], vec[1]) / (a * b))
    return np.array([a, b, c, np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)],
                    dtype=np.float64)

def dist(ri: np.ndarray, rj: np.ndarray, cell: np.ndarray) -> float:
    '''calculate the distance under PBC
    
    Parameters
    ----------
    ri: np.ndarray
        the Cartesian coordinates of the first atom
    rj: np.ndarray
        the Cartesian coordinates of the second atom
    cell: np.ndarray
        the cell matrix in shape (3, 3)
    
    Returns
    -------
    float
        the distance between the two atoms under PBC
    '''
    
    ri_direct = np.linalg.solve(cell.T, ri)
    rj_direct = np.linalg.solve(cell.T, rj)
    rij_direct = (ri_direct - rj_direct + 0.5) % 1 - 0.5
    rij = np.dot(cell.T, rij_direct)
    return np.linalg.norm(rij)

def rdf(cell: np.ndarray, 
        tau: np.ndarray, 
        icenter = 0, 
        nbin = 100,
        direct=True) -> np.ndarray:
    '''
    calculate the Radial distribution function (RDF) of the group of atoms
    '''
    # we will use direct coordinates in the following
    taud = tau if direct else np.linalg.solve(cell, tau.T).T
    r = [np.linalg.norm(((taud[icenter] - t + 0.5)%1 - 0.5)@cell.T) for t in taud]
    rmax = np.linalg.norm(np.array([0.5, 0.5, 0.5])@cell.T)
    hist, bins = np.histogram(r, bins=nbin, range=(0, rmax))

    # normalize the histogram with the volume of the shell
    g = hist.flatten()
    g[0] = 0 # the first bin is always 0
    return bins[:-1], g

def center(atoms: np.ndarray, weight = None, **kwargs):
    '''calculate the center of selected atoms
    
    Parameters
    ----------
    atoms : np.ndarray
        the coordinates of the atoms
    weight : None|str
        the weight of the atoms, default is None, which corresponding to
        the equal weight for each atom/geometrical center, can also be 
        'mass' for the mass center of the atoms
    elem : list
        the list of atomic symbols, if weight is 'mass', this parameter
        must be given
    
    Returns
    -------
    np.ndarray
        the center of the selected atoms
    '''
    power = np.ones(atoms.shape[0])
    if weight == 'mass':
        elem = kwargs.get('elem')
        if elem is None:
            errmsg = 'elem should be given when weight is mass'
            logging.error(errmsg)
            raise ValueError(errmsg)
        index = os.path.join(os.path.dirname(where_am_i), 'data', 'table', 'index.json')
        with open(index) as f:
            index = json.load(f)
        mass = os.path.join(os.path.dirname(where_am_i), 'data', 'table', 'mass.json')
        with open(mass) as f:
            mass = json.load(f)
        power = np.array([mass[index[e]] for e in elem])
    
    # calculate the center
    return np.average(atoms, axis=0, weights=power)
 
class TestStructureUtil(unittest.TestCase):
    def test_convert_between_abc_and_vec(self):
        lat = np.array([10, 10, 10, 90, 90, 90])
        vec = abc_angles_to_vec(lat)
        lat2 = vec_to_abc_angles(vec)
        self.assertTrue(np.allclose(lat, lat2))
        
        lat = np.array(np.random.rand(3).tolist() + [60, 60, 60]).astype(float)
        vec = abc_angles_to_vec(lat)
        lat2 = vec_to_abc_angles(vec)
        self.assertTrue(np.allclose(lat, lat2))
        
    def test_dist(self):
        r0 = [0, 0, 0]
        r1 = [1, 0, 0]
        a, b, c, alpha, beta, gamma = 10, 10, 10, 90, 90, 90
        cell = abc_angles_to_vec([a, b, c, alpha, beta, gamma])
        self.assertTrue(np.allclose(dist(r0, r1, cell), 1))
        r1 = [55, 0, 0]
        self.assertTrue(np.allclose(dist(r0, r1, cell), 5))
        r1 = [10, 0, 10]
        self.assertTrue(np.allclose(dist(r0, r1, cell), 0))
        r1 = [10.1, 10, 0]
        self.assertTrue(np.allclose(dist(r0, r1, cell), 0.1))
        r1 = [10.1, 10.1, 100.1]
        self.assertTrue(np.allclose(dist(r0, r1, cell), 0.1 * np.sqrt(3)))
        r1 = [4.9, 105.1, 5.1]
        self.assertTrue(np.allclose(dist(r0, r1, cell), 4.9 * np.sqrt(3)))
        
    @unittest.skip('overall workflow, not always needed to test')
    def test_rdf(self):
        from AbacusZeolite.data.IZA import download
        from ase.io.cif import read_cif
        import matplotlib.pyplot as plt
        
        fn = download(name='CHA')
        parsed = read_cif(fn)
        cell = parsed.get_cell()
        tauc = parsed.get_positions()
        
        r, hist = rdf(cell, tauc, icenter=0, nbin=100, direct=False)
        plt.plot(r, hist)
        plt.savefig('rdf.png')
        plt.close()
        
if __name__ == '__main__':
    flog = f'abacus-zeolite.structure.util@{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=flog)
    exit_after_test = test_init()
    unittest.main(exit=exit_after_test)

    logging.shutdown()
    