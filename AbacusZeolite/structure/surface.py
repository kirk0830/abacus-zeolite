'''
one important part of heteogeneous catalysis is the surface/slab system
'''

from ase.build import surface
from ase.build import bulk, add_adsorbate
from ase.io import write
from ase import Atoms

import unittest
import numpy as np
import logging
import time

from AbacusZeolite.test.util import init as test_init

class TestSurfaceUtil(unittest.TestCase):
    
    @unittest.skip('experiment, this indeed works, so we skip it now')
    def test_Rh_hcp0001_build(self):
        # test the build of hcp0001 of Rh
        a = 2.73
        c = 4.5
        Rh = bulk('Rh', 'hcp', a=a, c=c)
        Rh_0001 = surface(Rh, indices=(0, 0, 1), layers=3, vacuum=10)
        # export as cif
        write('Rh_0001.cif', Rh_0001)
    
    @unittest.skip('experiment, this indeed works, so we skip it now')
    def test_Rh_hcp0001_supercell_build(self):
        # test the build of hcp0001 of Rh and duplicate it in XY direction for times
        a = 2.73
        c = 4.5
        Rh = bulk('Rh', 'hcp', a=a, c=c)
        Rh_0001 = surface(Rh, indices=(0, 0, 1), layers=3, vacuum=10).repeat((2, 2, 1))
        # export as cif
        write('Rh_0001_supercell.cif', Rh_0001)
    
    @unittest.skip('experiment, this indeed works, so we skip it now')
    def test_Rh_hcp0001_221_CO_adsorbate_build(self):
        # test the build of 221-duplicated hcp0001 of Rh and add CO adsorbate
        a = 2.73
        c = 4.5
        Rh = bulk('Rh', 'hcp', a=a, c=c)
        slab = surface(Rh, indices=(0, 0, 1), layers=2, vacuum=10).repeat((3, 3, 1))
        # add CO adsorbate
        adsorbate = Atoms('CO', positions=[[0, 0, 0], [0, 0, 1.1]])
        add_adsorbate(slab=slab, 
                      adsorbate=adsorbate, 
                      height=1.5, 
                      position=(1.5, 1.5) # X, Y position
                      )
        # export as cif
        write('Rh_0001_221_CO.cif', slab)
    
if __name__ == '__main__':
    flog = f'abacus-zeolite.structure.surface@{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=flog)
    exit_after_test = test_init()
    unittest.main(exit=exit_after_test)

    logging.shutdown()
    