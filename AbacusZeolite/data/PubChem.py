'''this is benefitted from the PubChem python interface, for more information, see
https://pubchempy.readthedocs.io/en/latest/index.html
'''
import re
import os
import unittest
import logging
import time

import pandas as pd
import numpy as np
import pubchempy as pcp
from ase import Atoms
from ase.io import write

from AbacusZeolite.test.util import init as test_init

def get_structures_by_name(name: str):
    '''
    get the structure of a molecule by its name
    '''
    mol = []
    for c in pcp.get_compounds(name, 'name', record_type='3d'):
        atoms = []
        for a in c.atoms:
            atoms.append((a.element, a.x, a.y, a.z))
        elem, x, y, z = zip(*atoms)
        tauc = np.array(list(zip(x, y, z)))
        mol.append(Atoms(symbols=elem, positions=tauc))
    return mol

class TestPubChem(unittest.TestCase):
    
    #@unittest.skip('Linking to online database, avoid querying too frequently.')
    def test_get_structure_by_name(self):
        name = 'oxygen'
        
        mol = get_structures_by_name(name)
        for i, m in enumerate(mol):
            write(f'{name}-{i}.xyz', m)
        
if __name__ == '__main__':
    flog = f'abacus-zeolite.data.pubchem@{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=flog)
    exit_after_test = test_init()
    unittest.main(exit=exit_after_test)