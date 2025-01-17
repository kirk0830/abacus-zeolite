'''
this module stores functions that prepare the input for the calculator
'''
# in-built modules
import os
import json
import re

import numpy as np

where_am_i = os.path.dirname(os.path.abspath(__file__))

def build_structure(elem: list,
                    atoms: np.ndarray,
                    cell: np.ndarray,
                    alat = 1.8897259886,
                    direct = True,
                    outdir = None,
                    fn = 'STRU',
                    **kwargs):
    '''write the ABACUS STRU file for the structure
    
    Parameters
    ----------
    elem : list
        the list of atomic symbols
    atoms : np.ndarray
        the coordinates of the atoms, in shape of (natom, 3)
    cell : np.ndarray
        the cell matrix, in shape of (3, 3)
    alat : float
        the lattice constant, in unit of Bohr. The default value
        is 1.8897259886 Bohr, which allows the input of the
        Angstrom unit.
    direct : bool
        whether the coordinates are in direct space. Default is True
    outdir : str
        the output directory, default is the current working directory
    fn : str
        the filename of the output file, default is 'STRU'
    fpsp : str
        the pseudopotential file, for KS-DFT calculation, this must
        be provided, otherwise will cause the crash of the program
    forb : str
        the basis set file, for KS-DFT calculation with NAO basis,
        this must be provided, otherwise will cause the crash of the
        program
    mass : list
        the list of atomic masses, if not provided, will try to read
        from the database
    
    
    numerical descriptor is not included in recent supporting plan.
    
    Returns
    -------
    fn : str
        the full path of the output file
    '''
    outdir = outdir or os.getcwd()
    
    out = 'ATOMIC_SPECIES\n'
    mass = kwargs.get('mass')
    
    if mass is None:
        with open(os.path.join(where_am_i, 'data', 'index.json')) as f:
            index = json.load(f)
        with open(os.path.join(where_am_i, 'data', 'mass.json')) as f:
            mass = json.load(f)
        
        temp = [re.match(r'([A-Z][a-z]?)', e).group(1) for e in elem]
        mass = [mass[index[e]] for e in temp]
    
    fpsp = kwargs.get('fpsp')
    fpsp = fpsp or [''] * len(elem)
    for e, m, f in zip(elem, mass, fpsp):
        out += f'{e:<4} {m:>8.4f} {f}\n'
    
    forb = kwargs.get('forb')
    if forb is not None:
        out += f'\nNUMERICAL_ORBITAL\n'
        for f in forb:
            out += f'{f}\n'
    
    out += f'\nLATTICE_CONSTANT\n{alat}\n'
    out += f'\nLATTICE_VECTORS\n'
    for v in cell:
        out += f'{v[0]:>15.10f} {v[1]:>15.10f} {v[2]:>15.10f}\n'
    
    out += f'\nATOMIC_POSITIONS\n'
    # ABACUS 