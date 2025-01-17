'''
this is the module for interfacing with the IZA database

Website:
https://www.iza-structure.org/databases/

The webpage that collects all the zeolites:
https://america.iza-structure.org/IZA-SC/ftc_table.php

https://america.iza-structure.org/IZA-SC/download_cif.php?ID=133
'''

INDEX = 'https://www.iza-structure.org/databases/'
TABLE = 'https://america.iza-structure.org/IZA-SC/ftc_table.php'

import time
import requests
import unittest
import logging
import os
import json

from AbacusZeolite.test.util import init as test_init

where_am_i = os.path.dirname(os.path.abspath(__file__))

def _request_iza_id_table():
    '''
    send a request to get the table of zeolite names and IZA IDs.
    This relies on the HTML parse by BeautifulSoup
    '''
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        errmsg = 'Failed to import BeautifulSoup'
        logging.error(errmsg)
        raise ImportError(errmsg)
    
    logging.info('Request on IZA table sent >>')
    response = requests.get(TABLE)
    if response.status_code != 200:
        errmsg = 'Failed to get information from IZA'
        logging.error(errmsg)
        raise ValueError(errmsg)
    logging.info('<< Request on IZA table returned')
    
    soup = BeautifulSoup(response.text, 'html.parser')
    # search all <td> tags whose CSS class is 'CodeTable',
    # the content of <td> is <a>, whose href contains the
    # IZA ID in the pattern of 'framework.php?ID=xxx', we
    # extract the number part. The name of zeolite is the
    # content of the <a> tag, an example would be:

    # <div class="FwTable">
    #  <table cellpadding="3" cellspacing="2" class="center">
    #   <tr>
    #    <td class="CodeTable">
    #     <a href="framework.php?ID=18">
    #      ABW
    #     </a>
    #    </td>
    #    <td class="CodeTable">
    #     <a href="framework.php?ID=19">
    #      ACO
    #     </a>
    #    </td>
    out = {}
    for td in soup.find_all('td', class_='CodeTable'):
        a = td.find('a')
        iza_id = int(a['href'].split('=')[-1])
        name = a.text.strip()
        out[name] = iza_id
    return out

def _request_cif_via_id(iza_id):
    '''
    send a request to the IZA website to get the structure information
    '''
    logging.info(f'Request on IZA-ID {iza_id} sent >>')
    url = f'https://america.iza-structure.org/IZA-SC/download_cif.php?ID={iza_id}'
    response = requests.get(url)
    if response.status_code != 200:
        errmsg = f'Failed to get information from IZA for IZA-ID {iza_id}'
        logging.error(errmsg)
        raise ValueError(errmsg)
    logging.info(f'<< Request on IZA-ID {iza_id} returned')
    return response.text

def _download_kernel_impl(iza_id, outdir, fn):
    '''
    download the CIF file for a given IZA ID
    
    Parameters
    ----------
    iza_id : int
        the IZA ID of the zeolite
    outdir : str
        the directory to save the CIF file
    fn : str
        the filename of the CIF file
    
    Returns
    -------
    str
        the path to the downloaded CIF file
        
    Raises
    ------
    ValueError
        if the download fails
    '''
    logging.info(f'Downloading IZA-ID {iza_id} >>')
    
    # short circuit if the file already exists
    if os.path.exists(os.path.join(outdir, fn)):
        logging.info(f'<< IZA-ID {iza_id} already downloaded')
        return os.path.join(outdir, fn)
    
    # download the CIF file
    try:
        cif = _request_cif_via_id(iza_id)
    except ValueError:
        errmsg = f'Failed to download IZA-ID {iza_id}'
        logging.error(errmsg)
        raise ValueError(errmsg)
    
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, fn), 'w') as f:
        f.write(cif)
    logging.info(f'<< IZA-ID {iza_id} downloaded')
    return os.path.join(outdir, fn)

def download(ftab = f'{os.path.join(os.path.dirname(where_am_i), "data", "table", "IZA-id.json")}',
             outdir = f'{os.path.join(os.path.dirname(where_am_i), "data", "download")}',
             **kwargs):
    '''
    download the CIF file for a given request, either by IZA ID or by name
    
    Parameters
    ----------
    ftab : str
        the path to the IZA ID table file, default value is fine, no need to change
        in most cases
    outdir : str
        the directory to save the CIF file
    id : int
        the IZA ID of the zeolite
    name : str
        the name of the zeolite
    
    Returns
    -------
    str
        the path to the downloaded CIF file
    '''
    if all(k in kwargs for k in ['id', 'name']):
        errmsg = 'Cannot specify both IZA ID and name'
        logging.error(errmsg)
        raise ValueError(errmsg)
    if not any(k in kwargs for k in ['id', 'name']):
        errmsg = 'Must specify either IZA ID or name'
        logging.error(errmsg)
        raise ValueError(errmsg)

    if 'id' in kwargs:
        iza_id = kwargs['id']
    elif 'name' in kwargs:
        with open(ftab) as f:
            tab = json.load(f)
        if kwargs['name'] not in tab:
            errmsg = f'Zeolite name {kwargs["name"]} not found in the IZA table'
            logging.error(errmsg)
            raise ValueError(errmsg)
        iza_id = tab[kwargs['name']]

    return _download_kernel_impl(iza_id, outdir, f'{iza_id}.cif')
    
class IZAInterfaceTest(unittest.TestCase):
    
    @unittest.skip('Do not bother the IZA website too much')
    def test_request_iza_id_table(self):
        tab = _request_iza_id_table()
        print(tab)
        
if __name__ == '__main__':
    flog = f'abacus-zeolite.structure.iza@{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=flog)
    exit_after_test = test_init()
    unittest.main(exit=exit_after_test)
    
    logging.shutdown()
