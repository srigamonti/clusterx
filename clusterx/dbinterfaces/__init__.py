""""
This module is designed to obtain data from a NOMAD OASIS database and
make it usable for CELL.
It is specifically designed for the OASIS of the SOL group but may be
modified to work for other OASIS databases or NOMAD itself.

1.  Create a conda environment with python
    $conda create --name myenv python
2.  Install CELL
    If you are a developer at HU, clone the code via
    $git clone git@git.physik.hu-berlin.de:srigamonti/clusterx.git
    then enter the cloned repository
    $cd clusterx
    Then make sure to follow the development branch
    $git checkout develop
    $pip install -e . --use-pep517
3.  Install dotenv which is needed for this module
    $pip install python-dotenv
4.  In order to use jupyter notebooks,
    create a kernel for the virtual environment
    $python -m ipykernel install --user --name=myenv

Following adjustment needs to be made for the current version of CELL
    Open model.py and replace all "fit_params" with "params"
"""
import requests
import numpy as np
from ase import Atoms


BASE_URL_SOL = 'https://sol-oasis.physik.hu-berlin.de/nomad-oasis/api/v1/'
# BASE_URL_NOMAD = 'http://nomad-lab.eu/prod/v1/api/v1/'
# elementary_charge = 1.602176634e-19 C = e        
# as defined in NOMAD constants_en.txt
elementary_charge = 1.602176634*10**(-19)