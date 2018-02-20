# Copyright 2015-2018 HU
# (see accompanying license files for details).

"""CELL (aka clusterX)"""

import numpy as np
from clusterx.config import CellConfig

#from clusterx.parent_lattice import ParentLattice

config = CellConfig()
config.read()
config.write()
cfg = config.config



PRECISION = int(cfg["GENERAL"]["PRECISION"])
T2D = config.is_2D()

#__all__ = ['ParentLattice', 'Atom']
__version__ = '1.0.0.dev5'

