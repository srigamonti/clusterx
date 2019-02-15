# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

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

