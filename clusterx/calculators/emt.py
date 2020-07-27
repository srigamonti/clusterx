# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from ase.calculators.emt import EMT
#from ase.calculators.calculator import names
import ase.calculators.emt

"""From https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/emt.html#EMT
parameters = {
    #      E0     s0    V0     eta2    kappa   lambda  n0
    #      eV     bohr  eV     bohr^-1 bohr^-1 bohr^-1 bohr^-3
    'Al': (-3.28, 3.00, 1.493, 1.240, 2.000, 1.169, 0.00700),
    'Cu': (-3.51, 2.67, 2.476, 1.652, 2.740, 1.906, 0.00910),
    'Ag': (-2.96, 3.01, 2.132, 1.652, 2.790, 1.892, 0.00547),
    'Au': (-3.80, 3.00, 2.321, 1.674, 2.873, 2.182, 0.00703),
    'Ni': (-4.44, 2.60, 3.673, 1.669, 2.757, 1.948, 0.01030),
    'Pd': (-3.90, 2.87, 2.773, 1.818, 3.107, 2.155, 0.00688),
    'Pt': (-5.85, 2.90, 4.067, 1.812, 3.145, 2.192, 0.00802),
    # extra parameters - just for fun ...
    'H': (-3.21, 1.31, 0.132, 2.652, 2.790, 3.892, 0.00547),
    'C': (-3.50, 1.81, 0.332, 1.652, 2.790, 1.892, 0.01322),
    'N': (-5.10, 1.88, 0.132, 1.652, 2.790, 1.892, 0.01222),
    'O': (-4.60, 1.95, 0.332, 1.652, 2.790, 1.892, 0.00850)}
"""

ase.calculators.emt.parameters.update( {
    # more parameters "for fun"
    #      E0     s0    V0     eta2    kappa   lambda  n0
    #      eV     bohr  eV     bohr^-1 bohr^-1 bohr^-1 bohr^-3
    'Fe': (-3.28, 3.00, 1.493, 1.240, 2.000, 1.169, 0.00700),#same as Al
    'Si': (-3.51, 2.67, 2.476, 1.652, 2.740, 1.906, 0.00910),#same as Cu
    'Ge': (-2.96, 3.01, 2.132, 1.652, 2.790, 1.892, 0.00547),#same as Ag
    'Sn': (-3.90, 2.87, 2.773, 1.818, 3.107, 2.155, 0.00688),#same as Pd
    'Ba': (-3.80, 3.00, 2.321, 1.674, 2.873, 2.182, 0.00703),#same as Au
    'He': (-4.44, 2.60, 3.673, 1.669, 2.757, 1.948, 0.01030), #same as Ni
    'Na': (-5.10, 1.88, 0.132, 1.652, 2.790, 1.892, 0.01222), #same as N
    'Zn': (-3.51, 2.77, 2.376, 1.652, 2.740, 1.976, 0.01010), #very similar to Cu
    'B': (-4.80, 1.83, 0.232, 1.652, 2.790, 1.892, 0.01252), #
    'X': (-3.50, 1.81, 0.332, 1.652, 2.790, 1.892, 0.01322) #same as C
})


class EMT2(EMT):
    def __init__(self, **kwargs):
        EMT.__init__(self, **kwargs)
