from ase.calculators.emt import EMT 
#from ase.calculators.calculator import names 
import ase.calculators.emt

ase.calculators.emt.parameters.update( {
    # parameters "for fun"
    #      E0     s0    V0     eta2    kappa   lambda  n0
    #      eV     bohr  eV     bohr^-1 bohr^-1 bohr^-1 bohr^-3
    'Fe': (-3.28, 3.00, 1.493, 1.240, 2.000, 1.169, 0.00700),#same as Al
    'Si': (-3.51, 2.67, 2.476, 1.652, 2.740, 1.906, 0.00910),#same as Cu
    'Ge': (-2.96, 3.01, 2.132, 1.652, 2.790, 1.892, 0.00547),#same as Ag
    'Ba': (-3.80, 3.00, 2.321, 1.674, 2.873, 2.182, 0.00703),#same as Au
    'Sr': (-3.82, 2.95, 2.156, 1.774, 2.853, 2.182, 0.00703)
})


class EMT2(EMT,object):
    def __init__(self, **kwargs):
        EMT.__init__(self, **kwargs)



