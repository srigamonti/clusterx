# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from ase.build import cut, add_adsorbate
from ase.spacegroup import crystal

def HatZnO_10m10(a=3.2493, c=5.2054, u=0.345, nlayers=12, ads_symbol="X",vacuum = 20.0):
    """Build parent lattice for H adsorbed on ZnO(10m10) surface

    The 10m10 surface is built from cutting the bulk Wurtzite ZnO structure on
    the (non-polar) a-c plane. The space group of bulk ZnO  is 186 (P6_3mc).
    It has two "2b" Wyckoff sites, (1/3,2/3,u),(2/3,1/3,u+1/2), with parameters
    u=0 and u=0.345. u*c corresponds to the short in-plane ZnO bond length.
    ZnO departs slightly from the perfect hexagonal cell: for ZnO c/a=1.602,
    while for the perfect hexagonal it is c/a=1.633.

    Adsorption of H atoms occurs on top of Zn or on top of O.

    Parameters:

    a and c:  float
        bulk lattice parameters of Wurtzite ZnO
    u: float
        short ZnO bond in units of c. Bulk Wyckoff sites are (1/3,2/3,0) and
        (1/3,2/3,u)
    nlayers: integer
        number of ZnO layers of the built slab
    ads_symbol: string
        Symbol of the adsorbed species. For the default parameters only "H" or
        "X" (vacancy) make physical sense.
    vacuum: float
        amount of vacuum between top-most adsorbed atom and slab bottom surface.
    """
    from clusterx.utils import sort_atoms
    zno_bondlength_over_c = u

    zno = crystal(['Zn','O'],[(1./3.,2./3.,0),(1./3.,2./3.,zno_bondlength_over_c)],
        spacegroup = 186,
        cellpar = [a,a,c,90,90,120]
        )
    znos = cut(zno,a=(1,0,0),b=(0,0,1),c=(1,2,0),nlayers=nlayers) # a: x; b=z; c=y
    znos.rotate('y','-z',rotate_cell = True)
    cell = znos.get_cell()
    cell[2,2] *= -1
    znos.set_cell(cell)
    znos.center(vacuum=vacuum/2.0,axis=2)
    znos.translate([0,0,-vacuum/2.0])
    znos.set_pbc([True,True,False])

    _znos = sort_atoms(znos)
    _znos.center(vacuum=vacuum/2.0,axis=2)
    _znos.translate([0,0,-vacuum/2.0])
    add_adsorbate(_znos,ads_symbol,1.769,position=(0.5*a,0.3826*c)) # H@Zn
    add_adsorbate(_znos,ads_symbol,0.986,position=(0.5*a,0.9500*c)) # H@O
    return _znos
