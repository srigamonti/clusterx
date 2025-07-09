"""Default objects for testing purposes."""

from ase.spacegroup import crystal

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell


def get_clathrate_plat():
    a = 10.5148
    x = 0.185
    y = 0.304
    z = 0.116
    wyckoff = [
        (0, y, z), #24k
        (x, x, x), #16i
        (1/4., 0, 1/2.), #6c
        (1/4., 1/2., 0), #6d
        (0, 0 , 0) #2a
    ]

    # Build the parent lattice
    pri = crystal(['Si','Si','Si','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])
    sub = crystal(['Al','Al','Al','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])
    plat = ParentLattice(atoms=pri,substitutions=[sub])
    return plat


def get_clathrate_supercell(p_cell=[(2,0,0),(0,2,0),(0,0,2)]):
    plat = get_clathrate_plat()
    return SuperCell(plat,p=p_cell)
