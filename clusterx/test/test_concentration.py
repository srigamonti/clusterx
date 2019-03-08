# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import clusterx as c
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.utils import dict_compare
from ase import Atoms
import numpy as np

def test_concentration():
    """Test calculation of concentration
    """

    cell = [[3,0,0],
            [0,1,0],
            [0,0,5]]
    positions = [
        [0,0,0],
        [1,0,0],
        [2,0,0]]
    pbc = [True,True,False]

    pri = Atoms(['H','H','H'], positions=positions, cell=cell, pbc=pbc)
    su1 = Atoms(['C','H','H'], positions=positions, cell=cell, pbc=pbc)
    su2 = Atoms(['H','He','H'], positions=positions, cell=cell, pbc=pbc)
    su3 = Atoms(['H','N','H'], positions=positions, cell=cell, pbc=pbc)

    plat = ParentLattice(pri,substitutions=[su1,su2,su3],pbc=pbc)

    scell = SuperCell(plat,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    strset = StructuresSet(plat)
    nstr = 20
    #for i in range(nstr):
    #    strset.add_structure(scell.gen_random(nsubs={}))
    strset.add_structure(
        Structure(scell,
                  decoration_symbols=["H","H","H",
                                      "C","He","H",
                                      "C","He","H"]
                  )
        )

    strset.add_structure(
        Structure(scell,
                  decoration_symbols=["C","H","H",
                                      "C","N","H",
                                      "C","He","H"]
                  )
        )

    fc_ref = [
        {1: [0.33333, 0.66666, 0.0], 2: [0.33333, 0.66666]},
        {1: [0.33333, 0.33333, 0.33333], 2: [0.0, 1.0]}
    ]
    strset.serialize(path="test_cluster_selector_structures_set.json")
    isok = True
    for i,s in enumerate(strset):
        fc = s.get_fractional_concentrations()
        isok *= dict_compare(fc,fc_ref[i],tol=1e-5)

    assert isok
