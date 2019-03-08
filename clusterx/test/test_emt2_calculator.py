# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import clusterx as c
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.calculators.emt import EMT2
from ase import Atoms
import numpy as np

def test_emt2_calculator():
    """Test calcualtion of energies with EMT2 calculator.

    After successful execution of the test, the generated structures may be visualized with the command::

        ase gui test_emt2_calculator#.json

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
    strset.add_structure(Structure(scell,[1,2,1,6,7,1,1,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,1,1,1,1,1,1,1,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,2,1,1,7,1,6,7,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,1,1,6,1,1,1,7,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,7,1,6,7,1,6,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,7,1,6,2,1,1,1,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,1,1,1,1,1,1,1,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,1,1,1,7,1,6,7,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,2,1,6,2,1,6,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,1,1,6,2,1,1,1,1]),write_to_db=True)
    strset.serialize(path="test_cluster_expansion_structures_set.json")

    strset.set_calculator(EMT2())
    energies = strset.calculate_property()

    # Generate output
    print ("\n\n========Test writes========")
    print(test_emt2_calculator.__doc__)
    print("Energies:\n")
    print(np.array2string(np.array(energies),separator=",",max_line_width=1000))
    print ("===========================\n")

    print ("========Asserts========")

    assert np.allclose( [ 420.02464215,  11.85279614, 200.11679975,  27.15487534, 162.41156144, 193.56777501,  11.85279614,  35.84803264, 585.51975722, 213.05311213], energies,atol=1e-5)
    #print(comat)
