# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from clusterx import utils
from clusterx.parent_lattice import ParentLattice
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from ase.build import bulk
from ase.data import atomic_numbers as an
from ase import Atoms
import numpy as np

def test_get_unique_supercells():
    """Test generation of unique supercells.

    Three cases are tested: Square 2D lattice of index 4 (the case of
    Fig.1 of [1]); FCC lattice of index 4 (cf. the seven non-decorated
    structures of Fig. 2 and Table IV in [2]); and the simple cubic lattice
    (Fig.11 and Table IV of [2])

    [1] Computational Materials Science 59 (2012) 101–107
    [2] Phys. Rev. B 77, 224115 2008
    """


    for case in range(3):

        if case == 0: #Square (2D, i.e. pbc = (1,1,0))
            a=3.1
            index = 4
            cell = np.array([[1,0,0],[0,1,0],[0,0,1]])
            positions = np.array([[0,0,0]])
            sites = [[12,13]]
            pris = Atoms(cell=cell*a, positions=positions*a)

            pl = ParentLattice(pris, sites=sites, pbc=(1,1,0))

            unique_scs, unique_trafos = utils.get_unique_supercells(index,pl)

            sset = StructuresSet(pl)
            for t in unique_trafos:
                scell = SuperCell(pl,t)
                sset.add_structure(Structure(scell,scell.get_atomic_numbers()),write_to_db = True)

            sset.serialize(path="test_get_unique_supercells-square_lattice.json", overwrite=True)
            print("\nFound ",len(unique_scs), " unique HNFs for a 2D square lattice of index ",index)
            #print("SCS: ", unique_scs)
            #print("TRA: ", unique_trafos)
            isok0 = len(unique_scs) == 4 and unique_scs[1][1][1] == 12.4

        if case == 1: #FCC
            a=3
            index = 4
            cell = np.array([[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]])
            positions = np.array([[0,0,0]])
            sites = [[an["Cu"],an["Au"]]]
            pris_fcc = Atoms(cell=cell*a,positions=positions*a,pbc=(1,1,1))

            pl = ParentLattice(pris_fcc,sites=sites)

            unique_scs, unique_trafos = utils.get_unique_supercells(index,pl)

            sset = StructuresSet(pl)
            for t in unique_trafos:
                scell = SuperCell(pl,t)
                sset.add_structure(Structure(scell,scell.get_atomic_numbers()),write_to_db = True)

            sset.serialize(path="test_get_unique_supercells-fcc.json")
            print("Found ",len(unique_scs), " unique HNFs for a FCC lattice of index ",index)
            #print("SCS: ", unique_scs)
            #print("TRA: ", unique_trafos)
            isok1 = len(unique_scs) == 7 and unique_trafos[4][2][2] == 4

        if case == 2: #Simple cubic
            a=3.1
            index = 4
            cell = np.array([[1,0,0],[0,1,0],[0,0,1]])
            positions = np.array([[0,0,0]])
            sites = [[12,13]]
            pris = Atoms(cell=cell*a, positions=positions*a)

            pl = ParentLattice(pris, sites=sites, pbc=(1,1,1))

            unique_scs, unique_trafos = utils.get_unique_supercells(index,pl)

            sset = StructuresSet(pl)
            for t in unique_trafos:
                scell = SuperCell(pl,t)
                sset.add_structure(Structure(scell,scell.get_atomic_numbers()),write_to_db = True)

            sset.serialize(path="test_get_unique_supercells-sc.json")
            print("Found ",len(unique_scs), " unique HNFs for a simple cubic lattice of index ",index)
            #print("SCS: ", unique_scs)
            #print("TRA: ", unique_trafos)
            isok2 = len(unique_scs) == 9 and unique_scs[4][2][2] == 12.4

    assert isok0 and isok1 and isok2
