# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import clusterx as c
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.utils import dict_compare
from ase import Atoms
import numpy as np

def test_concentration():
    """Test calculation of concentration
    """

    cell = [[4,0,0],
            [0,1,0],
            [0,0,5]]
    positions = [
        [0,0,0],
        [1,0,0],
        [2,0,0],
        [3,0,0]]
    pbc = [True,True,False]

    pri = Atoms(['H','Ba','H','Ba'], positions=positions, cell=cell, pbc=pbc)
    su1 = Atoms(['C','Ba','H','Ba'], positions=positions, cell=cell, pbc=pbc)
    su2 = Atoms(['H','He','H','He'], positions=positions, cell=cell, pbc=pbc)
    su3 = Atoms(['H','N','H','N'], positions=positions, cell=cell, pbc=pbc)

    plat = ParentLattice(pri,substitutions=[su1,su2,su3],pbc=pbc)

    scell = SuperCell(plat,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    s = Structure(scell,
                  decoration_symbols=["C","He","H","Ba",
                                      "H","He","H","N",
                                      "H","Ba","H","He"]
                  )

    # From atoms:
    tags = s.get_tags()

    # From parent lattice:
    natoms = s.get_natoms()
    n_sub_sites = s.get_n_sub_sites()
    sub_sites = s.get_substitutional_sites()
    sub_tags = s.get_substitutional_tags()
    spect_sites = s.get_spectator_sites()
    spect_tags = s.get_spectator_tags()
    nsites_per_type = s.get_nsites_per_type()
    sites = s.get_sites()
    idx_subs = s.get_idx_subs()
    atidx_site0 = s.get_atom_indices_for_site_type(0)
    atidx_site1 = s.get_atom_indices_for_site_type(1)
    atidx_site2 = s.get_atom_indices_for_site_type(2)
    # From SuperCell
    index = s.get_index()
    trafo = s.get_transformation()

    # From Structure
    numbers = s.get_atomic_numbers()
    sigmas = s.get_sigmas()

    print("tags",repr(tags))
    print("natoms", natoms)
    print("n_sub_sites", n_sub_sites)
    print("sub_sites", sub_sites)
    print("sub_tags", sub_tags)
    print("spect_sites", spect_sites)
    print("spect_tags", spect_tags)
    print("nsites_per_type", nsites_per_type)
    print("sites", sites)
    print("idx_subs", idx_subs)
    print("index", index)
    print("trafo", repr(trafo))
    print("atom_indices_for_site_type 0", atidx_site0)
    print("atom_indices_for_site_type 1", atidx_site1)
    print("atom_indices_for_site_type 2", atidx_site2)
    print("numbers", repr(numbers))
    print("sigmas", repr(sigmas))


    isok = True
    isok *= (tags == [1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 0, 2]).all()
    isok *= natoms == 12
    isok *= n_sub_sites == 2
    isok *= (np.asarray(sub_sites) == [0, 1, 3, 4, 5, 7, 8, 9, 11]).all()
    isok *= (sub_tags == [1,2]).all()
    isok *= (spect_sites == np.array([2, 6, 10])).all()
    isok *= (spect_tags == np.array([0])).all()
    isok *= dict_compare(nsites_per_type, {0: 3, 1: 3, 2: 6})
    isok *= dict_compare(sites, {0: np.array([1, 6]), 1: np.array([56,  2,  7]), 2: np.array([1]), 3: np.array([56,  2,  7]), 4: np.array([1, 6]), 5: np.array([56,  2,  7]), 6: np.array([1]), 7: np.array([56,  2,  7]), 8: np.array([1, 6]), 9: np.array([56,  2,  7]), 10: np.array([1]), 11: np.array([56,  2,  7])})
    isok *= dict_compare(idx_subs, {0: np.array([1]), 1: np.array([1, 6]), 2: np.array([56,  2,  7])})
    isok *= index == 3
    isok *= (trafo == [[1, 0, 0],[0, 3, 0],[0, 0, 1]]).all()
    isok *= (atidx_site0 == np.array([ 2,  6, 10])).all()
    isok *= (atidx_site1 == np.array([0, 4, 8])).all()
    isok *= (atidx_site2 == np.array([ 1,  3,  5,  7,  9, 11])).all()
    isok *= (numbers == [ 6,  2,  1, 56,  1,  2,  1,  7,  1, 56,  1,  2]).all()
    isok *= (sigmas == [1, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 1]).all()

    assert isok
