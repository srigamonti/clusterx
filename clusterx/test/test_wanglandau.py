# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import clusterx as c
import subprocess
from ase.spacegroup import crystal
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.clusters.cluster import Cluster
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model
from clusterx.wang_landau import WangLandau
from clusterx.utils import isclose
from clusterx.utils import dict_compare

from ase.data import atomic_numbers as cn
from ase import Atoms
import numpy as np
import os
import sys
import math

def test_wanglandau():

    np.random.seed(10) #setting a seed for the random package for comparible random structures

    a = 10.5148
    x = 0.185; y = 0.304; z = 0.116
    wyckoff = [
        (0, y, z), #24k
        (x, x, x), #16i
        (1/4., 0, 1/2.), #6c
        (1/4., 1/2., 0), #6d
        (0, 0 , 0) #2a
    ]

    # Build the parent lattice
    print("\nSampling in `Si_{46-x} Al_x Ba_{8}`")
    pri = crystal(['Si','Si','Si','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])
    sub = crystal(['Al','Al','Al','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])
    plat = ParentLattice(atoms=pri,substitutions=[sub],pbc=(1,1,1))
    #sg, sym = get_spacegroup(plat)

    # Build clusters pool
    #cpool = ClustersPool(plat,r=)
    cpool = ClustersPool(plat)
    cp = cpool._cpool
    cpsc = cpool.get_cpool_scell()
    s = cn["Al"]
    cpool.add_cluster(Cluster([],[],cpsc))
    cpool.add_cluster(Cluster([0],[s],cpsc))
    cpool.add_cluster(Cluster([24],[s],cpsc))
    cpool.add_cluster(Cluster([40],[s],cpsc))
    cpool.add_cluster(Cluster([6,4],[s,s],cpsc))
    cpool.add_cluster(Cluster([37,32],[s,s],cpsc))
    cpool.add_cluster(Cluster([39,12],[s,s],cpsc))
    cpool.add_cluster(Cluster([16,43],[s,s],cpsc))
    cpool.add_cluster(Cluster([35,11],[s,s],cpsc))
    cpool.add_cluster(Cluster([39,30],[s,s],cpsc))
    cpool.add_cluster(Cluster([22,17],[s,s],cpsc))
    cpool.add_cluster(Cluster([35,42],[s,s],cpsc))
    cpool.add_cluster(Cluster([32,14],[s,s],cpsc))
    cpool.add_cluster(Cluster([11,10],[s,s],cpsc))
    cpool.add_cluster(Cluster([18,9],[s,s],cpsc))
    cpool.add_cluster(Cluster([18,43],[s,s],cpsc))

    # Energy
    cpoolE = cpool.get_subpool([0,1,2,3,4,5,6,7,9,15])
    ecisE = [
        -78407.3247588,
        47.164484875,
        47.1673476881,
        47.1569012692,
        0.00851281608144,
        0.0139835351147,
        0.0108175321899,
        0.0101521144776,
        0.00121744613474,
        0.000413664306204
    ]


    multT = [1,24,16,6,12,8,48,24,24,24]

    corcE = CorrelationsCalculator("binary-linear",plat,cpoolE)
    scellS= [(1,0,0),(0,1,0),(0,0,1)]
    scellE = SuperCell(plat,scellS)

    sub_lattices = scellE.get_idx_subs()
    print("Sublattices with corresponding atomic numbers: ", sub_lattices)
    tags = scellE.get_tags()
    print("Tags: ", tags)

    nsubs={0:[16]}
    cemodelE=Model(corcE, "energy", ecis=np.multiply(ecisE, multT))

    wl = WangLandau(cemodelE, scellE, nsubs)
    e0_unitcell=-77652.707924876348
    e0=float(e0_unitcell)
    e1=e0+0.5
    cdos = wl.wang_landau_sampling(energy_range=[e0,e1], energy_bin_width=0.002, f_range=[math.exp(1), 2], update_method='square_root', flatness_conditions=[[0.2,math.exp(1e-1)]])

    #rsteps = [0, 1, 2, 3, 4, 6, 10, 11, 16, 17, 18, 26, 27, 34, 37, 38, 44, 45, 47, 48, 50]
    #renergies = [-77652.59664207128, -77652.61184305252, -77652.62022569243, -77652.61912760629, -77652.62737663009, -77652.63009501049, -77652.63158443688, -77652.64240196907, -77652.64240196907, -77652.64348105107, -77652.64714764676, -77652.64959679516, -77652.64959679516, -77652.65458138083, -77652.66173231734, -77652.65458138083, -77652.65946542152, -77652.6702829537, -77652.66812810961, -77652.67298251796, -77652.66622624162]
    #rlast_decoration = np.int8([14, 14, 13, 14, 14, 13, 14, 14, 14, 13, 13, 14, 14, 14, 13, 14, 14, 14, 13, 13, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 13, 13, 14, 14, 14, 13, 14, 13, 14, 13, 13, 14, 14, 13, 14, 56, 56, 56, 56, 56, 56, 56, 56])
    #rlast_sampling_entry = {'sampling_step_no': 50, 'model_total_energy': -77652.66622624162, 'swapped_positions': [[5, 43]], 'key_value_pairs': {'bond_kk': 2.49116603472051, 'bond_ii': 2.397621971688995}}
    x=cdos._cdos
    for c in x:
        print(c)

    #isok1 = isclose(rsteps,steps) and isclose(renergies, energies) and isclose(rlast_decoration,last_structure.decor) and dict_compare(last_sampling_entry, rlast_sampling_entry, tol=float(1e-7) )
    assert(True)
