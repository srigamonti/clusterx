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
from clusterx.thermodynamics.wang_landau import WangLandau
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

    wl = WangLandau(cemodelE, scellE, ensemble = "canonical", nsubs = nsubs)
    e0_unitcell=-77652.707924876348
    e0=float(e0_unitcell)
    e1=e0+0.5
    cdos = wl.wang_landau_sampling(energy_range=[e0,e1], energy_bin_width=0.002, f_range=[math.exp(1), 2], update_method='square_root', flatness_conditions=[[0.1,math.exp(1e-1)]])

    #x=cdos._cdos
    #energy_bins = []
    #gs = []
    #for c in x:
    #    if c[1] > float(1):
    #        energy_bins.append(float(c[0]))
    #        gs.append(float(c[1]))
    energy_bins, gs = cdos.get_cdos(ln = True, normalization = False)

    renergy_bins = [-77652.64792487655, -77652.64592487656, -77652.64192487657, -77652.6339248766, -77652.63192487661, -77652.62992487662, -77652.62792487662, -77652.62592487663, -77652.62392487664, -77652.62192487664, -77652.61992487665, -77652.61792487666, -77652.61592487666, -77652.61392487667, -77652.61192487668, -77652.60992487668, -77652.60792487669, -77652.6059248767, -77652.6039248767, -77652.60192487671, -77652.59992487672, -77652.59792487673, -77652.59592487673, -77652.59392487674, -77652.59192487675, -77652.58992487675, -77652.58792487676, -77652.58592487677, -77652.58392487677, -77652.58192487678, -77652.57992487679, -77652.5779248768, -77652.5759248768, -77652.57392487681, -77652.57192487681, -77652.56992487682, -77652.56792487683, -77652.56592487684, -77652.56392487684, -77652.56192487685, -77652.55992487686, -77652.55792487686, -77652.55592487687, -77652.55392487688, -77652.55192487688, -77652.54992487689, -77652.5479248769, -77652.5459248769, -77652.54392487691, -77652.54192487692, -77652.53992487692, -77652.53792487693, -77652.53592487694, -77652.53392487695, -77652.53192487695, -77652.52992487696, -77652.52792487697, -77652.52592487697, -77652.52392487698, -77652.52192487699, -77652.519924877, -77652.517924877, -77652.515924877, -77652.51392487701, -77652.50992487703, -77652.50792487703, -77652.50392487705, -77652.49392487708, -77652.4879248771]
    rgs = [3.0, 2.0, 3.0, 2.0, 6.0, 4.0, 5.0, 10.0, 3.0, 8.0, 8.0, 9.0, 8.0, 9.0, 10.0, 7.0, 10.0, 10.0, 8.0, 13.0, 10.0, 9.0, 10.0, 11.0, 10.0, 11.0, 11.0, 8.0, 12.0, 12.0, 13.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0, 12.0, 13.0, 11.0, 11.0, 11.0, 11.0, 11.0, 10.0, 8.0, 9.0, 9.0, 10.0, 8.0, 12.0, 9.0, 9.0, 9.0, 3.0, 5.0, 8.0, 8.0, 3.0, 6.0, 9.0, 3.0, 5.0, 8.0, 2.0, 7.0, 2.0, 2.0, 2.0]
    
    isok1 = isclose(renergy_bins,energy_bins) and isclose(rgs, gs)
    assert(isok1)
