# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import sys
import clusterx
import clusterx.clusters_selector_with_splitbregman
from clusterx.clusters_selector_with_splitbregman import ClustersSelector
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.clusters.cluster import Cluster
from clusterx.calculators.emt import EMT2
from ase import Atoms
from clusterx.parent_lattice import ParentLattice
from clusterx.correlations import CorrelationsCalculator
from clusterx.super_cell import SuperCell
from clusterx.structures_set import StructuresSet
from clusterx.structure import Structure
from clusterx.utils import isclose

import numpy as np

def test_cluster_selector_split_bregman():
    
    strset, energies, comat, clmults, cpool = get_structure_set()

    lamb = 0.9
    mu_min = 0.00001
    mu_max = 0.10
    mu_step = 0.001
    
    ####### Class ClusterSelector ##########
    clsel = ClustersSelector('split_bregman', cpool, sparsity_max=mu_max, sparsity_min=mu_min, sparsity_step=mu_step, l=lamb, LOO_idx=18)
    clsel.select_clusters(comat, energies, mult = clmults)
    ##################################
    cp2 = clsel.optimal_clusters._cpool
    npoints=[]
    radius=[]
    for i,c in enumerate(cp2):
        npoints.append(c.npoints)
        radius.append(c.radius)

    precomp_npoints = [0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4]
    precomp_radius = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 1.4142135623730951, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 1.4142135623730951, 1.4142135623730951, 1.4142135623730951, 1.4142135623730951, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 1.4142135623730951, 1.4142135623730951]

    print ("========Asserts========")
    isok1 = isclose(npoints, precomp_npoints)
    isok2 = isclose(radius, precomp_radius)
    assert(isok1)
    assert(isok2)
    print("\n Test of split_bregman in CELL was successful.\n\n")


def get_structure_set():

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
    cpool = ClustersPool(plat, npoints=[0,1,2,3,4], radii=[0,0,2.3,2.3,1.42])
    #cpool = ClustersPool(plat, npoints=[1,2,3,4], radii=[0,2.3,1.42,1.42])
    cpool.write_clusters_db(cpool.get_cpool(),cpool.get_cpool_scell(),"cpool.json")
    corrcal = CorrelationsCalculator("trigonometric", plat, cpool)

    scell = SuperCell(plat,np.array([(1,0,0),(0,3,0),(0,0,1)]))
 
    strset = StructuresSet(plat, filename="test_cluster_selector_structures_set.json")
    #nstr = 20
    #for i in range(nstr):
    #    strset.add_structure(scell.gen_random(nsubs={}))
    strset.add_structure(Structure(scell,[1,2,1,6,7,1,1,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,1,1,1,1,1,1,1,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,2,1,1,7,1,6,7,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,1,1,6,1,1,1,7,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,7,1,6,7,1,6,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,7,1,6,2,1,1,1,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,1,1,1,1,1,1,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,1,1,1,7,1,6,7,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,2,1,6,2,1,6,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,1,1,6,2,1,1,1,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,1,1,6,7,1,1,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,2,1,6,2,1,1,1,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,7,1,1,1,1,6,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,1,1,1,7,1,1,7,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,2,1,6,7,1,6,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,7,1,1,2,1,1,1,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,7,1,1,7,1,1,1,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,2,1,6,1,1,6,7,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[1,7,1,1,1,1,6,2,1]),write_to_db=True)
    strset.add_structure(Structure(scell,[6,1,1,1,7,1,1,1,1]),write_to_db=True)
    
    # Get the DATA(comat) + TARGET(energies)
    comat = corrcal.get_correlation_matrix(strset)

    strset.set_calculator(EMT2())
    energies = strset.calculate_property()
   
    clmults = cpool.get_multiplicities()

    return strset, energies, comat, clmults, cpool

if __name__ == "__main__":
    test_cluster_selector_split_bregman()

