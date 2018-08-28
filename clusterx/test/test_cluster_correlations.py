import clusterx as c
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from ase import Atoms
import numpy as np

def test_cluster_correlations():
    """Test calculation of cluster correlations.

    After successful execution of the test, the generated structure and clusters pool may be visualized with the command::

        ase gui test_clusters_correlations_structure_#.json
        ase gui test_clusters_correlations_cpool.json

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
    cpool = ClustersPool(plat, npoints=[1,2], radii=[0,1.2])
    corrcal = CorrelationsCalculator("trigonometric", plat, cpool)

    scell1 = SuperCell(plat,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    structure1 = Structure(scell1,[1,1,1,6,7,1,1,2,1])
    corrs1 = corrcal.get_cluster_correlations(structure1)

    # Doubling of structure1. Correlations should not change.
    scell2 = SuperCell(plat,np.array([(1,0,0),(0,6,0),(0,0,1)]))
    structure2 = Structure(scell2,[1,1,1,6,7,1,1,2,1,1,1,1,6,7,1,1,2,1])
    corrs2 = corrcal.get_cluster_correlations(structure2)

    # Testing for Chebychev polinomials as single site basis functions.
    # Testing does not (yet) include multi - sublattices. 
    
    su21 = Atoms(['C','C','C'], positions=positions, cell=cell, pbc=pbc)
    su22 = Atoms(['N','N','N'], positions=positions, cell=cell, pbc=pbc)

    plat2 = ParentLattice(pri,substitutions=[su21,su22],pbc=pbc)
    cpool2 = ClustersPool(plat2, npoints=[1,2], radii=[0,1.2])
    corrcal2 =  CorrelationsCalculator("chebychev", plat2, cpool2)

    scell21 = SuperCell(plat2,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    structure21 = Structure(scell21,[1,1,1,6,7,1,6,7,1])
    corrs21 = corrcal2.get_cluster_correlations(structure21)

    # Doubling of structure21. Correlations should not change.
    scell22 = SuperCell(plat2,np.array([(1,0,0),(0,6,0),(0,0,1)]))
    structure22 = Structure(scell22,[1,1,1,6,7,1,6,7,1,1,1,1,6,7,1,6,7,1])
    corrs22 = corrcal2.get_cluster_correlations(structure22)

    
    # Generate output
    print ("\n\n========Test writes========")
    print (test_cluster_correlations.__doc__)
    #atom_idxs, atom_nrs = cpool.get_cpool_arrays()
    scell = cpool.get_cpool_scell()
    cpool.write_clusters_db(cpool.get_cpool(),scell,"test_cluster_correlations_cpool.json")
    structure1.serialize(fmt="json",fname="test_cluster_correlations_structure_1.json")
    structure2.serialize(fmt="json",fname="test_cluster_correlations_structure_2.json")
    #print("Correlations 1: ",corrs1)
    #print("Correlations 2: ",corrs2)
    print ("=========Chebychev=========\n")

    scell = cpool2.get_cpool_scell()
    cpool2.write_clusters_db(cpool2.get_cpool(),scell,"test_cluster_correlations_chebychev_cpool.json")
    structure21.serialize(fmt="json",fname="test_cluster_correlations_structure_1_chebychev.json")
    structure22.serialize(fmt="json",fname="test_cluster_correlations_structure_2_chebychev.json")
    print("Correlations 21: ",corrs21)
    print("Correlations 22: ",corrs22)    

    print ("===========================\n")

    print ("========Asserts========")

    assert np.allclose([-0.33333333,0.,-0.,0.33333333,0.57735027,-0.33333333,-0.25,-0.,-0.25],corrs1,atol=1e-5)
    assert np.allclose([-0.33333333,0.,-0.,0.33333333,0.57735027,-0.33333333,-0.25,-0.,-0.25],corrs2,atol=1e-5)
    #print(corrs)
