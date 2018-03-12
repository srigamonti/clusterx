import clusterx as c
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from ase import Atoms
import numpy as np

def test_cluster_expansion():
    """Test generation of clusters pools.
    
    After successful execution of the test, the generated clusters may be visualized with the command::
        
        ase gui test_clusters_generation_#.json

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

    scell = SuperCell(plat,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    structures = []
    structures.append(Structure(scell,[1,2,1,6,7,1,1,2,1]))
    structures.append(Structure(scell,[6,1,1,1,1,1,1,1,1]))
    structures.append(Structure(scell,[1,2,1,1,7,1,6,7,1]))
    structures.append(Structure(scell,[1,1,1,6,1,1,1,7,1]))
    structures.append(Structure(scell,[6,7,1,6,7,1,6,2,1]))
    structures.append(Structure(scell,[1,7,1,6,2,1,1,1,1]))
    structures.append(Structure(scell,[6,1,1,1,1,1,1,1,1]))
    structures.append(Structure(scell,[1,1,1,1,7,1,6,7,1]))
    structures.append(Structure(scell,[1,2,1,6,2,1,6,2,1]))
    structures.append(Structure(scell,[6,1,1,6,2,1,1,1,1]))
    strset = StructuresSet(plat, filename="structures_set.json")
    corrs = corrcal.get_cluster_correlations(structure)
    
    # Generate output
    print ("\n\n========Test writes========")
    atom_idxs, atom_nrs = cpool.get_cpool_arrays()
    scell = cpool.get_cpool_scell()
    cpool.write_orbit_db(cpool.get_cpool(),scell,"test_cluster_correlations_cpool.json")
    structure.serialize(fmt="json",fname="test_cluster_correlations_structure.json")
    print("Correlations: ",corrs)
    print ("===========================\n")

    print ("========Asserts========")
    
    assert np.allclose([-0.33333333,0.,-0.,0.33333333,0.57735027,-0.33333333,-0.25,-0.,-0.25],corrs,atol=1e-5)
    #print(corrs)
