import clusterx as c
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
#from clusterx.model import ModelConstructor
from clusterx.fitter import Fitter
from ase import Atoms
import numpy as np
from clusterx.calculators.emt import EMT2
from clusterx.utils import isclose
from clusterx.clusters_selector import ClustersSelector

from clusterx.visualization import plot_optimization_vs_number_of_clusters
from clusterx.visualization import plot_optimization_vs_sparsity



def test_clusters_selector_lasso():
    """Test model optimization

    After successful execution of the test, the generated structures and
    clusters pool may be visualized with the command::

        ase gui test_clusters_selector_[...].json

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
    cpool = ClustersPool(plat, npoints=[0,1,2,3,4], radii=[0,2.3,2.3,2.3,2.3])
    #cpool = ClustersPool(plat, npoints=[1,2,3,4], radii=[0,2.3,1.42,1.42])
    cpool.write_clusters_db(cpool.get_cpool(),cpool.get_cpool_scell(),"cpool.json")
    corrcal = CorrelationsCalculator("trigonometric", plat, cpool)

    scell = SuperCell(plat,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    strset = StructuresSet(plat, filename="test_cluster_selector_structures_set.json")
    nstr = 20
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

    clsel = ClustersSelector('lasso', cpool, sparsity_max=0.1, sparsity_min=0.01)
    #clsel = ClustersSelector('combinations', cpool, fitter_size = "linreg", nclmax=2)
    #clsel = ClustersSelector('size+combinations', cpool, fitter_size = "linreg", nclmax = 2, set0 = [2,1])
    clsel.select_clusters(comat,energies)

    #clsets.get_subpool(clsel.get_optimal_clusters())

    cp2 = clsel.optimal_clusters._cpool
    npoints=[]
    radius=[]
    for i,c in enumerate(cp2):
        npoints.append(c.npoints)
        radius.append(c.radius)

    #plot_optimization_vs_number_of_clusters(clsel)
    #plot_optimization_vs_sparsity(clsel)
        
    print(npoints)
    print(radius)
    print(clsel.rmse)
    print(clsel.cvs)
    print(clsel.lasso_sparsities)
    print(clsel.ecis)
    print(clsel.opt_rmse)
    print(clsel.opt_mean_cv)

    rnpoints = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    rradius = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.4142135623730951, 2.0, 2.0, 2.0, 2.0, 2.0, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.0, 2.0, 2.0, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979])
    rrmse = np.array([1.0362039383487336, 0.9344465657331334, 0.8325591301898301, 0.730958066121657, 0.6297819874397272, 0.5292746707497706, 0.4282700680600906, 0.32120255311475965, 0.2141350357080133, 0.10706751830169857])
    rcvs = np.array([20.90974525188963, 20.680683374055917, 20.489908755488713, 20.29064319974821, 20.09380404992501, 19.939031976312478, 19.826338620327313, 19.75313945265403, 19.72277055047523, 19.742075745376187])
    rsparsities=np.array([0.1, 0.09000000000000001, 0.08000000000000002, 0.07000000000000002, 0.06000000000000002, 0.05000000000000002, 0.040000000000000015, 0.030000000000000013, 0.02000000000000001, 0.01000000000000001])
    recis = np.array([212.64680242224546, -14.748596089593418, 174.25793826725473, -327.8290056313618, 5.030466117063875, -2.600317890766761, -14.154347622437898, 18.438344976756053, -2.600317890766704, -14.15434762243787, 18.43834497675618, 5.030466117064102, -9.304991072216614, -2.285927878877856, 2.6892727749404854, -6.755621889068123, -6.755621889068109, 0.1754865986121055, -2.285927878877856, 2.6892727749404854, -9.304991072216628, -9.304991072216628, 1.1177216021907932, -5.779299067667393, 7.655791799498951, 1.1177216021907932, -5.779299067667393, 7.655791799498951, 7.655791799498951, 1.1177216021908074, -5.779299067667393])
    ropt_rmse = 1.6281354602834095e-13
    ropt_mean_cv = 34.459748414541146
            
    isok = isclose(rsparsities,clsel.lasso_sparsities) and isclose(rnpoints, npoints) and isclose(rradius, radius) and isclose(rrmse, clsel.rmse) and isclose(rcvs, clsel.cvs) and isclose(recis, clsel.ecis) and isclose(ropt_rmse, clsel.opt_rmse) and isclose(ropt_mean_cv, clsel.opt_mean_cv).all()

    assert(isok)
