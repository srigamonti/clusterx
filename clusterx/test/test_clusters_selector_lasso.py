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
#from clusterx.model import ModelConstructor
from ase import Atoms
import numpy as np
from clusterx.calculators.emt import EMT2
from clusterx.utils import isclose
from clusterx.clusters_selector import ClustersSelector

from clusterx.visualization import plot_optimization_vs_number_of_clusters
#from clusterx.visualization import plot_optimization_vs_sparsity
#from clusterx.visualization import plot_predictions_vs_target



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
    cpool = ClustersPool(plat, npoints=[0,1,2,3,4], radii=[0,0,2.3,2.3,2.3])
    #cpool = ClustersPool(plat, npoints=[1,2,3,4], radii=[0,2.3,1.42,1.42])
    cpool.write_clusters_db(cpool.get_cpool(),cpool.get_cpool_scell(),"cpool.json")
    corrcal = CorrelationsCalculator("trigonometric", plat, cpool)

    scell = SuperCell(plat,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    strset = StructuresSet(plat)
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
    strset.serialize(path="test_cluster_selector_structures_set.json")
    # Get the DATA(comat) + TARGET(energies)
    comat = corrcal.get_correlation_matrix(strset)
    strset.set_calculator(EMT2())
    energies = strset.calculate_property()

    clsel = ClustersSelector(method='lasso', sparsity_max=0.10, sparsity_min=0.01, max_iter=1000000000, tol=1e-12, sparsity_scale = "piece_log", cv_splits=None)
    #clsel = ClustersSelector('combinations', cpool, fitter_size = "linreg", nclmax=2)
    #clsel = ClustersSelector('size+combinations', cpool, fitter_size = "linreg", nclmax = 2, set0 = [2,1])
    #clsel.select_clusters(comat,energies)
    clsel.select_clusters(strset,cpool,"energy")

    #clsets.get_subpool(clsel.get_optimal_clusters())

    cp2 = clsel.optimal_clusters._cpool
    npoints=[]
    radius=[]
    for i,c in enumerate(cp2):
        npoints.append(c.npoints)
        radius.append(c.radius)


    #plot_optimization_vs_number_of_clusters(clsel)
    #plot_optimization_vs_sparsity(clsel)
    #plot_predictions_vs_target(clsel,energies)

    print(npoints)
    print(radius)
    print(clsel.rmse)
    print(clsel.cvs)
    print(clsel.lasso_sparsities)
    #print(clsel.ecis)
    #print(clsel.opt_rmse)
    #print(clsel.opt_mean_cv)


    #rnpoints = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4])
    #rradius = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.4142135623730951, 2.0, 2.0, 2.0, 2.0, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.0, 2.0, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979])
    rrmse = np.array([1.1238069575021878, 1.0239049953712067, 0.9254221370244412, 0.8288643762290191, 0.7226927293756511, 0.6022439411604689, 0.48179515294641406, 0.3613463647261995, 0.24091799698549654, 0.12313893026982561])
    rcvs = np.array([24.520162666223413, 24.47588256858358, 24.42919121561706, 24.40470667935038, 24.300311045176606, 24.1644250876413, 24.058706181807352, 23.961767942955234, 23.870034993318413, 23.804108166835555])
    rsparsities = np.array([0.1, 0.09000000000000001, 0.08000000000000002, 0.07000000000000002, 0.06000000000000002, 0.05000000000000002, 0.040000000000000015, 0.030000000000000013, 0.02000000000000001, 0.01000000000000001])
    #recis = np.array([206.92731536838295, -18.54425141178249, 170.72499878521265, -341.6385700209728, 3.6945184426263276, -0.9255507872094945, -16.909793298762356, 9.438786909564618, -0.9255507872093524, -16.90979329876261, 9.438786909564675, -0.4407888360176852, -11.651139238655997, 2.8139332773482777, 3.908658511183328, -13.805594369619236, -13.805594369619236, 3.9086585111832286, -11.651139238656082, -11.651139238656082, 3.9086585111832286, 1.1571456191802043, 8.941504905599736, 8.941504905599736, 1.1571456191802043, 1.15714561918019, 8.941504905599707])
    ropt_rmse = 0.029503125864994085
    ropt_mean_cv = 30.842354678471292

    """
    print("CV-----------------------")
    for i,(r,rr) in enumerate(zip(clsel.cvs,rcvs)):
        print("line:  ",i,r,rr,r-rr)

    print("sparsities-----------------------")
    for i,(r,rr) in enumerate(zip(clsel.ecis,recis)):
        print("line:  ",i,r,rr,r-rr)

    isok = isclose(rsparsities,clsel.lasso_sparsities) and isclose(rnpoints, npoints) and isclose(rradius, radius) and isclose(rrmse, clsel.rmse) and isclose(rcvs, clsel.cvs) and isclose(recis, clsel.ecis) and isclose(ropt_rmse, clsel.opt_rmse) and isclose(ropt_mean_cv, clsel.opt_mean_cv).all()
    """
    #isok = isclose(rsparsities,clsel.lasso_sparsities) and isclose(rrmse, clsel.rmse) and isclose(rcvs, clsel.cvs) and isclose(ropt_rmse, clsel.opt_rmse) and isclose(ropt_mean_cv, clsel.opt_mean_cv).all()
    rtol=1e-2
    isok = isclose(rsparsities,clsel.lasso_sparsities,rtol=rtol) and isclose(rrmse, clsel.rmse,rtol=rtol) and isclose(rcvs, clsel.cvs,rtol=rtol)

    assert(isok)
