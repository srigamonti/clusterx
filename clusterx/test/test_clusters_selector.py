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

#from clusterx.visualization import plot_optimization_vs_number_of_clusters
#from clusterx.visualization import plot_predictions_vs_target



def test_clusters_selector():
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
    cpool = ClustersPool(plat, npoints=[0,1,2,3,4], radii=[0,0,2.3,2.3,1.42])
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
    strset.serialize(path="test_cluster_selector_structures_set.json", overwrite=True)
    # Get the DATA(comat) + TARGET(energies)
    comat = corrcal.get_correlation_matrix(strset)
    strset.set_calculator(EMT2())
    energies = strset.calculate_property()

    #fitter_model = Fitter(method = "skl_LinearRegression")


    clsel = ClustersSelector(method='linreg', clusters_sets = "size")

    clsel.select_clusters(strset,cpool,"energy")

    #clsets.get_subpool(clsel.get_optimal_clusters())

    cp2 = clsel.optimal_clusters._cpool
    clset=[]
    npoints=[]
    radius=[]
    for i,c in enumerate(cp2):
        clset.append(i)
        npoints.append(c.npoints)
        radius.append(c.radius)

    #plot_optimization_vs_number_of_clusters(clsel)
    #plot_predictions_vs_target(clsel, energies)

    print(clsel.predictions)

    #print(clsel)
    print(clset)
    print(npoints)
    print(radius)
    print(clsel.rmse)
    print(clsel.cvs)
    #print(clsel.opt_ecis)
    print(clsel.opt_rmse)
    print(clsel.opt_mean_cv)

    rclset=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    rnpoints=np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    rradius=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 1.4142135623730951, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979])
    rrmse=np.array([153.2841213331535, 16.901897707237335, 6.700789709328075, 3.999376257086468, 3.9993762570864715, 3.999376257086461, 1.3982627586177953e-12, 2.458069680489386e-13, 3.9889575076243714e-13, 1.1566745888836136e-12, 4.3970985792892635e-13, 4.630264154207033e-13])
    rcvs=np.array([161.35170666647736, 21.25070484696078, 21.19282786602867, 18.253021305083315, 18.253021305083205, 18.253021305083017, 27.15157877280046, 112.1639192203313, 119.88992472711024, 48.55167467287421, 116.65562192812081, 120.96386699529319])
    #recis=np.array([217.50240347215293, -9.31812155582009, 170.06833867676676, -329.5269026624669, -15.744168851528736, 24.772705003474446, -4.329642636893942, -11.671744819665708, -5.984567915150287, 28.909090631366357, -10.002053026806305, 17.14173277787912, -4.32964263689407, -11.671744819665765, -5.984567915150287, 28.909090631366315, -15.74416885152874, 24.772705003474446, -10.002053026806333, 17.141732777879128, -10.002053026806319, 17.141732777879128])
    #ropt_rmse=3.999376257086461
    #ropt_mean_cv=18.253021305083017

    print("CV-----------------------")
    for i,(r,rr) in enumerate(zip(clsel.cvs,rcvs)):
        print("line:  ",i,r,rr,r-rr)

    """
    print("ECI-----------------------")
    for r,rr in zip(clsel.opt_ecis,recis):
        print("line:  ",r,rr,r-rr)

    print("RMSE-----------------------")
    print(clsel.opt_rmse,ropt_rmse)

    print("CV-----------------------")
    print(clsel.opt_mean_cv,ropt_mean_cv)

    print(isclose(rcvs, clsel.cvs))
    """

    #isok = isclose(rclset,clset) and isclose(rnpoints, npoints) and isclose(rradius, radius) and isclose(rrmse, clsel.rmse) and isclose(rcvs, clsel.cvs) and isclose(recis, clsel.opt_ecis) and isclose(ropt_rmse, clsel.opt_rmse) and isclose(ropt_mean_cv, clsel.opt_mean_cv).all()

    isok1 = isclose(rclset,clset) and isclose(rnpoints, npoints) and isclose(rradius, radius) and isclose(rrmse, clsel.rmse) and isclose(rcvs, clsel.cvs)
    assert(isok1)

    opt_cpool = clsel.get_optimal_cpool()
    opt_cpool.serialize(db_name = "cpool.json")

    opt2_cpool = ClustersPool(json_db_filepath = "cpool.json")
    dictcpool = opt2_cpool.get_cpool_dict()

    npoints2 = dictcpool.get('npoints')
    radii2 = dictcpool.get('radii')
    nclusters2 = dictcpool.get('nclusters')
    
    isok2 =  isclose(len(rclset),nclusters2) and isclose(rnpoints, npoints2) and isclose(rradius, radii2)
    assert(isok2)
