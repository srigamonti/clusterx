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
    cpool = ClustersPool(plat, npoints=[0,1,2,3,4], radii=[0,2.3,2.3,2.3,1.42])
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

    #fitter_model = Fitter(method = "skl_LinearRegression")


    clsel = ClustersSelector('linreg', cpool, clusters_sets = "size", alpha0 = 2.3)
    #clsel = ClustersSelector('linreg', cpool, clusters_sets = "combinations", nclmax=2)
    #clsel = ClustersSelector('linreg', cpool, clusters_sets = "size+combinations", nclmax = 2, set0 = [2,1])
    clsel.select_clusters(comat,energies)

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

    #print(clsel)
    print(clset)
    print(npoints)
    print(radius)
    print(clsel.rmse)
    print(clsel.cvs)
    print(clsel.ecis)
    print(clsel.opt_rmse)
    print(clsel.opt_mean_cv)

    rclset=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    rnpoints=np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    rradius=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 1.4142135623730951, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.23606797749979, 2.23606797749979, 2.23606797749979, 2.23606797749979])
    rrmse=np.array([153.2841213331535, 16.901897707237335, 6.700789709328075, 3.9993762570864653, 3.999376257086457, 3.9993762570864604, 4.332708758890144e-13, 2.3963837243895845e-13, 2.5095842348474555e-13, 1.5137228741553587e-13, 2.0922437301566996e-13, 2.3512953151993736e-13])
    rcvs=np.array([161.35170666647736, 21.25070484696078, 21.192827866028615, 18.253021305083323, 18.253021305083177, 18.253021305083085, 26.826078769571474, 101.35271637863913, 107.52614106050373, 46.94157574864736, 107.7467725586574, 122.45989146285723])
    recis=np.array([217.50240347215268, -9.318121555820007, 170.0683386767669, -329.52690266246697, -15.744168851529114, 24.772705003474496, -4.329642636893923, -11.671744819665836, -5.984567915150468, 28.909090631366023, -10.002053026806427, 17.14173277787906, -4.329642636893866, -11.67174481966575, -5.984567915150468, 28.909090631366052, -15.744168851529029, 24.772705003474496, -10.00205302680642, 17.141732777879074, -10.00205302680642, 17.141732777879074])
    ropt_rmse=3.9993762570864604
    ropt_mean_cv=18.253021305083085

    print("CV-----------------------")
    for i,(r,rr) in enumerate(zip(clsel.cvs,rcvs)):
        print("line:  ",i,r,rr,r-rr)

    """
    print("ECI-----------------------")
    for r,rr in zip(clsel.ecis,recis):
        print("line:  ",r,rr,r-rr)

    print("RMSE-----------------------")
    print(clsel.opt_rmse,ropt_rmse)

    print("CV-----------------------")
    print(clsel.opt_mean_cv,ropt_mean_cv)

    print(isclose(rcvs, clsel.cvs))
    """
    
    isok = isclose(rclset,clset) and isclose(rnpoints, npoints) and isclose(rradius, radius) and isclose(rrmse, clsel.rmse) and isclose(rcvs, clsel.cvs) and isclose(recis, clsel.ecis) and isclose(ropt_rmse, clsel.opt_rmse) and isclose(ropt_mean_cv, clsel.opt_mean_cv).all()
    assert(isok)
