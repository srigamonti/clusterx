import clusterx as c
import subprocess
from ase.spacegroup import crystal
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.clusters.cluster import Cluster
from clusterx.correlations import CorrelationsCalculator
from clusterx.model import Model
from clusterx.monte_carlo import MonteCarlo
from clusterx.monte_carlo import MonteCarloTrajectory
from clusterx.utils import isclose
from clusterx.utils import dict_compare

from ase.data import atomic_numbers as cn
from ase import Atoms
import numpy as np


def test_metropolis():

    subprocess.call(["rm","-f","test_clathrate_mc-cluster_orbit.json"])
    subprocess.call(["rm","-f","test_clathrate_mc-cpool.json"])

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
    
    #mult=cpool.get_multiplicities()
    #m=[]
    #for cl in cpoolE.get_cpool():
    #    m.append(cl.get_multiplicity(sc_sym["rotations"],sc_sym["translations"]))    
    #print(mult)
    #multE=cpoolE.get_multiplicities()
    #print(multE)
    
    multT = [1,24,16,6,12,8,48,24,24,24]
    corcE = CorrelationsCalculator("binary-linear",plat,cpoolE)
    scellS= [(1,0,0),(0,1,0),(0,0,1)]
    scellE = SuperCell(plat,scellS)

    cemodelE=Model(corcE, ecisE, multT)

    #struc = scellE.gen_random({0:[16]})
    mc = MonteCarlo(cemodelE, scellE, {0:[16]})

    nmc=50

    # Boltzmann constant in Ha/K
    kb = float(3.16681009610757e-6)
    # temperature in K
    temp = 1000

    print("Samplings steps",nmc)
    print("Temperature",temp)
    
    traj = mc.metropolis([kb,temp], nmc)

    steps = traj.get_sampling_step_nos()
    energies = traj.get_model_total_energies()
    print(traj.get_decoration_at_step(steps[1]))
    
    print(steps[-1])
    last_decoration = traj.get_sampling_step_entries_at_step(steps[-1])
    #print(decoration)

    rsteps = [0, 1, 2, 3, 4, 6, 10, 11, 16, 17, 18, 26, 27, 34, 37, 38, 44, 45, 47, 48, 50]
    renergies = [-77652.59664207128, -77652.61184305252, -77652.62022569243, -77652.61912760629, -77652.62737663009, -77652.63009501049, -77652.63158443688, -77652.64240196907, -77652.64240196907, -77652.64348105107, -77652.64714764676, -77652.64959679516, -77652.64959679516, -77652.65458138083, -77652.66173231734, -77652.65458138083, -77652.65946542152, -77652.6702829537, -77652.66812810961, -77652.67298251796, -77652.66622624162]
    rlast_decoration = {'sampling_step_no': 50,
                        'model_total_energy': -77652.66622624162,
                        'decoration': np.int8([14, 14, 13, 14, 14, 13, 14, 14, 14, 13, 13, 14, 14, 14, 13, 14, 14, 14, 13, 13, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 13, 13, 14, 14, 14, 13, 14, 13, 14, 13, 13, 14, 14, 13, 14, 56, 56, 56, 56, 56, 56, 56, 56]),
                        'key_value_pairs': {}}
#                            [14, 14, 13, 13, 14, 14, 13, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 13, 14, 14, 13, 13, 14, 14, 13, 13, 14, 14, 14, 14, 13, 13, 14, 13, 13, 14, 56, 56, 56, 56, 56, 56, 56, 56]
#                            [14, 14, 13, 14, 14, 13, 14, 14, 14, 13, 13, 14, 14, 14, 13, 14, 14, 14, 13, 13, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 13, 13, 14, 14, 14, 13, 14, 13, 14, 13, 13, 14, 14, 13, 14, 56, 56, 56, 56, 56, 56, 56, 56]

    
    isok1 = isclose(rsteps,steps) and isclose(renergies, energies) and dict_compare(last_decoration,rlast_decoration)
    assert(isok1)


    ecisBkk= [
        2.45499482287556,
        0.008755635590555,
        -0.00369049905517,
        -0.00514045920119
    ]

    ecisBii= [
        2.3096951176787144,
        0.0020040504912059,
        0.0114729488335313,
        0.004843235304331
    ]
            
    cpoolBonds = ClustersPool(plat, npoints=[0,1], radii=[0,0])
    corcBonds = CorrelationsCalculator("binary-linear", plat, cpoolBonds)

    multB=[1,24,16,6]
    cemodelBkk=Model(corcBonds, ecisBkk, multB, prop = 'bond_kk')
    cemodelBii=Model(corcBonds, ecisBii, multB, prop = 'bond_ii')

    traj.calculate_model_properties([cemodelBkk,cemodelBii,cemodelE])

    traj.write_to_file() #filename = 'trajectory-bonds.json'

    bondskk = traj.get_model_properties('bond_kk')
    bondsii = traj.get_model_properties('bond_ii')

    rbondskk = [2.4772699399288944, 2.4772699399288944, 2.4772699399288944, 2.4787199000749247, 2.4648238052830997, 2.4648238052830997, 2.4787199000749247, 2.4787199000749247, 2.4787199000749247, 2.4787199000749247, 2.49116603472051, 2.5036121693663045, 2.5036121693663045, 2.49116603472051, 2.4787199000749247, 2.49116603472051, 2.5036121693663045, 2.5036121693663045, 2.4897160745744795, 2.4772699399288944, 2.49116603472051]
    rbondsii = [2.400461156502162, 2.400461156502162, 2.400461156502162, 2.4070908700313525, 2.4099300548444713, 2.4099300548444713, 2.4070908700313525, 2.4070908700313525, 2.4070908700313525, 2.4070908700313525, 2.397621971688995, 2.3881530733466856, 2.3881530733466856, 2.397621971688995, 2.4070908700313525, 2.397621971688995, 2.3881530733466856, 2.3881530733466856, 2.3909922581598044, 2.400461156502162, 2.397621971688995]

    isok2 = isclose(rbondskk,bondskk) and isclose(rbondsii,bondsii)
    
    assert(isok2)    
