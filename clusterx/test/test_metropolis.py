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
import os
import sys

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
    cemodelBkk=Model(corcBonds, 'bond_kk', ecis=np.multiply(ecisBkk, multB))
    cemodelBii=Model(corcBonds, 'bond_ii', ecis=np.multiply(ecisBii, multB))

    mc = MonteCarlo(cemodelE, scellE, nsubs, models = [cemodelBkk, cemodelBii])

    nmc=50
    # Boltzmann constant in Ha/K
    kb = float(3.16681009610757e-6)
    # temperature in K
    temp = 1000

    print("Samplings steps",nmc)
    print("Temperature",temp)

    traj = mc.metropolis([kb,temp], nmc, write_to_db = True)

    steps = traj.get_sampling_step_nos()
    energies = traj.get_model_total_energies()

    #traj.write_to_file()

    structure = traj.get_structure(0)
    print("Initial structure: ",structure.decor)

    bondskk1 = traj.get_model_properties('bond_kk')
    bondsii1 = traj.get_model_properties('bond_ii')
    print(bondskk1)
    print(bondsii1)

    print("Total energy at sampling step", steps[2], ": ", energies[2])
    struc1 = traj.get_structure_at_step(steps[2])
    print("Decoration at sampling step", steps[2],": ", struc1.decor)
    decoration1 = struc1.decor
    print("Decoration at sampling step", steps[2], "read from atoms object: ", struc1.get_atomic_numbers())
    struc1.serialize(fname="configuration2.json")

    strucmin = traj.get_lowest_non_degenerate_structure()
    print("\nDecoration with the lowest energy: ", strucmin.get_atomic_numbers())
    print("Energy of this structure: ", min(energies))
    strucmin.serialize(fname="lowest-non-generate-configuration.json")

    print("Configurations accepted at steps: ",steps)
    last_sampling_entry = traj.get_sampling_step_entry_at_step(steps[-1])
    last_structure = traj.get_structure_at_step(steps[-1])

    rsteps = [0, 1, 2, 3, 4, 6, 10, 11, 16, 17, 18, 26, 27, 34, 37, 38, 44, 45, 47, 48, 50]
    renergies = [-77652.59664207128, -77652.61184305252, -77652.62022569243, -77652.61912760629, -77652.62737663009, -77652.63009501049, -77652.63158443688, -77652.64240196907, -77652.64240196907, -77652.64348105107, -77652.64714764676, -77652.64959679516, -77652.64959679516, -77652.65458138083, -77652.66173231734, -77652.65458138083, -77652.65946542152, -77652.6702829537, -77652.66812810961, -77652.67298251796, -77652.66622624162]
    rlast_decoration = np.int8([14, 14, 13, 14, 14, 13, 14, 14, 14, 13, 13, 14, 14, 14, 13, 14, 14, 14, 13, 13, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 13, 13, 14, 14, 14, 13, 14, 13, 14, 13, 13, 14, 14, 13, 14, 56, 56, 56, 56, 56, 56, 56, 56])
    rlast_sampling_entry = {'sampling_step_no': 50, 'model_total_energy': -77652.66622624162, 'swapped_positions': [[5, 43]], 'key_value_pairs': {'bond_kk': 2.49116603472051, 'bond_ii': 2.397621971688995}}

    isok1 = isclose(rsteps,steps) and isclose(renergies, energies) and isclose(rlast_decoration,last_structure.decor) and dict_compare(last_sampling_entry, rlast_sampling_entry, tol=float(1e-7) )
    assert(isok1)

    traj._models =[]
    for i in range(len(traj._trajectory)):
        traj._trajectory[i]['key_value_pairs']={}

    bondskk2 = traj.get_model_properties('bond_kk')
    bondsii2 = traj.get_model_properties('bond_ii')
    print(bondskk2)
    print(bondsii2)

    traj.calculate_model_properties([cemodelBkk,cemodelBii])

    print("Cluster expansion models for the properties: ",[mo.property for mo in traj._models])

    #Tests of functions in MonteCarloTrajector
    print("\nTests of functions in MonteCarloTrajector:")
    ids = traj.get_id_sampling_step(steps[2])
    print(ids)
    prop_at_id = traj.get_model_property(2,'bond_kk')
    print(prop_at_id)
    stepx = traj.get_id('bond_kk',2.4772699399288944)
    print(stepx)
    stepx = traj.get_id('model_total_energy',-77652.65458138083)
    print(stepx)

    bondskk = traj.get_model_properties('bond_kk')
    bondsii = traj.get_model_properties('bond_ii')

    rbondskk = [2.4772699399288944, 2.4772699399288944, 2.4772699399288944, 2.4787199000749247, 2.4648238052830997, 2.4648238052830997, 2.4787199000749247, 2.4787199000749247, 2.4787199000749247, 2.4787199000749247, 2.49116603472051, 2.5036121693663045, 2.5036121693663045, 2.49116603472051, 2.4787199000749247, 2.49116603472051, 2.5036121693663045, 2.5036121693663045, 2.4897160745744795, 2.4772699399288944, 2.49116603472051]
    rbondsii = [2.400461156502162, 2.400461156502162, 2.400461156502162, 2.4070908700313525, 2.4099300548444713, 2.4099300548444713, 2.4070908700313525, 2.4070908700313525, 2.4070908700313525, 2.4070908700313525, 2.397621971688995, 2.3881530733466856, 2.3881530733466856, 2.397621971688995, 2.4070908700313525, 2.397621971688995, 2.3881530733466856, 2.3881530733466856, 2.3909922581598044, 2.400461156502162, 2.397621971688995]

    isok2 = isclose(rbondskk,bondskk) and isclose(rbondsii,bondsii)
    assert(isok2)

    trajx = MonteCarloTrajectory()

    if os.path.isfile("trajectory.json"):
        trajx.read()
        #print(trajx._trajectory[0])
        #print(trajx._scell._plat.get_nsites_per_type())

        energies2 = trajx.get_model_total_energies()
        steps2 = trajx.get_sampling_step_nos()

        struc2 = trajx.get_structure_at_step(steps2[2])
        decoration2 = struc2.decor
        last_sampling_entry2 = trajx.get_sampling_step_entry_at_step(steps2[-1])

        isok3 = isclose(renergies, energies2) and isclose(decoration2,decoration1) and isclose(steps2,rsteps) and dict_compare(last_sampling_entry,last_sampling_entry2, tol = float(1.0e-7))

    else:
        isok3 = False

    assert(isok3)

    #Clathrate ternary `Si_{46-x-y} Al_x Vac_y Ba_{8-z} Sr_z`
    print("\nSampling in `Si_{46-x-y} Al_x Vac_y Ba_{8-z} Sr_z`")
    sub2 = crystal(['X','X','X','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])
    sub3 = crystal(['Si','Si','Si','Sr','Sr'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])

    plat2 = ParentLattice(atoms=pri,substitutions=[sub,sub2,sub3])

    scellS2= [(2,0,0),(0,2,0),(0,0,2)]
    scellE2 = SuperCell(plat2,scellS2)

    struc = scellE2.gen_random({0:[112,16], 1:[0]})

    idx_subs = scellE2.get_idx_subs()
    print("Sublattices with corresponding atomic numbers: ",idx_subs)

    cpoolE2 = ClustersPool(plat2, npoints=[0,1], radii=[0,0])
    corcE2 = CorrelationsCalculator("trigonometric", plat2, cpoolE2)

    multT2=cpoolE2.get_multiplicities()
    print("Cluster multiplicities: ",multT2)
    print("Corresponding radii: ",cpoolE2.get_all_radii())
    print("Corresponding points: ",cpoolE2.get_all_npoints())

    scellSize=np.prod(np.dot(scellS2,(1,1,1)))
    print("Multiplications of the parent cell: ", scellSize)
    smultT2=np.zeros(len(multT2))
    for i in range(0,len(multT2)):
        smultT2[i]=int(multT2[i]*scellSize)

    ecisE2 = [
        -78407.325,
        23.16,
        23.15,
        23.14,
        23.13,
        23.12,
        23.11,
        23.10,
        23.09
    ]

    cemodelE2=Model(corcE2, "energy2",ecis=np.multiply(ecisE2, smultT2))

    # Sampling in sublattice with index 0 - ternary sampling
    print("Start sampling in sublattice with index 0:")
    mc2 = MonteCarlo(cemodelE2, scellE2, {0:[112,16],1:[0]}, sublattice_indices=[0], filename = "trajectory-ternary.json")

    nmc=30
    # temperature in K
    temp = 600
    print("Samplings steps ",nmc)
    print("Temperature ",temp)

    traj2 = mc2.metropolis([kb,temp], nmc)

    steps2 = traj2.get_sampling_step_nos()
    energies2 = traj2.get_model_total_energies()
    last_entry2 = traj2.get_sampling_step_entry_at_step(steps2[-1])
    last_structure2 = traj2.get_structure_at_step(steps2[-1])

    traj2.write_to_file()

    print("Configurations accepted at steps: ",steps2)

    rsteps2 = [0, 1, 3, 7, 8, 12, 14, 18, 21, 23, 24, 25, 26, 28, 29, 30]
    renergies2 = [-634734.608306379, -634734.608306379, -634734.608306379, -634734.620985871, -634734.620985871, -634734.620985871, -634734.620985871, -634734.620985871, -634734.620985871, -634734.633665363, -634734.633665363, -634734.6683063792, -634734.6683063792, -634734.680985871, -634734.680985871, -634734.7156268872]

    rlast_entry2={'sampling_step_no': 30, 'model_total_energy': -634734.7156268872, 'swapped_positions': [[351, 4]], 'key_value_pairs': {}}
    rlast_decoration2 = np.int8([14, 14, 14, 14, 13, 14, 14, 14, 14, 14, 14, 13, 14, 14, 14, 14, 14, 13, 14, 13,  0, 13, 14, 14, 14, 13, 14, 13, 13, 13, 14, 13, 13, 13, 14, 13, 13, 14, 14, 14, 14, 14, 14, 13, 14, 13, 56, 56, 56, 56, 56, 56, 56, 56,
                                 13, 13, 14, 14, 14, 14, 14, 13, 14, 14, 14, 14, 13, 14, 13, 13,  0, 13, 13, 14, 14, 13, 14, 13, 14, 13, 13, 14, 13, 14,  0, 14, 14, 14, 14, 13, 14, 13, 14, 14, 13, 13, 14, 14, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56,
                                 14, 13, 13, 14, 14, 13, 14,  0, 14, 14, 14, 13, 14,  0, 14, 14, 14, 14, 13, 14, 13, 14, 14,  0, 14, 14, 13, 13, 13, 13, 14, 13, 14, 14, 14, 13, 14, 14, 13, 14,  0, 14, 13, 14, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56,
                                 14, 13, 14, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 13, 14, 13, 14, 14, 13, 14, 14, 14, 13, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14, 13, 13, 14, 14, 14, 14, 13, 14, 14, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56,
                                 14, 14, 14, 14, 13, 14, 14, 14, 13, 14, 14, 13, 14, 13, 14, 14, 14, 13, 13, 13, 14, 14, 14, 13, 14, 14, 14,  0, 14,  0,  0, 14, 13, 14, 13, 13, 13, 14, 13, 14, 14, 14, 14, 14, 14, 13, 56, 56, 56, 56, 56, 56, 56, 56,
                                 13, 14, 13, 13, 13, 14, 14, 14, 14, 13, 14, 13, 14, 14, 14, 14, 14, 14, 13, 14, 14, 14, 14, 14, 13, 14, 14, 14, 14, 14, 14, 13, 14, 13, 14, 13, 14, 13, 13, 14,  0,  0, 14, 14, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56,
                                 14, 14, 13, 14, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 13, 14, 14, 14, 13, 13, 14, 14,  0, 13, 14, 14, 13, 13, 14, 13, 14, 13, 13, 13, 14, 13, 13, 13, 14, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56,
                                 14, 14, 14, 13, 14, 14, 13, 14, 14,  0, 14, 14, 13, 14, 13, 14, 14, 14, 14, 14, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14,  0, 14, 14, 13, 14, 13,  0, 14, 13, 13, 14, 13, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56])

    isok4 = isclose(rsteps2,steps2) and isclose(renergies2, energies2) and isclose(last_structure2.decor,rlast_decoration2) and dict_compare(last_entry2,rlast_entry2, tol=float(1.0e-7))
    assert(isok4)

    # Sampling in the sublattices with indizes 0 and 1 - ternary sampling in sublattice 0 and binary sampling in sublattice 1
    print("\nStart sampling in the two sublattices with indices 0 and 1:")
    print("Samplings steps",nmc)
    print("Temperature",temp)

    mc3 = MonteCarlo(cemodelE2, scellE2, {0:[112,16],1:[8]}, filename = "trajectory-multi-lattice.json", last_visited_structure_name = "last-visited-structure-mc-multi-lattice.json")
    traj3 = mc3.metropolis([kb,temp], nmc, write_to_db = True)

    steps3 = traj3.get_sampling_step_nos()
    energies3 = traj3.get_model_total_energies()

    #traj3.write_to_file()

    print("Configurations accepted at steps: ",steps3)

    rsteps3 = [0, 1, 2, 4, 5, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28]
    renergies3 = [-634365.0390243438, -634365.0390243438, -634365.0590243443, -634365.0590243443, -634365.0590243443, -634365.0590243443, -634365.0590243443, -634365.1063448524, -634365.1063448524, -634365.1409858714, -634365.1536653634, -634365.1536653634, -634365.1536653634, -634365.1536653634, -634365.2229473958, -634365.2229473958, -634365.2483063795, -634365.2483063795, -634365.2483063795, -634365.2683063787, -634365.2883063791, -634365.2883063791, -634365.2883063791]

    isok5 = isclose(rsteps3,steps3) and isclose(renergies3, energies3)
    assert(isok5)
