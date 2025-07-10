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
from clusterx.thermodynamics.monte_carlo import MonteCarlo
from clusterx.thermodynamics.monte_carlo import MonteCarloTrajectory
from clusterx.utils import isclose
from clusterx.utils import dict_compare
from clusterx.cli.metropolis import metropolis

import pytest
from ase.data import atomic_numbers as cn
from ase import Atoms
import numpy as np
import os


@pytest.fixture
def plat():
    # parent lattice with single site and one substitution
    cell = [1, 1, 1]
    positions = [[0, 0, 0]]
    pbc = [True, True, False]

    pri = Atoms(["Cu"], positions=positions, cell=cell, pbc=pbc)
    sub = Atoms(["Au"], positions=positions, cell=cell, pbc=pbc)
    plat = ParentLattice(pri, substitutions=[sub], pbc=pbc)
    return plat


@pytest.fixture
def cpool(plat):
    # one-point clusters and nearest neighbor two-point clusters
    return ClustersPool(plat, npoints=[1, 2], radii=[0, 1.1])


@pytest.fixture
def model(plat, cpool):
    # Ising-like model
    corc = CorrelationsCalculator("trigonometric", plat, cpool)
    ecisE = [0., -1.]
    multT = cpool.get_multiplicities()
    cemodel = Model(corc, "energy", ecis=np.multiply(ecisE, multT))
    return cemodel


@pytest.fixture
def wyckoff_sites():
    x, y, z = .185, .304, .116
    return [
        (0, y, z), #24k
        (x, x, x), #16i
        (1/4., 0, 1/2.), #6c
        (1/4., 1/2., 0), #6d
        (0, 0 , 0) #2a
    ]


@pytest.fixture
def cell_a():
    return 10.5148


@pytest.fixture
def pristine_crystal(wyckoff_sites, cell_a):
    return crystal(
        ['Si','Si','Si','Ba','Ba'],
        wyckoff_sites,
        spacegroup=223,
        cellpar=[cell_a, cell_a, cell_a, 90, 90, 90]
    )

@pytest.fixture
def sub_Al(wyckoff_sites, cell_a):
    return crystal(
        ['Al','Al','Al','Ba','Ba'],
        wyckoff_sites,
        spacegroup=223,
        cellpar=[cell_a, cell_a, cell_a, 90, 90, 90]
    )


@pytest.fixture
def sub_X(wyckoff_sites, cell_a):
    return crystal(
        ['X','X','X','Ba','Ba'],
        wyckoff_sites,
        spacegroup=223,
        cellpar=[cell_a, cell_a, cell_a, 90, 90, 90]
    )


@pytest.fixture
def sub_Sr(wyckoff_sites, cell_a):
    return crystal(
        ['Al','Al','Al','Sr','Sr'],
        wyckoff_sites,
        spacegroup=223,
        cellpar=[cell_a, cell_a, cell_a, 90, 90, 90]
    )


@pytest.fixture
def plat_binary(
    pristine_crystal,
    sub_Al
):
    return ParentLattice(
        atoms=pristine_crystal,
        substitutions=[sub_Al],
        pbc=(1,1,1)
    )


@pytest.fixture
def plat_full_sub(
    pristine_crystal,
    sub_Al,
    sub_X,
    sub_Sr
):
    return ParentLattice(
        atoms=pristine_crystal,
        substitutions=[sub_Al, sub_X, sub_Sr],
        pbc=(1,1,1)
    )


@pytest.fixture
def cemodel_binary(pristine_crystal, sub_Al):
    plat = ParentLattice(
        atoms=pristine_crystal,substitutions=[sub_Al],pbc=(1,1,1))

    cpool = ClustersPool(plat)
    cpsc = cpool.get_cpool_scell()
    s = cn["Al"]
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
    cpool = cpool.get_subpool([0,1,2,3,4,5,6,7,9,15])
    ecis = [
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
    corc = CorrelationsCalculator("binary-linear",plat,cpool)
    cemodel = Model(corc, "energy", ecis=np.multiply(ecis, multT))
    return cemodel


@pytest.fixture
def cemodel_full_sub():
    plat = ParentLattice(
        atoms=pristine_crystal, substitutions=[sub_Al,sub_X,sub_Sr])
    scellS2= [(2,0,0),(0,2,0),(0,0,2)]

    cpoolE2 = ClustersPool(plat, npoints=[1], radii=[0])
    corcE2 = CorrelationsCalculator("trigonometric", plat, cpoolE2)

    multT2=cpoolE2.get_multiplicities()
    scellSize=np.prod(np.dot(scellS2,(1,1,1)))
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
        23.10
    ]

    return Model(corcE2, "energy2",ecis=np.multiply(ecisE2, smultT2))


def test_cli(plat, model):
    model_filepath = "model_mc.pickle"
    model.serialize(model_filepath)
    plat_filepath = "plat_mc.json"
    plat.serialize(plat_filepath)
    sc_shape = [8, 8]
    nsubs = {0: [int(np.prod(sc_shape) / 2)]}  # one substitution for the whole supercell
    metropolis(
        # class init arguments
        plat_filepath=plat_filepath,
        model_filepath=model_filepath,
        sc_shape=sc_shape,
        nsubs=nsubs,
        ensemble="canonical",
        sublattice_indices=[],
        chemical_potentials=None,
        models_aux_filepaths=[],
        no_of_swaps=1,
        predict_swap=True,
        error_reset=None,
        filename="trajectory.json",
        # metropolis sampling arguments
        no_of_sampling_steps=100,
        scale_factor=[1.0],
        temperature=1.0,
        boltzmann_constant=1.0,
        initial_decoration=None,
        acceptance_ratio=None,
    )


@pytest.mark.xfail(raises=AssertionError, reason="Ref values not updated")
def test_metropolis_clathrate_Si_Al(
    wyckoff_sites,
    pristine_crystal,
    sub_Al
):
    np.random.seed(10) #setting a seed for the random package for comparible random structures

    # Build the parent lattice
    print("\nSampling in `Si_{46-x} Al_x Ba_{8}`")
    plat = ParentLattice(
        atoms=pristine_crystal,substitutions=[sub_Al],pbc=(1,1,1))

    # Build clusters pool
    #cpool = ClustersPool(plat,r=)
    cpool = ClustersPool(plat)
    cp = cpool._cpool
    cpsc = cpool.get_cpool_scell()
    s = cn["Al"]
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

    cpoolBonds = ClustersPool(plat, npoints=[1], radii=[0])
    corcBonds = CorrelationsCalculator("binary-linear", plat, cpoolBonds)

    multB=[1,24,16,6]
    cemodelBkk=Model(corcBonds, 'bond_kk', ecis=np.multiply(ecisBkk, multB))
    cemodelBii=Model(corcBonds, 'bond_ii', ecis=np.multiply(ecisBii, multB))

    mc = MonteCarlo(
        cemodelE,
        scellE,
        ensemble="canonical",
        nsubs=nsubs,
        models=[cemodelBkk, cemodelBii]
    )

    nmc=50
    # Boltzmann constant in Ha/K
    kb = float(3.16681009610757e-6)
    # temperature in K
    temp = 1000
    info_units = {'temp':'K','kb':'Ha/K','energy':'Ha','scale_factor':None}
    
    print("Samplings steps",nmc)
    print("Temperature",temp)
    scale_factor = []
    traj = mc.metropolis(
        no_of_sampling_steps=nmc,
        scale_factor=scale_factor,
        temperature=temp,
        boltzmann_constant=kb,
        serialize=True,
        info_units=info_units
    )

    steps = traj.get_sampling_step_nos()
    energies = traj.get_energies()

    structure = traj.get_structure(0)
    print("Initial structure: ",structure.decor)

    bondskk1 = traj.get_properties('bond_kk')
    bondsii1 = traj.get_properties('bond_ii')
    print("Bonds kk: ", bondskk1)
    print("Bonds ii: ", bondsii1)

    print("Total energy at sampling step", steps[2], ": ", energies[2])
    struc1 = traj.get_structure_at_step(steps[2])
    print("Decoration at sampling step", steps[2],": ", struc1.decor)
    decoration1 = struc1.decor
    print("Decoration at sampling step", steps[2], "read from atoms object: ", struc1.get_atomic_numbers())
    struc1.serialize(filepath="configuration2.json")

    strucmin = traj.get_lowest_energy_structure()
    print("\nDecoration with the lowest energy: ", strucmin.get_atomic_numbers())
    print("Energy of this structure: ", min(energies))
    strucmin.serialize(filepath="lowest-non-generate-configuration.json")

    print("Configurations accepted at steps: ",steps)
    last_sampling_entry = traj.get_sampling_step_entry_at_step(steps[-1])
    last_structure = traj.get_structure_at_step(steps[-1])

    #rsteps = [0, 1, 2, 3, 4, 6, 10, 11, 16, 17, 18, 26, 27, 34, 37, 38, 44, 45, 47, 48, 50]
    rsteps = [0, 1, 2, 3, 4, 5, 10, 11, 14, 16, 18, 19, 22, 24, 26, 31, 34, 37, 38, 43, 45]
    #renergies = [-77652.59664207128, -77652.61184305252, -77652.62022569243, -77652.61912760629, -77652.62737663009, -77652.63009501049, -77652.63158443688, -77652.64240196907, -77652.64240196907, -77652.64348105107, -77652.64714764676, -77652.64959679516, -77652.64959679516, -77652.65458138083, -77652.66173231734, -77652.65458138083, -77652.65946542152, -77652.6702829537, -77652.66812810961, -77652.67298251796, -77652.66622624162]
    renergies = [-77652.59664207, -77652.61184305, -77652.62022569, -77652.61912761, -77652.62737663, -77652.63941161, -77652.6413147, -77652.6413147, -77652.6413147, -77652.64023562, -77652.63585217, -77652.63585217, -77652.63369732, -77652.62652738, -77652.63709316, -77652.64142517, -77652.63927033, -77652.6445384, -77652.64132329, -77652.64132329, -77652.65963884]
    #rlast_decoration = np.int8([14, 14, 13, 14, 14, 13, 14, 14, 14, 13, 13, 14, 14, 14, 13, 14, 14, 14, 13, 13, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 13, 13, 14, 14, 14, 13, 14, 13, 14, 13, 13, 14, 14, 13, 14, 56, 56, 56, 56, 56, 56, 56, 56])
    rlast_decoration = np.int8([14, 14, 14, 13, 14, 13, 14, 14, 14, 14, 13, 13, 14, 14, 14, 14, 14, 13, 13, 13, 14, 14, 13, 14, 13, 14, 14, 14, 13, 14, 13, 14, 14, 14, 14, 13, 14, 14, 14, 14, 13, 13, 14, 14, 13, 13, 56, 56, 56, 56, 56, 56, 56, 56])
    #rlast_sampling_entry = {'sampling_step_no': 50, 'model_total_energy': -77652.66622624162, 'swapped_positions': [[5, 43]], 'key_value_pairs': {'bond_kk': 2.49116603472051, 'bond_ii': 2.397621971688995}}
    rlast_sampling_entry = {'sampling_step_no': 45, 'energy': -77652.65963884031, 'swapped_positions': [[3, 25]], 'key_value_pairs': {'bond_kk': 2.4897160745744795, 'bond_ii': 2.3909922581598044}}

    rtraj_info={'number_of_sampling_steps': nmc, 'temperature': temp, 'boltzmann_constant': kb}
    rtraj_info.update({'info_units':info_units})
    traj_info={}
    traj_info.update({'number_of_sampling_steps': traj._nmc})
    traj_info.update({'temperature': traj._temperature})
    traj_info.update({'boltzmann_constant': traj._boltzmann_constant})
    if traj._scale_factor is not None:
        traj_info.update({'scale_factor': traj._scale_factor})
    if traj._acceptance_ratio is not None:
        traj_info.update({'scale_factor': traj._acceptance_ratio})
    for key in traj._keyword_arguments:
        traj_info.update({key:traj._keyword_arguments[key]})

    np.testing.assert_allclose(steps, rsteps, rtol=1e-4)
    np.testing.assert_allclose(energies, renergies, rtol=1e-4)
    np.testing.assert_allclose(
        last_structure.decor, rlast_decoration, rtol=1e-4)
    assert dict_compare(
        last_sampling_entry, rlast_sampling_entry, tol=float(1e-7))
    assert dict_compare(traj_info,rtraj_info)
    
    print("before set none", traj.get_properties('bond_kk'))
    print("before set none", traj.get_properties('bond_ii'))
    traj._models =[]
    for i in range(len(traj._trajectory)):
        traj._trajectory[i]['key_value_pairs']={}

    bondskk2 = traj.get_properties('bond_kk')
    bondsii2 = traj.get_properties('bond_ii')
    print(bondskk2)
    print(bondsii2)

    traj.calculate_properties([cemodelBkk,cemodelBii])

    print("Cluster expansion models for the properties: ",[mo.property_name for mo in traj._models])

    #Tests of functions in MonteCarloTrajector
    print("\nTests of functions in MonteCarloTrajector:")
    ids = traj.get_nid_sampling_step(steps[2])
    print(ids)
    prop_at_id = traj.get_property(2,'bond_kk')
    print(prop_at_id)
    stepx = traj.get_nids('bond_kk',2.4772699399288944)
    print(stepx)
    stepx = traj.get_nids('energy',-77652.65458138083)
    print(stepx)

    bondskk = traj.get_properties('bond_kk')
    bondsii = traj.get_properties('bond_ii')
    print("2",bondskk,bondsii)
    

    #rbondskk=[2.4772699399288944, 2.4772699399288944, 2.4772699399288944, 2.4787199000749247, 2.4648238052830997, 2.4648238052830997, 2.4787199000749247, 2.4787199000749247, 2.4787199000749247, 2.4787199000749247, 2.49116603472051, 2.5036121693663045, 2.5036121693663045, 2.49116603472051, 2.4787199000749247, 2.49116603472051, 2.5036121693663045, 2.5036121693663045, 2.4897160745744795, 2.4772699399288944, 2.49116603472051]
    rbondskk = [2.47726994, 2.47726994, 2.47726994, 2.4787199,  2.46482381, 2.46482381, 2.4787199, 2.4787199, 2.4787199, 2.4787199, 2.4787199, 2.4787199, 2.46482381, 2.4787199, 2.4787199, 2.49116603, 2.47726994, 2.46482381, 2.47726994, 2.47726994, 2.48971607]
    #rbondsii = [2.400461156502162, 2.400461156502162, 2.400461156502162, 2.4070908700313525, 2.4099300548444713, 2.4099300548444713, 2.4070908700313525, 2.4070908700313525, 2.4070908700313525, 2.4070908700313525, 2.397621971688995, 2.3881530733466856, 2.3881530733466856, 2.397621971688995, 2.4070908700313525, 2.397621971688995, 2.3881530733466856, 2.3881530733466856, 2.3909922581598044, 2.400461156502162, 2.397621971688995]
    rbondsii = [2.40046116, 2.40046116, 2.40046116, 2.40709087, 2.40993005, 2.40993005, 2.40709087, 2.40709087, 2.40709087, 2.40709087, 2.40709087, 2.40709087, 2.40993005, 2.40709087, 2.40709087, 2.39762197, 2.40046116, 2.40993005, 2.40046116, 2.40046116, 2.39099226]

    np.testing.assert_allclose(bondskk, rbondskk, rtol=1e-4)
    np.testing.assert_allclose(bondsii, rbondsii, rtol=1e-4)

    cp = traj.calculate_average_property(prop_name = 'C_p', no_of_equilibration_steps = 2)
    u = traj.calculate_average_property(prop_name = 'U', no_of_equilibration_steps = 2)
    avg_bond_kk = traj.calculate_average_property(prop_name = 'bond_kk', no_of_equilibration_steps = 2)
    avg_bond_ii = traj.calculate_average_property(prop_name = 'bond_ii', no_of_equilibration_steps = 2)
    u2 = traj.calculate_average_property(prop_name = 'energy', no_of_equilibration_steps = 2)
    averages1 = [cp, u, avg_bond_kk, avg_bond_ii, u2]
    print("averages1",cp,u, avg_bond_kk, avg_bond_ii, u2)
    raverages1 = [7.939914262878631, -77652.64031004661, 2.4779505334667857, 2.4035730628525886, -77652.64031004661]
    
    def test_average(prop_array, **kwargs):
        bondskk = np.average(prop_array[0])
        bondsii = np.average(prop_array[1])
        energy = np.average(prop_array[2])
        temperature = float(kwargs['temperature'])
        return  bondskk, bondsii, energy, (bondsii+bondskk)/(1.0*2), energy/(1.0*temperature)
    
    avg_bond_kk2, avg_bond_ii2, u3, avg_bond, ut = traj.calculate_average_property(average_func=test_average, no_of_equilibration_steps = 2, props_list=['bond_kk','bond_ii','energy'], temperature = traj._temperature )
    averages2 = [avg_bond_kk2, avg_bond_ii2, u3, avg_bond, ut]
    print("averages2", avg_bond_kk2, avg_bond_ii2, u3, avg_bond, ut)
    raverages2 = [2.4779505334667884, 2.403573062852589, -77652.64031004666, 2.4407617981596887, -77.65264031004665]

    np.testing.assert_allclose(averages1, raverages1, rtol=1e-4)
    np.testing.assert_allclose(averages2, raverages2, rtol=1e-4)

    trajx = MonteCarloTrajectory()

    if os.path.isfile("trajectory.json"):
        trajx.read()
        #print(trajx._trajectory[0])
        #print(trajx._scell._plat.get_nsites_per_type())

        energies2 = trajx.get_energies()
        steps2 = trajx.get_sampling_step_nos()

        struc2 = trajx.get_structure_at_step(steps2[2])
        decoration2 = struc2.decor
        last_sampling_entry2 = trajx.get_sampling_step_entry_at_step(steps2[-1])

        isok3 = isclose(renergies, energies2) and isclose(decoration2,decoration1) and isclose(steps2,rsteps) and dict_compare(last_sampling_entry,last_sampling_entry2, tol = float(1.0e-7))

    else:
        isok3 = False

    assert(isok3)


@pytest.mark.xfail(raises=AssertionError, reason="Ref values not updated")
def test_metropolis_clathrate_full_subs(
    pristine_crystal,
    sub_Al,
    sub_X,
    sub_Sr
):
    #Clathrate ternary `Si_{46-x-y} Al_x Vac_y Ba_{8-z} Sr_z`
    print("\nSampling in `Si_{46-x-y} Al_x Vac_y Ba_{8-z} Sr_z`")
    plat2 = ParentLattice(
        atoms=pristine_crystal, substitutions=[sub_Al,sub_X,sub_Sr])

    scellS2= [(2,0,0),(0,2,0),(0,0,2)]
    scellE2 = SuperCell(plat2,scellS2)

    idx_subs = scellE2.get_idx_subs()
    print("Sublattices with corresponding atomic numbers: ",idx_subs)

    cpoolE2 = ClustersPool(plat2, npoints=[1], radii=[0])
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
        23.10
    ]

    cemodelE2=Model(corcE2, "energy2",ecis=np.multiply(ecisE2, smultT2))

    # Sampling in sublattice with index 0 - ternary sampling
    print("Start sampling in sublattice with index 0:")
    mc2 = MonteCarlo(cemodelE2, scellE2, ensemble = "canonical", nsubs = {0:[112,16],1:[0]})

    nmc=30
    # temperature in K
    temp = 600
    print("Samplings steps ",nmc)
    print("Temperature ",temp)
    scale_factor = []
    kb = float(3.16681009610757e-6)

    traj2 = mc2.metropolis(
        no_of_sampling_steps=nmc,
        scale_factor=scale_factor,
        temperature=temp,
        boltzmann_constant=kb,
        serialize=True,
        filename="trajectory-ternary.json"
    )
    steps2 = traj2.get_sampling_step_nos()
    energies2 = traj2.get_energies()
    last_entry2 = traj2.get_sampling_step_entry_at_step(steps2[-1])
    last_structure2 = traj2.get_structure_at_step(steps2[-1])

    traj2.serialize()
    print(last_structure2.decor)

    print("Configurations accepted at steps: ",steps2)

    #rsteps2 = [0, 1, 3, 7, 8, 12, 14, 18, 21, 23, 24, 25, 26, 28, 29, 30]
    rsteps2 = [0, 1, 3, 7, 8, 10, 12, 16, 17, 22, 23, 24, 25, 26, 29]
    #renergies2 = [-634734.608306379, -634734.608306379, -634734.608306379, -634734.620985871, -634734.620985871, -634734.620985871, -634734.620985871, -634734.620985871, -634734.620985871, -634734.633665363, -634734.633665363, -634734.6683063792, -634734.6683063792, -634734.680985871, -634734.680985871, -634734.7156268872]
    renergies2 = [-634734.60830638, -634734.60830638, -634734.60830638, -634734.62098587, -634734.62098587, -634734.62098587, -634734.62098587, -634734.62098587, -634734.62098587, -634734.62098587, -634734.62098587, -634734.62098587, -634734.62098587, -634734.65562689, -634734.65562689]
    #rlast_entry2={'sampling_step_no': 30, 'model_total_energy': -634734.7156268872, 'swapped_positions': [[351, 4]], 'key_value_pairs': {}}
    rlast_entry2={'sampling_step_no': 29, 'energy': -634734.6556268871, 'swapped_positions': [[247, 243]], 'key_value_pairs': {}}
    #rlast_decoration2 = np.int8([14, 14, 14, 14, 13, 14, 14, 14, 14, 14, 14, 13, 14, 14, 14, 14, 14, 13, 14, 13,  0, 13, 14, 14, 14, 13, 14, 13, 13, 13, 14, 13, 13, 13, 14, 13, 13, 14, 14, 14, 14, 14, 14, 13, 14, 13, 56, 56, 56, 56, 56, 56, 56, 56,
    #                             13, 13, 14, 14, 14, 14, 14, 13, 14, 14, 14, 14, 13, 14, 13, 13,  0, 13, 13, 14, 14, 13, 14, 13, 14, 13, 13, 14, 13, 14,  0, 14, 14, 14, 14, 13, 14, 13, 14, 14, 13, 13, 14, 14, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56,
    #                             14, 13, 13, 14, 14, 13, 14,  0, 14, 14, 14, 13, 14,  0, 14, 14, 14, 14, 13, 14, 13, 14, 14,  0, 14, 14, 13, 13, 13, 13, 14, 13, 14, 14, 14, 13, 14, 14, 13, 14,  0, 14, 13, 14, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56,
    #                             14, 13, 14, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 13, 14, 13, 14, 14, 13, 14, 14, 14, 13, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14, 13, 13, 14, 14, 14, 14, 13, 14, 14, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56,
    #                             14, 14, 14, 14, 13, 14, 14, 14, 13, 14, 14, 13, 14, 13, 14, 14, 14, 13, 13, 13, 14, 14, 14, 13, 14, 14, 14,  0, 14,  0,  0, 14, 13, 14, 13, 13, 13, 14, 13, 14, 14, 14, 14, 14, 14, 13, 56, 56, 56, 56, 56, 56, 56, 56,
    #                             13, 14, 13, 13, 13, 14, 14, 14, 14, 13, 14, 13, 14, 14, 14, 14, 14, 14, 13, 14, 14, 14, 14, 14, 13, 14, 14, 14, 14, 14, 14, 13, 14, 13, 14, 13, 14, 13, 13, 14,  0,  0, 14, 14, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56,
    #                             14, 14, 13, 14, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 13, 14, 14, 14, 13, 13, 14, 14,  0, 13, 14, 14, 13, 13, 14, 13, 14, 13, 13, 13, 14, 13, 13, 13, 14, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56,
    #                             14, 14, 14, 13, 14, 14, 13, 14, 14,  0, 14, 14, 13, 14, 13, 14, 14, 14, 14, 14, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14,  0, 14, 14, 13, 14, 13,  0, 14, 13, 13, 14, 13, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56])
    rlast_decoration2 = np.int8([14, 14, 14, 14,  0, 14, 14, 13, 14,  0, 14, 13, 14, 14, 14, 14, 14,  0, 14, 13, 13, 13, 14, 14, 14, 13, 14, 13, 13, 13, 14, 13, 13, 13, 14, 13, 14, 14, 14, 14, 14, 14, 14, 13, 14, 13, 56, 56, 56, 56, 56, 56, 56, 56,
                                 13,  0, 14, 14, 14, 14, 14, 13, 14, 14, 14, 14, 13, 14,  0, 13, 13, 13, 13, 13, 14, 13, 14, 13, 14, 13, 13, 14, 13, 14,  0, 14, 14, 14, 14, 13, 14, 13, 14, 14, 13, 13, 14, 14, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56,
                                 14, 13, 13, 14, 13, 13, 14,  0, 14, 14, 14, 13, 14, 13, 14, 14, 14, 14, 13, 14, 13, 14, 14, 13, 14, 14, 13, 13, 13,  0, 14, 13, 14, 14, 14, 13, 14, 14, 13, 14,  0, 14, 13, 14, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56,
                                 14, 13, 14, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14, 14, 13, 14, 13, 14, 14, 14, 14, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14,  0, 13, 14, 14, 14, 14, 13, 14, 14, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56,
                                 14, 14, 14, 14, 13, 14, 14, 14, 13, 14, 14, 13, 14, 13, 14, 14, 14, 13, 13,  0, 14, 14, 14, 13, 14, 14, 14, 14, 14,  0, 13,  0, 13, 13, 13, 13, 13, 14, 13, 14, 14, 14, 14, 14, 14, 13, 56, 56, 56, 56, 56, 56, 56, 56,
                                 13, 14, 13, 13, 13, 14, 14, 14, 14, 14, 14, 13, 14, 14, 14, 14, 14, 14, 13, 14, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 13, 14, 13, 14, 13, 14, 13, 13, 14,  0,  0, 14, 14, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56,
                                 14, 14, 13, 14, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 13, 14, 14, 13, 13, 14, 13, 14, 13, 13, 14, 14, 13, 14, 14, 14, 14, 13, 13, 13, 14, 13, 13, 13, 14, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56,
                                 14, 14, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 13, 14, 13, 14, 14, 14, 14, 14, 14, 13, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14, 13, 14, 13, 14, 14, 13,  0, 14, 13, 13, 14, 13, 14, 14, 56, 56, 56, 56, 56, 56, 56, 56])

    np.testing.assert_allclose(steps2, rsteps2, rtol=1e-4)
    np.testing.assert_allclose(energies2, renergies2, rtol=1e-4)
    np.testing.assert_allclose(last_structure2.decor, rlast_decoration2, rtol=1e-4)
    assert dict_compare(last_entry2,rlast_entry2, tol=float(1e-7))


@pytest.mark.xfail(reason="maybe this test is not needed anymore")
def test_metropolis_extra():
    # Sampling in the sublattices with indizes 0 and 1 - ternary sampling in sublattice 0 and binary sampling in sublattice 1
    print("\nStart sampling in the two sublattices with indices 0 and 1:")

    nmc=30
    # temperature in K
    temp = 600
    print("Samplings steps",nmc)
    print("Temperature",temp)

    mc3 = MonteCarlo(cemodelE2, scellE2,ensemble = "canonical", nsubs = {0:[112,16],1:[8]})
    traj3 = mc3.metropolis(
        no_of_sampling_steps=nmc,
        scale_factor=scale_factor,
        temperature=temp,
        boltzmann_constant=kb,
        serialize=True,
        filename = "trajectory-multi-lattice.json"
    )
    steps3 = traj3.get_sampling_step_nos()
    energies3 = traj3.get_energies()
    print(steps3)
    print(energies3)

    print("Configurations accepted at steps: ",steps3)

    #rsteps3 = [0, 1, 2, 4, 5, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28]
    rsteps3 = [0, 1, 2, 4, 5, 7, 9, 10, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 27, 28, 29]
    #renergies3 = [-634365.0390243438, -634365.0390243438, -634365.0590243443, -634365.0590243443, -634365.0590243443, -634365.0590243443, -634365.0590243443, -634365.1063448524, -634365.1063448524, -634365.1409858714, -634365.1536653634, -634365.1536653634, -634365.1536653634, -634365.1536653634, -634365.2229473958, -634365.2229473958, -634365.2483063795, -634365.2483063795, -634365.2483063795, -634365.2683063787, -634365.2883063791, -634365.2883063791, -634365.2883063791]
    renergies3 = [-634365.03902434, -634365.03902434, -634365.05902434, -634365.07902434, -634365.07902434, -634365.07902434, -634365.07902434, -634365.12634485, -634365.12634485, -634365.12634485, -634365.12634485, -634365.12634485, -634365.16098587, -634365.16098587, -634365.18634485, -634365.20634486, -634365.20634486, -634365.20634486, -634365.20634486, -634365.20634486, -634365.20634486]

    np.testing.assert_allclose(steps3, rsteps3, rtol=1e-4)
    np.testing.assert_allclose(energies3, renergies3, rtol=1e-4)
