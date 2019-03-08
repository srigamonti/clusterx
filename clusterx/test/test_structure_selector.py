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
from clusterx.structure_selector import StructureSelector
from ase import Atoms
import numpy as np
import sys
from ase import Atoms

def test_structure_selector():

    """Test structure selection routine
       Tests all subroutines and some variables of the StructureSelector class
    """

    cell = [[1,0,0],
            [0,1,0],
            [0,0,5]]
    positions = [[0,0,0]]
    pbc = [True,True,False]

    pri = Atoms(['H'], positions=positions, cell=cell, pbc=pbc)
    su1 = Atoms(['C'], positions=positions, cell=cell, pbc=pbc)

    plat = ParentLattice(pri,substitutions=[su1],pbc=pbc)
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
    """
    #cpool = ClustersPool(plat, npoints=[0,1,2,3,4], radii=[0,0,2.3,1.42,1.42])
    cpool = ClustersPool(plat, npoints=[1,2], radii=[0,1.1])

    cpool.write_clusters_db(cpool.get_cpool(),cpool.get_cpool_scell(),"cpool.json")
    corrcal = CorrelationsCalculator("trigonometric", plat, cpool)

    scell = SuperCell(plat,np.array([(3,0,0),(0,3,0),(0,0,1)]))
    #training_set = StructuresSet(plat, filename="test_structure_selection_training_set.json")
    training_set = StructuresSet(plat)
    ntrainingstr = 14

    for idx in range(ntrainingstr):
        training_set.add_structure(scell.gen_random(nsubs={0:[5]}))

    training_set.serialize(path="test_structure_selection_training_set.json", overwrite=True)

    structure_selector = StructureSelector(cluster_pool = cpool, training_set = training_set)

    comat = corrcal.get_correlation_matrix(training_set)
    covariance_matrix_inv = np.dot(comat.T, comat)

    'Test 1'
    covariance_correct = np.allclose(np.identity(len(cpool)), np.dot(structure_selector._covariance_matrix, covariance_matrix_inv))
    if not(covariance_correct):
        print('Covariance was NOT correctly obtained.')
        sys.exit()

    'Test 2'
    test = False
    try:
        print('Do not worry about the next error msg.')
        structure_selector.set_candidate_set(None)
        test = True
    except:
        pass

    if(test):
        print('set_candidate_set sets candidate_set to something thats no structure set.')
        sys.exit()

    'Test 3'
    method_list = ['averagedConcentration', 'infiniteCrystalFiniteClusters', 'byConcentration', 'vdWalleAndCeder']
    for method in method_list:
        tau = structure_selector.calculate_population_variance(domain_calculation_method = method, concentration = 0.2)

        if not(isinstance(tau, float)):
            print('tau is no float!')
            sys.exit()
        elif(tau < 0):
            print('tau is negative!')
            sys.exit()

    'Test 4'
    test_str = 'global_averagedConcentration'
    'TODO: WHAT IS THE DIIFERENCE BETWEEN is AND ==???'
    if(test_str.replace('global_','') is 'averagedConcentration'):
        print(test_str[7:])
        print('Test for transmitting domain_matrix_method failed!')
        sys.exit()

    test_str = 'global_averagedConcentration'
    global_or_greedy = test_str[:6]
    if not (global_or_greedy == 'global'):
        print('Second transmitting test failed as well.')
        sys.exit()

    'Test 5'

    #candidate_set = StructuresSet(plat, filename="test_structure_selection_candidate_set.json")
    candidate_set = StructuresSet(plat)
    ncandidatestr = 6
    for idx in range(ncandidatestr):
        candidate_set.add_structure(scell.gen_random(nsubs={0:[5]}))

    candidate_set.serialize(path="test_structure_selection_candidate_set.json", overwrite=False, rm_vac=False)

    structure_selector.set_candidate_set(candidate_set)
    method_list = ['global_averagedConcentration', 'global_infiniteCrystalFiniteClusters', 'global_byConcentration', 'global_vdWalleAndCeder', 'greedy']
    for method in method_list:
        structure = candidate_set.get_structure(structure_selector.select_structure(method = method, concentration = 0.2))
        if not(isinstance(structure, Structure)):
            print('Returned structure is not structure object! "structure_selector.select_structure()"')
            sys.exit()

    'Test 6'
    'Does this selection optimally reduce the selector criterion with the tau?'
    method_list = ['global_averagedConcentration', 'global_infiniteCrystalFiniteClusters', 'global_byConcentration', 'global_vdWalleAndCeder', 'greedy']
    for method in method_list:
        structure_idx = structure_selector.select_structure(method = method, concentration = 0.2)
        if(method == 'greedy'):
            print('Ignore next error msgs.')

        try:
            calculate_tau_method = method[7:]

            tau = np.zeros(ncandidatestr)
            for candidate_idx in range(ncandidatestr):
                candidate = candidate_set.get_structure(candidate_idx)
                dummy_file_str = "test_structure_selection_dummy_set"+str(candidate_idx)+".json"
                dummy_set = StructuresSet(plat)
                dummy_set.add_structures(structures = "test_structure_selection_training_set.json")
                dummy_set.add_structure(candidate)
                dummy_set.serialize(path="test_structure_selection_dummy_set"+str(candidate_idx)+".json", overwrite=True)
                dummy_structure_selector = StructureSelector(cluster_pool = cpool, training_set = dummy_set)
                tau[candidate_idx] = dummy_structure_selector.calculate_population_variance(domain_calculation_method = calculate_tau_method, concentration = 0.2)

            structure_chosen = np.argmin(tau)

            if not(structure_chosen == structure_idx):
                print('structure selection failed. Criteria are not optimally reducing prediction variance according to themselves.')
                print(structure_chosen)
                print(structure_idx)
                print(method)
                sys.exit()
            else:
                print('Test succeeded')

        except:
            pass


    'Test 7'

    test = True
    try:
        print('Do not worry about the next error msg.')
        structure_selector._calculate_domain_matrix()
        test = False
    except:
        pass

    if not(test):
        print('Calculate domain matrix reacts unexpectedly.')
        sys.exit()

    test = True
    try:
        print('Do not worry about the next error msg.')
        structure_selector._calculate_domain_matrix(method = 'byConcentration')
        test = False
    except:
        pass

    if not(test):
        print('Calculate domain matrix reacts unexpectedly.')
        sys.exit()

    domain_matrix = structure_selector._calculate_domain_matrix(method = 'byConcentration', concentration = 0.2)
    print(domain_matrix)
