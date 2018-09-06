import clusterx as c
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from ase import Atoms
import numpy as np
import sys
from ase import Atoms

class StructureSelector():
    """Structure selector class

    **Parameters**
    
    Compulsory:
    ``cluster_pool``: ClustersPool() object
       The optimal set of clusters for the training data.
    ``training_set``: StructuresSet() object
       Contains the current training set
    Optional:
    ``candidate_set``: StructuresSet() object
       Pool of new structures, that might be selected to enter the training set.
       If it remains 'None', the class can only assess the training set.
    ``correlations_calculator``: CorrelationsCalculator object
       Used to calculate the correlations of structures and clusters.
       IF THIS IS LEFT EMPTY (or set to None) the correlations are calculated with 
       a trigonometric basis.      

    """

    def __init__(self, cluster_pool = None, training_set = None, candidate_set = None, correlations_calculator = None):
        'Barricade against bad input:'
        if not(isinstance(cluster_pool, ClustersPool)):
            print('No cluster pool handed to StructureSelector.__init__()')
            sys.exit()
        if not(isinstance(training_set, StructuresSet)):
            print('No training set handed to StructureSelector.__init__()')
            sys.exit()
        if not(candidate_set is None or isinstance(candidate_set, StructuresSet)):
            print('Wrong candidate set handed to StructureSelector.__init__()')
            sys.exit()
        if not(correlations_calculator is None or isinstance(correlations_calculator, CorrelationsCalculator)):
            print('Wrong correlations calculator handed to StructureSelector.__init__()')
            sys.exit()

        self._cluster_pool = cluster_pool
        self._training_set = training_set
        self._candidate_set = candidate_set

        if(correlations_calculator is None):
            self._correlations_calculator = CorrelationsCalculator("trigonometric", training_set.get_parent_lattice(), cluster_pool)
        else:
            self._correlations_calculator = correlations_calculator

        'Calculate covariance matrix'
        correlation_matrix = self._correlations_calculator.get_correlation_matrix(self._training_set)
        self._covariance_matrix = np.linalg.inv(np.dot(correlation_matrix.T, correlation_matrix))

    def set_candidate_set(self, new_candidate_set):
        if not(isinstance(new_candidate_set, StructuresSet)):
            print('No candidate set handed to set_candidate_set. StructureSelector()')
            print('StructureSelectorError')
            sys.exit()
        self._candidate_set = new_candidate_set

    def calculate_variance_multiplier_expectation(self, domain_calculation_method = 'averagedConcentration', concentration = None):
        
        '''
        Computes 'goodness' of the training set

        In the paper Mueller2010 PRB 82, 184107 the expectation of the multiplier to the variance
        to the average prediction error is called #TAU#
        This #TAU# is computed here.
        '''

        domain_matrix = self._calculate_domain_matrix(method = domain_calculation_method, concentration = concentration)
        covariance_matrix = self._covariance_matrix

        'compute Hadamard product and sum'
        variance_multiplier_expectation = np.sum(np.sum(np.multiply(domain_matrix, covariance_matrix)))

        return variance_multiplier_expectation
        
    def select_structure(self, method = None, concentration = None):      
        if(self._candidate_set is None):
            print('There are no candidate structures. select_structures()')
            sys.exit()
        if not(isinstance(method, str) and (method[:6] == 'greedy' or method[:6] == 'global')):
            text_file = open('somefile.txt', 'w')
            text_file.write()
            print('No method inserted to select_structures. select_structures()')
            sys.exit()

        '''Choose a method for the selection of the new structure.
           method can be "global_#OPTION#" or "greedy_#OPTION#"
           For global, the options can be 
           'averagedConcentration', 'infiniteCrystalFiniteClusters', 'byConcentration', 'vdWalleAndCeder'
           'byConcentration' needs concentration in floats
        '''
        
        global_or_greedy = method[:6]
        covariance_matrix = self._covariance_matrix

        if(global_or_greedy == 'global'):
            domain_calculation_method = method.replace("global_","")
            domain_matrix = self._calculate_domain_matrix(method = domain_calculation_method, concentration = concentration)

            squared_covariance = np.dot(covariance_matrix, covariance_matrix)

            candidate_correlations = self._correlations_calculator.get_correlation_matrix(self._candidate_set)

            number_candidates ,number_clusters = candidate_correlations.shape

            'score vector for each structure in selection'
            candidate_scores = np.zeros(number_candidates)

            for candidate_idx in range(number_candidates):
                candidate_corr = candidate_correlations[candidate_idx,:]
                left_dyad = np.dot(squared_covariance, candidate_corr)
                left_matrix = np.outer(left_dyad, candidate_corr)
                candidate_scores[candidate_idx] = 1/(1+np.dot(candidate_corr, np.dot(covariance_matrix, candidate_corr)))*np.sum(np.sum(np.multiply(left_matrix, domain_matrix)))

            return np.argmax(candidate_scores)

        elif(global_or_greedy == 'greedy'):
            candidate_correlations = self._correlations_calculator.get_correlation_matrix(self._candidate_set)

            number_candidates ,number_clusters = candidate_correlations.shape

            'score vector for each structure in selection'
            candidate_scores = np.zeros(number_candidates)

            for candidate_idx in range(number_candidates):
                candidate_corr= candidate_correlations[candidate_idx,:]
                candidate_scores[candidate_idx] = np.dot(candidate_corr, np.dot(covariance_matrix, candidate_corr))

            return np.argmax(candidate_scores)

        else:
            print('No valid selector chosen. select_structure() in StructureSelector.')
            sys.exit()
    
    def _calculate_domain_matrix(self, method = None, concentration = None):
        if(method is None):
            print('Error in StructureSelector(), _calculate_domain_matrix(). No method for calculation chosen!')
            sys.exit()
        method_list = ['averagedConcentration', 'infiniteCrystalFiniteClusters', 'byConcentration', 'vdWalleAndCeder']
        if not(method is s for s in method_list):
            print('Error in StructureSelector(), _calculate_domain_matrix(). No valid option for calculation chosen!')
            sys.exit()

        '''
        Calculate domain matrix from Mueller2010 paper PRB 82, 184107
        '''
        
        npoints = self._cluster_pool.get_all_npoints()
        number_clusters = len(npoints)
        domain_matrix = np.zeros((number_clusters, number_clusters), dtype = float)

        'Domain matrix uniformly averaged over concentration for structures in infinite crystal'
        if(method == 'averagedConcentration'):
            'Number of occupations a cluster depends on'

            for rowIdx in range(number_clusters):
                for colIdx in range(number_clusters):
                    sum_points = npoints[rowIdx] + npoints[colIdx]
                    domain_matrix[rowIdx, colIdx] = 1/(sum_points + 1)*((sum_points + 1) % 2)
                    
        elif(method == 'infiniteCrystalFiniteClusters'):
            'The crystal be infinite, the clusters have finite number of points they depend on'
            domain_matrix[0,0] = 1
        elif(method == 'byConcentration'):
            'Calculate domain matrix for infinite crystal at given concentration'
            if not(isinstance(concentration, float)):
                print('Concentration missing to calculate domain_matrix. None returned')
                sys.exit()
            'Number of occupations a cluster depends on'
            for rowIdx in range(number_clusters):
                for colIdx in range(number_clusters):
                    domain_matrix[rowIdx, colIdx] = np.power((2*concentration - 1), npoints[rowIdx] + npoints[colIdx])
                    
        elif(method == 'vdWalleAndCeder'):
            'Use A. vd Walles and G. Ceders method'
            domain_matrix = domain_matrix + np.identity(number_clusters)
        else:
            print('No valid method passed to _calculate_domain_matrix() in StructureSelector()')
            sys.exit()
            
        return domain_matrix
