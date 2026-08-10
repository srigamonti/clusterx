# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from clusterx.structures_set import StructuresSet
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from typing import List
import numpy as np
import warnings


class StructureSelector:
    """Structure selector class
    Objects of this class are used to select optimal structures to enhance the
    predictive performance of the current cluster expansion model.
    Specifically, a selected structure is chosen such that, once its added to the training set,
    the variance of predicted property values for a population of structures outside
    the training data is reduced.

    **Parameters**

    ``cpool``: ClustersPool() object
        Cluster pool to be used to calculate the correlations of the
        training set and the domain matrix.
    ``training_set``: StructuresSet() object
        Contains the current training set.
    ``candidate_set``: StructuresSet() object
        Pool of new structures, that might be selected to enter the training set.
        If it remains 'None', the class can only assess the training set.
    ``correlations_calculator``: CorrelationsCalculator object
        Used to calculate the correlations of structures and clusters.
        If this is not assigned a CorrelationsCalculator object, a default
        CorrelationsCalculator object with a trigonometric basis and the chosen
        cpool is used.
    ``regularization_strength``: float
        The default is 0.0, such that no regularization is applied to the
        Gram matrix X^T X.
        Regularization should be applied if the Gram matrix is not invertible,
        e.g. if number of structures in training set is smaller than
        the number of clusters in the cpool.
        Then an identity matrix multiplied by the regularization_strength is
        added to the Gram matrix, such that it becomes invertible.

    """

    def __init__(
        self,
        cpool: ClustersPool,
        training_set: StructuresSet,
        candidate_set: StructuresSet | None = None,
        correlations_calculator: CorrelationsCalculator | None = None,
        regularization_strength: float = 0.0,
    ):
        if not (isinstance(cpool, ClustersPool)):
            raise TypeError("cpool needs to be a ClustersPool object.")
        if not (isinstance(training_set, StructuresSet)):
            raise TypeError("training_set needs to be a StructuresSet object.")
        if not (candidate_set is None or isinstance(candidate_set, StructuresSet)):
            raise TypeError("candidate_set needs to be a StructuresSet object or None.")
        if not (
            correlations_calculator is None
            or isinstance(correlations_calculator, CorrelationsCalculator)
        ):
            raise TypeError(
                "correlations_calculator needs to be a CorrelationsCalculator object or None."
            )

        self._cpool = cpool
        self._training_set = training_set
        self._candidate_set = candidate_set
        self._regularization_strength = regularization_strength

        if correlations_calculator is None:
            self._correlations_calculator = CorrelationsCalculator(
                "trigonometric", training_set.get_parent_lattice(), cpool
            )
        else:
            self._correlations_calculator = correlations_calculator

        correlation_matrix = self._correlations_calculator.get_correlation_matrix(
            self._training_set
        )
        # Calculate covariance matrix of training set
        if regularization_strength == 0:
            self._covariance_matrix = np.linalg.inv(
                np.dot(correlation_matrix.T, correlation_matrix)
            )
        else:
            gram_matrix = np.dot(correlation_matrix.T, correlation_matrix)
            regularized_covariance_matrix = np.linalg.inv(
                gram_matrix + regularization_strength * np.eye(gram_matrix.shape[0])
            )
            self._covariance_matrix = regularized_covariance_matrix

    def set_candidate_set(
        self,
        new_candidate_set: StructuresSet):
        """Set the candidate set

        Changes the set of candidates to be selected for the training set.
        The input parameter is, thus, a StructureSet object.

        Parameters:

        new_candidate_set: StructureSet object
            The new set of candidate structures.

        """
        if not (isinstance(new_candidate_set, StructuresSet)):
            raise TypeError(
                "set_candidate_set() needs a StructuresSet object as input."
            )
        self._candidate_set = new_candidate_set

    def calculate_population_variance(
        self,
        domain_calculation_method: str = "averagedConcentration",
        concentration: float | None = None,
        lower_conc_limit: float = 0.0,
        upper_conc_limit: float = 1.0
    ) -> float:
        """Calculate mean property prediction variance over population.

        The input parameter 'domain_calculation_method'
        defines with which method the domain matrix is calculated.
        (for an explanation of the different methods, see Mueller2010 PRB 82, 184107).

        The options for this input parameter are 'averagedConcentration', 'infiniteCrystalFiniteClusters',
        'byConcentration', 'vdWalleAndCeder'.
        If 'byConcentration' is chosen, one needs to supply the parameter 'concentration' in the range [0, 1].
        Parameters:

        domain_calculation_method: string
            Defines the method used to calculate the domain matrix.
            Choose from: 'averagedConcentration', 'infiniteCrystalFiniteClusters','byConcentration',
            'vdWalleAndCeder'.
        concentration: float
            Defines the concentration for which the domain matrix should be evaluated.
        lower_conc_limit: float
            Default: 0
            Used together with method 'global_averagedConcentration'; ignored for other methods
            Lower concentration limit for the system.
            Used in the calculation of the domain matrix as lower limit of integration over concentration.
        upper_conc_limit: float
            Default: 1
            Used together with method 'global_averagedConcentration'; ignored for other methods
            Upper concentration limit for the system.
            Used in the calculation of the domain matrix as upper limit of integration over concentration.
        """

        domain_matrix = self._calculate_domain_matrix(
            method=domain_calculation_method, concentration=concentration,
            lower_conc_limit=lower_conc_limit, upper_conc_limit=upper_conc_limit
        )
        covariance_matrix = self._covariance_matrix

        # Compute Frobenius product of covarinace matrix and domain matrix
        # this is equivalent to the value of tau defined in Mueller2010 PRB 82, 184107
        population_variance = np.trace(np.dot(covariance_matrix.T, domain_matrix))

        return population_variance

    def select_structure(
        self,
        method: str | None = None,
        concentration: float | None = None,
        recompute_correlation_matrix: bool = False,
        lower_conc_limit: float = 0.0,
        upper_conc_limit: float = 1.0,
    ) -> float:
        """Select a structure to add to the training set

        This method needs a candidate set. If no candidate set was passed to the class object,
        set it with set_candidate_set().

        The routine takes each structure and adds it temporarily to the training set. With the enlarged training set
        the population variance is computed. To compute the population variance,
        supply a 'method' parameter, and maybe a 'concentration' (see calculate_population_variance).
        Afterwards the structure is removed from the training set, and the next
        structure is taken. The routine selects the structure that minimizes the population variance.

        Parameters:

        method: string
            Defines the method used to calculate the domain matrix. Options are
            'global_averagedConcentration', 'global_infiniteCrystalFiniteClusters', 'global_byConcentration',
            'global_vdWalleAndCeder' and 'greedy'.
        concentration: float
            Defines the concentration for which the domain matrix should be evaluated.
            Provide a value in the range [0,1]. Must be provided if 'method' is 'global_byConcentration'.
            Ignored in all other cases
        recompute_gram_matrix: bool
            Default is False, such that the correlation matrix of the training set
            is only computed once in the beginning.
            In order to use an active learning workflow as defined in
            select_structures_set the correlation matrix has to be recomputed
            each time a structure is added to the training set, that is,
            in each step, since the training set changes in each iteration.
        lower_conc_limit: float
            Default: 0
            Used together with method 'global_averagedConcentration'; ignored for other methods
            Lower concentration limit for the system.
            Used in the calculation of the domain matrix as lower limit of integration over concentration.
        upper_conc_limit: float
            Default: 1
            Used together with method 'global_averagedConcentration'; ignored for other methods
            Upper concentration limit for the system.
            Used in the calculation of the domain matrix as upper limit of integration over concentration.
        """
        if self._candidate_set is None:
            raise ValueError("There are no candidate structures, use set_candidate_set")
        if not (
            isinstance(method, str)
            and (method[:6] == "greedy" or method[:6] == "global")
        ):
            raise ValueError(
                'Method needs to be a string starting with "greedy" or "global_".'
            )

        global_or_greedy = method[:6]

        if recompute_correlation_matrix is True:
            correlation_matrix = self._correlations_calculator.get_correlation_matrix(
                self._training_set
            )
            if self._regularization_strength == 0:
                covariance_matrix = np.linalg.inv(
                    np.dot(correlation_matrix.T, correlation_matrix)
                )
            else:
                gram_matrix = np.dot(correlation_matrix.T, correlation_matrix)
                regularized_covariance_matrix = np.linalg.inv(
                    gram_matrix
                    + self._regularization_strength * np.eye(gram_matrix.shape[0])
                )
                covariance_matrix = regularized_covariance_matrix
        else:
            covariance_matrix = self._covariance_matrix

        if global_or_greedy == "global":
            domain_calculation_method = method.replace("global_", "")
            domain_matrix = self._calculate_domain_matrix(
                method=domain_calculation_method, concentration=concentration,
                lower_conc_limit=lower_conc_limit,
                upper_conc_limit=upper_conc_limit
            )

            candidate_correlations = (
                self._correlations_calculator.get_correlation_matrix(
                    self._candidate_set
                )
            )

            number_candidates, _ = candidate_correlations.shape

            # score vector for each structure in selection
            candidate_scores = np.zeros(number_candidates)

            for candidate_idx in range(number_candidates):
                candidate_corr = candidate_correlations[candidate_idx, :]
                scaled_correlation = np.dot(covariance_matrix, candidate_corr)
                denominator = 1 + np.dot(candidate_corr, scaled_correlation)
                numerator = np.dot(
                    scaled_correlation, np.dot(domain_matrix, scaled_correlation)
                )
                candidate_scores[candidate_idx] = numerator / denominator

            return np.argmax(candidate_scores)

        elif global_or_greedy == "greedy":
            candidate_correlations = (
                self._correlations_calculator.get_correlation_matrix(
                    self._candidate_set
                )
            )

            number_candidates, _ = candidate_correlations.shape

            # score vector for each structure in selection
            candidate_scores = np.zeros(number_candidates)

            for candidate_idx in range(number_candidates):
                candidate_corr = candidate_correlations[candidate_idx, :]
                candidate_scores[candidate_idx] = np.dot(
                    candidate_corr, np.dot(covariance_matrix, candidate_corr)
                )

            return np.argmax(candidate_scores)

        else:
            raise ValueError("No valid selector chosen.")

    def _calculate_domain_matrix(
        self,
        method: str | None = None,
        concentration: float | None = None,
        lower_conc_limit: float = 0.0,
        upper_conc_limit: float = 1.0,
    ):
        if method is None:
            raise ValueError("No method for calculation chosen.")
        method_list = [
            "averagedConcentration",
            "infiniteCrystalFiniteClusters",
            "byConcentration",
            "vdWalleAndCeder",
        ]
        if not (method is s for s in method_list):
            raise ValueError(
                "No valid method for calculation chosen. Choose one of: "
                + str(method_list)
            )

        # Calculate domain matrices with methods explained in Mueller2010 paper PRB 82, 184107

        npoints = self._cpool.get_all_npoints()
        number_clusters = len(npoints)
        domain_matrix = np.zeros((number_clusters, number_clusters), dtype=float)
        npoints_row, npoints_col = np.meshgrid(npoints, npoints, indexing="ij")
        sum_points_matrix = npoints_row + npoints_col

        # Domain matrix uniformly averaged over concentration for structures in infinite crystal
        if method == "averagedConcentration":
            # averaged concentration method calculates the integral from lower to upper concentration limit:
            # integral ((2c-1)^(n_points_alpha + n_points_beta))
            # for integration borders 0 and 1 this simplifies to:
            if lower_conc_limit == 0.0 and upper_conc_limit == 1.0:
                domain_matrix= 1/(sum_points_matrix + 1)*((sum_points_matrix + 1) % 2)
            else:
                result_upper_limit = np.power((2*upper_conc_limit-1), (sum_points_matrix+1))
                result_lower_limit = np.power((2*lower_conc_limit-1), (sum_points_matrix+1))
                domain_matrix = 1/(2*(sum_points_matrix+1))*(result_upper_limit-result_lower_limit)

        elif method == "infiniteCrystalFiniteClusters":
            # The crystal be infinite, the clusters have finite number of points they depend on
            domain_matrix[0, 0] = 1

        elif method == "byConcentration":
            # Calculate domain matrix for infinite crystal at given concentration
            if not (isinstance(concentration, float)):
                raise TypeError("Concentration needs to be a float.")
            domain_matrix = np.power((2*concentration-1), sum_points_matrix)

        elif method == "vdWalleAndCeder":
            # Use A. vd Walles and G. Ceders method
            domain_matrix = np.identity(number_clusters)
        else:
            raise ValueError(f"Invalid method: {method}")

        return domain_matrix

    def select_structures_set(
        self,
        method: str,
        lower_conc_limit: float = 0.0,
        upper_conc_limit: float = 1.0,
        exclude_equivalent_candidates: bool = False,
        cpool_candidates: ClustersPool | None = None,
        nr_structures: int | None = None,
        concentrations: List[float] | None = None,
        candidate_correlations: np.ndarray | None = None,
    ) -> StructuresSet:
        """
        Active learning workflow to iteratively select a specified number of optimal structures from a candidate_set
        in order to reduce the variance of predicted property values among a population of
        structures for the training_set.

        In each iteration an optimal structure is selected from the current candidate_set using the specified method.
        The selected structure is added to the training_set and removed from the candidate_set.
        An additional StructuresSet containing the selected structures from all iterations is created and returned.

        ***Parameters***:
        method (string):
            method to use for selecting the structure; available options are:
            - 'global_averagedConcentration'
            - 'global_infiniteCrystalFiniteClusters'
            - 'global_vdWalleAndCeder'
            - 'global_byConcentration'
        lower_conc_limit: float
            Default: 0
            Used together with method 'global_averagedConcentration'; ignored for other methods
            Lower concentration limit for the system.
            Used in the calculation of the domain matrix as lower limit of integration over concentration.
        upper_conc_limit: float
            Default: 1
            Used together with method 'global_averagedConcentration'; ignored for other methods
            Upper concentration limit for the system.
            Used in the calculation of the domain matrix as upper limit of integration over concentration.
        nr_structures (int, optional):
            Default: None
            Specifies the number of structures you want to select from candidate_set.
            Not used if method is 'global_byConcentration', where on needs to provide a list of concentrations instead.
        exclude_equivalent_candidates (bool):
            Default: False
            To avoid the presence of symmetrically equivalent structures in the candidate set,
            this parameter can be set to True.
            In this case, the correlations of all candidates are calculated and those with equal correlations
            are removed from the candidate set.
            This could lead to symmetrically inequivalent structures also being excluded, as they could have same
            correlations, especially if correlations are computed with a small clusters pool. To reduce this risk,
            it is advised to provide a clusters pool as large as possible for cpool_candidates.
        cpool_candidates: (ClusterPool object, optional):
            Clusters pool to calculate the candidate correlations in order to exclude structures with the same
            correlations. You may want to use a large clusters pool here.
            cpool_candidates is only used once to calculate the correlations.
            For structure selection a smaller clusters pool (set in the structure_selector) is used, as
            otherwise it would be computationally too expensive.
        concentrations (list of floats, optional):
            Default: None
            List of concentrations for which structure selection should be optimized.
            This does not /have to) correspond to the concentrations of the selected structures.
            Use together with method: 'global_byConcentration'
        candidate_correlations (matrix: np.ndarray[float], optional):
            Default: None
            If not assigned, the correlation matrix for the candidate set is
            calculated with the correlations calculator of the structure selector.
            If you have already calculated this before, you can resuse the result
            and provide the candidate correlations matrix as input here.

        """
        method_list = [
            "global_averagedConcentration",
            "global_infiniteCrystalFiniteClusters",
            "global_byConcentration",
            "global_vdWalleAndCeder",
        ]
        if not (method is s for s in method_list):
            raise ValueError(
                "No valid method for calculation chosen. Choose one of: "
                + str(method_list)
            )

        len_orig_training_set = len(self._training_set)
        len_orig_candidate_set = len(self._candidate_set)
        print("Len original training set: ", len_orig_training_set)
        print("Len original candidate set: ", len(self._candidate_set))
        if candidate_correlations is None:
            if cpool_candidates is not None:
                basis = self._correlations_calculator.get_basis()
                plat = self._training_set.get_parent_lattice()
                correlations_calcluator_candidates = CorrelationsCalculator(
                    basis, plat, cpool_candidates
                )
                candidate_correlations = (
                    correlations_calcluator_candidates.get_correlation_matrix(
                        self._candidate_set
                    )
                )
            else:
                candidate_correlations = (
                    self._correlations_calculator.get_correlation_matrix(
                        self._candidate_set
                    )
                )
        if exclude_equivalent_candidates is True:
            _, unique_indices = np.unique(candidate_correlations, return_index=True, axis=0)
            sorted_unique_indices = np.sort(unique_indices)
            unique_candidate_set = self._candidate_set.get_subset(sorted_unique_indices)
            nr_unique_candidates = len(unique_candidate_set)
            self._candidate_set = unique_candidate_set.get_subset(
                list(range(nr_unique_candidates))
            )
            print("New len candidate set: ", len(self._candidate_set))
            if len(self._candidate_set) < 0.8 * len_orig_candidate_set:
                warnings.warn("Reduced candidate set contains less than 80 % of original candidate set."
                "Consider increasing cpool_candidates to reduce risk of excluding symmetrically distinct structures.")

        if method == "global_byConcentration":
            if concentrations is None:
                raise ValueError(
                    "To select with global_byConcentration"
                    "please provide a list of concentrations to select for."
                )
            if any (c < 0 or c > 1 for c in concentrations):
                raise ValueError("Concentrations need to be between 0 and 1.")
            for concentration in concentrations:
                selected_index = self.select_structure(
                    method="global_byConcentration",
                    concentration=concentration,
                    recompute_correlation_matrix=True,
                )
                print(f"Structure {selected_index} is selected from the candidate set")
                indices_remaining_candidates = [
                    i for i in range(len(self._candidate_set)) if i != selected_index
                ]
                new_candidate_set = self._candidate_set.get_subset(
                    indices_remaining_candidates
                )
                indices_train = list(range(len(self._training_set)))
                new_training_set = self._training_set.get_subset(indices_train)
                new_training_set.add_structure(self._candidate_set[selected_index])
                self._training_set = new_training_set
                self._candidate_set = new_candidate_set
            indices_selected_candidates = list(
                range(len_orig_training_set, len(new_training_set))
            )
            sset_selected_candidates = new_training_set.get_subset(
                indices_selected_candidates
            )

            return sset_selected_candidates

        else:
            if nr_structures is None:
                raise ValueError(
                    "Please provide the number of structures you want to select."
                )
            for j in range(nr_structures):
                selected_index = self.select_structure(
                    method=method, concentration=None, recompute_correlation_matrix=True,
                    lower_conc_limit=lower_conc_limit, upper_conc_limit=upper_conc_limit
                )
                print(
                    f"Step {j}: Structure {selected_index} is selected from the candidate set"
                )
                indices_remaining_candidates = [
                    i for i in range(len(self._candidate_set)) if i != selected_index
                ]
                new_candidate_set = self._candidate_set.get_subset(
                    indices_remaining_candidates
                )
                indices_train = list(range(len(self._training_set)))
                new_training_set = self._training_set.get_subset(indices_train)
                new_training_set.add_structure(self._candidate_set[selected_index])
                self._training_set = new_training_set
                self._candidate_set = new_candidate_set
            indices_selected_candidates = list(
                range(len_orig_training_set, len(new_training_set))
            )
            sset_selected_candidates = new_training_set.get_subset(
                indices_selected_candidates
            )

            return sset_selected_candidates
