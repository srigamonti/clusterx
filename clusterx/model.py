# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import os
import pickle
import warnings
from pathlib import Path
from typing import List

import numpy as np
from sklearn.pipeline import Pipeline

from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.clusters_selector import ClustersSelector
from clusterx.correlations import (
    CorrelationsCalculator,
    _cluster_correlations_cell_packed_raw,
    _cluster_correlations_flip_packed,
    _pack_cluster_interactions,
    cluster_correlations_flip,
    cluster_correlations_partial,
)
from clusterx.estimators.estimator_factory import EstimatorFactory
from clusterx.estimators.affine import affine_parameters
from clusterx.estimators.polynomial import evaluate_polynomial, polynomial_parameters
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
from clusterx.utils import transformation_containing_sphere_diag


class Model:
    """Model class

    Model class to manage property estimation based on correlations and ECIs.

    **Parameters:**

    corrc : CorrelationsCalculator, optional
        The correlations calculator corresponding to the optimal model.
    property_name : str or list of str, optional
        The name of the property/properties to be queried in `StructureSet`.
    estimator : object, optional
        An estimator object to predict property values. Alternatively, ECIs can be set.
    ecis : List[float], optional
        Effective cluster interactions (multiplied by multiplicities). Overrides `estimator`.
    filepath : str, optional
        Path to a serialized model file (.json or .pickle). If provided, all other arguments are ignored.
    standardize : bool, default=False
        If True, standardizes input using `sklearn.preprocessing.StandardScaler`.
    """

    def __new__(cls, *args, **kwargs):
        """Custom __new__ to allow object creation from file if filepath is provided."""
        if not args and not kwargs:
            return super(Model, cls).__new__(cls)

        filepath = kwargs.get("filepath")
        if filepath:
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == ".pickle":
                return cls._load_from_pickle(filepath)

            elif file_ext == ".json":
                inst = super(Model, cls).__new__(cls)
                inst._load_from_json(filepath)
                return inst

        return super(Model, cls).__new__(cls)

    def __init__(
        self,
        corrc: CorrelationsCalculator | None = None,
        property_name: str | List[str] | None = None,
        estimator: object | None = None,
        ecis: List[float] | None = None,
        filepath: str | None = None,
        standardize: bool = False,
    ):
        if filepath:
            # File-based initialization is handled in __new__
            return

        self.corrc = corrc
        self.property_name = property_name

        if ecis is not None and estimator is not None:
            raise ValueError("Only one of 'ecis' or 'estimator' should be provided, not both.")

        self.estimator = estimator
        self.ecis = ecis

        # Extract coefficients from estimator if given
        if isinstance(estimator, Pipeline):
            final_estimator = estimator[-1]
            self.ecis = final_estimator.coef_
            self.intercept = final_estimator.intercept_
        elif estimator is not None:
            self.ecis = getattr(estimator, "coef_", None)
            self.intercept = getattr(estimator, "intercept_", None)
        elif ecis is not None:
            self.ecis = ecis
            self.intercept = 0
        else:
            raise ValueError("Either 'ecis' or 'estimator' must be provided.")

        self.standardize = standardize
        self._basis = None
        self._mc = False
        self._num_mc_calls = 0
        self._mc_nclusters = 0
        self._mc_multiplicities: List[int] = []
        self._mc_start_time = 0
        self._mc_init_time = 0
        self.scell_reduced = None  # if needed, initialize with self.init_reduced_model
        self.initialized_interactions = False
        self.scell_factor = 1.0

        if self.standardize:
            from sklearn.preprocessing import StandardScaler

            self.stdscaler = StandardScaler()

        if corrc:
            self._basis = corrc.get_basis()

    @staticmethod
    def _load_from_pickle(filepath: str) -> "Model":
        """Load Model object from a pickle file."""
        try:
            stem = Path(filepath).stem
            dirname = os.path.dirname(filepath)
            filepath_corrc = os.path.join(dirname, stem + "_CCALC.pickle")

            with open(filepath_corrc, "rb") as fcorrc:
                corrc = pickle.load(fcorrc)

            with open(filepath, "rb") as f:
                model = pickle.load(f)

            model.corrc = corrc
            return model

        except (FileNotFoundError, pickle.UnpicklingError) as e:
            raise ValueError(f"Error loading model from pickle file: {e}") from e

    def _load_from_json(self, filepath: str):
        """Load model data from a JSON file."""
        from ase.db import connect

        try:
            db = connect(filepath)

            self.corrc = CorrelationsCalculator(db=db)

            modict = db.metadata.get("model_parameters", None)
            if not modict:
                raise ValueError("Error: 'model_parameters' missing in JSON metadata.")

            self.ecis = modict.get("ECIs", [])
            self.property_name = modict.get("property_name", None)
            self.standardize = modict.get("standardize", False)

        except Exception as e:
            raise ValueError(f"Error loading model from JSON file: {e}") from e

    def reset_mc(self, mc=False):
        self.invalidate_prediction_cache()
        self.__dict__.pop("_packed_interactions", None)
        self.__dict__.pop("_interaction_scell_key", None)
        self.corrc.reset_mc(mc)
        self._mc = mc
        self._num_mc_calls = 0
        self._mc_nclusters = 0
        self._mc_multiplicities = []
        self._mc_start_time = 0
        self._mc_init_time = 0
        self.initialized_interactions = False

    def init_reduced_model(self, structure):
        cpool = self.corrc.get_cpool()
        plat = cpool.get_plat()
        r_max = cpool.get_max_radii().max()
        cell = plat.get_cell()
        p_reduced = transformation_containing_sphere_diag(r_max, cell, full_sphere=True)
        # round up to odd, to get a central cell
        p_reduced = p_reduced + np.diag(np.diagonal(p_reduced % 2 == 0))
        for i, pbc in enumerate(plat.get_pbc()):
            if pbc == 0:
                p_reduced[i, i] = 1
        self.scell_reduced = SuperCell(plat, p_reduced)
        self.p_reduced = p_reduced
        self.scell_factor = np.linalg.det(structure.p) / np.linalg.det(p_reduced)

    def serialize(self, filepath=None, fmt=None, db_name=None):
        """Write cluster expansion model to Json database

        **Parameter:**

        ``filepath``: string
            Name of the file to store the Model instance. It can contain the relative or absolute path.
            If not given, and if db_name (deprecated) is None, filepathh is set
            to "cemodel.pickle" and fmt is overriden (set to pickle).
        ``fmt``: string
            It can take the values ``"pickle"`` or ``"json"``.
            So far, object can only be re-initiated from a ``"pickle"`` file, therefore this is the preferred format.
            If not given, it is inferred from the filepathh.
        ``db_name``: (DEPRECATED) string
            Name of the json file containing the database
        """

        if filepath is None and db_name is None:
            filepath = "cemodel.pickle"

        if db_name is not None:
            filepath = db_name

        fext = os.path.splitext(filepath)[1][1:]

        if fmt is None:
            if fext == "pickle":
                fmt = "pickle"
            if fext == "json":
                fmt = "json_db"

        if fmt == "pickle":
            self.pickle_file = filepath

            stem = Path(filepath).stem
            dirname = os.path.dirname(filepath)
            self._filepath_corrc = os.path.join(dirname, stem + "_CCALC.pickle")
            self.corrc.serialize(filepath=self._filepath_corrc)

            with open(filepath, "wb") as f:
                pickle.dump(self, f)

        if fmt == "json_db":
            from subprocess import call

            from ase.db.jsondb import JSONDatabase

            call(["rm", "-f", db_name])
            atoms_db = JSONDatabase(filename=db_name)

            cpool_atoms = self.corrc._cpool.get_cpool_atoms()

            for atoms in cpool_atoms:
                atoms_db.write(atoms)

            modeldict = self.corrc._cpool.get_cpool_dict()

            corr_dict = {}
            corr_dict.update({"basis": self.corrc.basis})
            if self.corrc._lookup_table is not None:
                corr_dict.update({"lookup": True})
            else:
                corr_dict.update({"lookup": False})
            modeldict.update({"correlations_calculator": corr_dict})

            modict = {}
            modict.update({"property_name": self.property_name})

            if self.ecis is None:
                eff_inter = np.asarray([el for el in self.estimator.coef_])
                if (eff_inter[0] < 1.0e-15) and (self.estimator.intercept_ > 1.0e-15):
                    eff_inter[0] = float(self.estimator.intercept_)
                modict.update({"ECIs": eff_inter})
            else:
                modict.update({"ECIs": self.ecis})
            modeldict.update({"model_parameters": modict})

            atoms_db.metadata = modeldict

    def get_plat(self):
        warnings.warn(
            "Using `get_plat()` is deprecated. Use `get_parent_lattice()` instead.",
            category=FutureWarning,
        )
        return self.get_parent_lattice()

    def get_parent_lattice(self):
        """Return parent lattice of the cluster expansion model."""
        return self.corrc._plat

    def get_ecis(self):
        """Return array of effective cluster interactions (ECIs) of the model"""
        return self.ecis
        # if self.ecis is not None:
        #     return self.ecis
        # else:
        #     if self.standardize:
        #         return self.estimator[-1].coef_
        #     else:
        #         return self.estimator.coef_

    def get_correlations_calculator(self):
        """Return correlations calculator of the Model object"""
        return self.corrc

    def predict(self, structure: Structure, reduce_structure: bool = False, flag=None):
        """Predict property with the optimal cluster expansion model.

        **Parameters:**

        ``structure``: Structure object
            structure object to calculate property to.

        ``reduce_structure``: bool, default: False
            If true, use reduced structure to predict, only works for diagonal
            supercell transformation.

        ``flag``: dict or None
            it flags whether the member corrc correlationsCalculator computed
            orbits from scratch. This can be useful to konw, in order to serialize the
            model instance to accelerate next property predictions

            Example usage::

                model = Model(filepath="myfilepath.pickle")
                model.predict(structure,flag={})
                if  flag["computed_orbits_from_scratch"]:
                   print("INFO: correlations calculator computed orbits from scrach.")
                   model.serialize(filepath="myfilepath.pickle")
        """
        return self.predict_on_correlations(self.get_correlations(structure, reduce_structure, flag=flag))

    def get_correlations(self, structure: Structure, reduce_structure: bool = False, flag=None):
        """Return global correlations, optionally streaming through reduced neighborhoods.

        The reduced structure must first be initialized with init_reduced_model
        and Structure.init_reduced_structure, as in reduced Monte Carlo. This
        path uses model-sized accumulators and only reduced-cell orbit geometry,
        independent of the full supercell size. It works for nonlinear models
        because correlations are accumulated before applying the estimator.
        """
        if not reduce_structure:
            return self.corrc.get_cluster_correlations(structure, flag=flag)

        local, _ = structure.get_reduced_structure(0)
        local_cell = local.get_supercell()
        if len(local.sigmas) != len(local_cell):
            raise ValueError(
                "Reduced correlation grids require odd repeat lengths. "
                "Initialize them with Model.init_reduced_model and Structure.init_reduced_structure."
            )
        self._ensure_interaction_dict(local_cell)
        if "_packed_interactions" not in self.__dict__:
            self._packed_interactions = _pack_cluster_interactions(self._site_clusters, self._cluster_orbits_array)
        packed = self._packed_interactions
        correlations = np.zeros(len(self._multiplicities))
        n_parent_sites = len(structure.get_parent_lattice())
        for index in range(0, len(structure.sigmas), n_parent_sites):
            local, first_site = structure.get_reduced_structure(index)
            if packed is not None:
                correlations += _cluster_correlations_cell_packed_raw(
                    local.sigmas, local.ems, first_site, n_parent_sites, self.corrc.basis_set_values,
                    *packed, self._cluster_indices, len(correlations),
                )
            else:
                for site in range(first_site, first_site + n_parent_sites):
                    correlations += cluster_correlations_partial(
                        local, site, self._site_clusters, self._cluster_orbits_array,
                        self._cluster_indices, self.corrc.basis_set_values, self._multiplicities,
                        normalize=False,
                    )
        nonempty = self._n_body_list != 0
        # Summing over sites counts every cluster once per member. Undo this,
        # normalize the orbit sums, and rescale from the reduced to the full cell.
        correlations[nonempty] /= (
            self._n_body_list[nonempty] * self._multiplicities[nonempty] * self.scell_factor
        )
        correlations[~nonempty] = 1.0  # Empty-cluster correlations are constant.
        return np.around(correlations, decimals=12)

    def predict_partial(self, structure: Structure, index: int, divide_n_body: bool = False):
        """Predict property using only the correlations around site `index`.
        This uses all correlations of all clusters that include this site.
        Note: only works for diagonal super cells.
        It is an additive energy contribution only for affine models.
        """
        _structure, _index = structure.get_reduced_structure(index)
        _scell = _structure.get_supercell()

        self._ensure_interaction_dict(_scell)

        correlations_partial = cluster_correlations_partial(
            structure=_structure,
            ind=_index,
            site_clusters=self._site_clusters,
            cluster_orbits_array=self._cluster_orbits_array,
            cluster_indices=self._cluster_indices,
            basis_set_values=self.corrc.basis_set_values,
            multiplicities=self._multiplicities,
        )

        if divide_n_body:  # undo double counting, when looping over all sites
            correlations_partial /= self._n_body_list

        correlations_partial /= self.scell_factor
        return self.predict_on_correlations(correlations_partial)

    def _init_interaction_dict(self, super_cell: SuperCell):
        """Construct list of which sites are contained in which orbit.
        Rebuilding interactions discards any derived packed geometry.
        """
        self.__dict__.pop("_packed_interactions", None)
        self.initialized_interactions = False
        cluster_orbits = self.corrc.get_cluster_orbits_for_scell(super_cell)
        cluster_indices = []  # to which primitive cluster each cluster belongs
        cluster_orbits_array = []  # all clusters from all orbits
        multiplicities = []  # relative to cluster_orbits_array
        n_body_list = []  # list of n's where n is number of sites for each primitive cluster
        for cluster_index, cluster_orbit in enumerate(cluster_orbits):
            multiplicities.append(len(cluster_orbit))
            n_body_list.append(len(cluster_orbit[0]))
            for cluster in cluster_orbit:
                cluster_indices.append(cluster_index)
                cluster_orbits_array.append(cluster)

        # List of lists, where each sublist contains indices
        # from cluster_orbits_array
        site_clusters = [[] for _ in range(len(super_cell))]
        for index, cluster in enumerate(cluster_orbits_array):
            for site in cluster.get_idxs():
                site_clusters[site].append(index)
        # n_interactions = sum([len(sc) for sc in site_clusters])
        # print(f"Info(Model): # saved interactions: {n_interactions}")

        self._cluster_orbits_array = np.array(cluster_orbits_array, dtype=object)
        self._cluster_indices = np.array(cluster_indices, dtype=int)
        self._multiplicities = np.array(multiplicities, dtype=int)
        self._n_body_list = np.array(n_body_list)
        self._site_clusters = site_clusters
        self._interaction_scell_key = self._interaction_key(super_cell)
        self.initialized_interactions = True

    @staticmethod
    def _interaction_key(super_cell):
        # Constant-size key; never retain another full supercell or a site map.
        return id(super_cell.get_parent_lattice()), super_cell.get_transformation().tobytes()

    def _ensure_interaction_dict(self, super_cell):
        if (not self.initialized_interactions
                or getattr(self, "_interaction_scell_key", None) != self._interaction_key(super_cell)):
            self._init_interaction_dict(super_cell)

    def predict_correlations_diff(
        self,
        structure: Structure,
        index: int,
        old_sigma: int,
        new_sigma: int,
        reduce_structure: bool = False,
    ):
        """Return difference in correlations when changing sigma from old_sigma
        to new_sigma at index in structure.

        For documentation on paremeters, see predict_flip."""
        if not reduce_structure:  # default case
            _structure = structure
            _index = index
            _scell = _structure.get_supercell()
        else:
            _structure, _index = structure.get_reduced_structure(index)
            _scell = _structure.get_supercell()

        self._ensure_interaction_dict(_scell)

        # Only reduced-cell geometry is packed. Cache a declined attempt as None
        # so a pool exceeding the byte budget does not retry on every flip.
        if reduce_structure and "_packed_interactions" not in self.__dict__:
            self._packed_interactions = _pack_cluster_interactions(self._site_clusters, self._cluster_orbits_array)
        packed = getattr(self, "_packed_interactions", None) if reduce_structure else None
        if packed is not None:
            correlations_diff = _cluster_correlations_flip_packed(
                _structure, _index, old_sigma, new_sigma, packed, self._cluster_indices,
                self.corrc.basis_set_values, self._multiplicities,
            )
        else:
            correlations_diff = cluster_correlations_flip(
                structure=_structure,
                ind=_index,
                old_sigma=old_sigma,
                new_sigma=new_sigma,
                site_clusters=self._site_clusters,
                cluster_orbits_array=self._cluster_orbits_array,
                cluster_indices=self._cluster_indices,
                basis_set_values=self.corrc.basis_set_values,
                multiplicities=self._multiplicities,
            )
        correlations_diff /= self.scell_factor

        return correlations_diff

    def predict_correlations_swap(
        self,
        structure: Structure,
        i: int,
        j: int,
        reduce_structure: bool = False,
    ) -> np.typing.ArrayLike:
        """Return difference in correlations when swapping sigmas at indices
        i and j in structure.

        For documentation on paremeters, see predict_swap."""
        sigma_i = structure.sigmas[i]
        sigma_j = structure.sigmas[j]

        corrs_i = self.predict_correlations_diff(structure, i, sigma_i, sigma_j, reduce_structure)
        structure.sigmas[i] = sigma_j
        corrs_j = self.predict_correlations_diff(structure, j, sigma_j, sigma_i, reduce_structure)
        structure.sigmas[i] = sigma_i  # restore original structure
        return corrs_i + corrs_j

    def predict_flip(
        self,
        structure: Structure,
        index: int,
        old_sigma: int,
        new_sigma: int,
        reduce_structure: bool = False,
    ):
        """Predict property change by flipping a species.

        Structure object remains unchanged
        This direct difference path assumes an affine model. Nonlinear Monte
        Carlo tracks the global correlation state with nonlinear=True instead.

        **Parameters:**

        ``structure``: Structure object
            structure object to calculate property difference to.

        ``atom_index``: int
            index of the atom to be substituted

        ``new_sigma``: int, default None

        ``reduce_structure``: bool, default: False
            If true, use reduced structure to predict flip energy, only works
            for diagonal supercell transformation.
        """
        correlations_diff = self.predict_correlations_diff(structure, index, old_sigma, new_sigma, reduce_structure)
        affine = self._get_affine_parameters(len(correlations_diff))
        if affine is not None:
            # Offsets from scaling and bias features cancel in an energy difference.
            return affine[0] @ correlations_diff
        return self.predict_on_correlations(correlations_diff) - self.intercept

    def invalidate_prediction_cache(self):
        """Discard folded coefficients after modifying a fitted estimator in place.

        Estimator/step replacement and refits replacing final ``coef_``/``intercept_``
        are detected automatically. Call this after editing fitted arrays in
        place or changing transformer/selector settings with set_params().
        Starting a MonteCarlo run through reset_mc() also clears the cache.
        """
        self.__dict__.pop("_affine_sources", None)
        self.__dict__.pop("_affine_n_features", None)
        self.__dict__.pop("_affine_parameters", None)
        self.__dict__.pop("_polynomial_sources", None)
        self.__dict__.pop("_polynomial_n_features", None)
        self.__dict__.pop("_polynomial_parameters", None)
        self.__dict__.pop("_polynomial_scalar_output", None)

    def __getstate__(self):
        # Derived caches must not outlive the fitted state or geometry they describe.
        return {key: value for key, value in self.__dict__.items()
                if not key.startswith(("_affine_", "_polynomial_"))
                and key not in ("_packed_interactions", "_interaction_scell_key")}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.invalidate_prediction_cache()
        self.__dict__.pop("_packed_interactions", None)
        self.__dict__.pop("_interaction_scell_key", None)

    def _get_affine_parameters(self, n_features):
        estimator = self.estimator
        if estimator is None:
            return None  # Explicit ECIs already use a direct dot product.
        steps = [step for _, step in estimator.steps] if isinstance(estimator, Pipeline) else [estimator]
        final = steps[-1]
        # Retain references, rather than only ids, so Python cannot reuse a freed
        # fitted array's id and accidentally make an old cache appear current.
        sources = (estimator, *steps, getattr(final, "coef_", None), getattr(final, "intercept_", None))
        previous = getattr(self, "_affine_sources", ())
        if (n_features != getattr(self, "_affine_n_features", None) or len(sources) != len(previous)
                or any(a is not b for a, b in zip(sources, previous))):
            self._affine_parameters = affine_parameters(estimator, n_features)
            self._affine_sources = sources
            self._affine_n_features = n_features
            self.ecis = getattr(final, "coef_", None)
            self.intercept = getattr(final, "intercept_", None)
        return self._affine_parameters

    def _get_polynomial_parameters(self, n_features):
        estimator = self.estimator
        if estimator is None:
            return None
        steps = estimator.steps if type(estimator) is Pipeline else [("regressor", estimator)]

        def cache_source(value):
            if value is None or np.isscalar(value):
                return ("value", value)
            return value

        sources = [estimator]
        for _, step in steps:
            sources.append(step)
            # The transformer object is tracked; powers are generated only for
            # selected features when the compiled cache is built.
            for attribute in ("coef_", "intercept_", "mean_", "scale_"):
                value = getattr(step, attribute, None)
                if value is not None:
                    sources.append(cache_source(value))
            fitted_selector = getattr(step, "estimator_", None)
            if fitted_selector is not None:
                sources.extend((fitted_selector, cache_source(getattr(fitted_selector, "coef_", None))))
        sources = tuple(sources)
        previous = getattr(self, "_polynomial_sources", ())
        source_changed = [
            index for index, (a, b) in enumerate(zip(sources, previous))
            if (a != b if isinstance(a, tuple) and a[:1] == ("value",) else a is not b)
        ]
        if (n_features != getattr(self, "_polynomial_n_features", None) or len(sources) != len(previous)
                or source_changed):
            result = polynomial_parameters(estimator, n_features)
            if result is None:
                self._polynomial_parameters = None
                self._polynomial_scalar_output = None
            else:
                self._polynomial_parameters, self._polynomial_scalar_output = result
            self._polynomial_sources = sources
            self._polynomial_n_features = n_features
        if self._polynomial_parameters is None:
            return None
        return self._polynomial_parameters, self._polynomial_scalar_output

    def predict_on_correlations(self, correlations: np.array):
        """Using calculated correlations, return a prediction."""
        if self.estimator is not None:
            affine = self._get_affine_parameters(len(correlations))
            if affine is not None:
                return affine[0] @ correlations + affine[1]
            polynomial = self._get_polynomial_parameters(len(correlations))
            if polynomial is not None:
                return evaluate_polynomial(polynomial[0], correlations, polynomial[1])
            return self.estimator.predict(correlations.reshape(1, -1))[0]
        else:
            return np.dot(self.ecis, correlations)

    def predict_swap(
        self,
        structure,
        i,
        j,
        reduce_structure: bool = False,
    ):
        """Predict property difference with the optimal cluster expansion model.

        This direct difference path assumes an affine model; use the nonlinear
        Monte Carlo path for models nonlinear in their correlations.

        **Parameters:**

        ``structure``: Structure object
            structure object to calculate property difference to.

        ``i``: int
            index of first atom position has been swapped

        ``j``: int
            index of second atom position has been swapped
        """

        sigma_i = structure.sigmas[i]
        sigma_j = structure.sigmas[j]

        de1 = self.predict_flip(structure, i, sigma_i, sigma_j, reduce_structure)
        structure.sigmas[i] = sigma_j
        de2 = self.predict_flip(structure, j, sigma_j, sigma_i, reduce_structure)
        structure.sigmas[i] = sigma_i  # restore original structure
        return de1 + de2

    def report_errors(self, sset):
        """Report fit and CV scores

        **Parameters**:

        ``sset``: StructuresSet object
            the scores are computed for the give structures set
        """
        properties = self.property_name if isinstance(self.property_name, list) else [self.property_name]
        for prop in properties:
            errfit = self.get_errors(sset, prop)
            errcv = self.get_cv_score(sset, prop)

            print("\n+-----------------------------------------------------------+")
            print(f"|{f'Report of Fit and CV scores: {prop}':^59}|")
            print("+-----------------------------------------------------------+")
            print(f"|{'':<19}|{'Fit':^19}|{'CV':^19}|")
            print("+-----------------------------------------------------------+")
            print(f"|{'RMSE':^19}|{errfit['RMSE']:^19.5f}|{errcv['RMSE-CV']:^19.5f}|")
            print(f"|{'R2':^19}|{errfit['R2']:^19.5f}|{errcv['R2-CV']:^19.5f}|")
            print(f"|{'MAE':^19}|{errfit['MAE']:^19.5f}|{errcv['MAE-CV']:^19.5f}|")
            print(f"|{'MaxAE':^19}|{errfit['MaxAE']:^19.5f}|{errcv['MaxAE-CV']:^19.5f}|")
            print("+-----------------------------------------------------------+\n")

    def get_errors(self, sset, property_name_single):
        """Compute RMSE, MAE and MaxAE for model in structures set."""
        calc_vals = sset.get_property_values(property_name=property_name_single)
        predictions = sset.get_predictions(self)
        if isinstance(self.property_name, list):
            index_prop = self.property_name.index(property_name_single)
            predictions = predictions[:, index_prop]

        abs_errors = np.abs(predictions - calc_vals)
        rmse = np.sqrt(np.mean(np.square(abs_errors)))
        # coefficient of determination
        r2 = 1 - np.sum(abs_errors**2) / np.sum((calc_vals - np.mean(calc_vals)) ** 2)
        mae = np.mean(abs_errors)
        maxae = np.max(abs_errors)
        i_sample_max = np.argmax(abs_errors)

        return {
            "RMSE": rmse,
            "R2": r2,
            "MAE": mae,
            "MaxAE": maxae,
            "sample_index_max": i_sample_max,
        }

    def get_all_errors(self, sset, srtd=False):
        """Return absolute errors of every structure in a structures set.

        **Parameters:**

        ``sset``: StructuresSet object
            the errors are calculated for every structure in the structures set
        ``srtd``: Boolean (default: ``False``)
            if ``True`` the output errors are sorted from smallest to largest.

        **Returns:**
        structured numpy array with fields::

            [('index','<i4'), ('error','<f4')]

        i.e., every element of the array contains a tuple with an integer, the structure index
        and a float, the absolute error
        """
        calc_vals = sset.get_property_values(property_name=self.property_name)
        predictions = sset.get_predictions(self)

        aes = []
        i_sample = 0
        for e, p in zip(calc_vals, predictions):
            aes.append((i_sample, np.abs(e - p)))
            i_sample += 1

        straes = np.array(aes, dtype=[("index", "<i4"), ("error", "<f4")])

        if srtd:
            saes = np.sort(straes, order="error")
            return saes
        else:
            return straes

    def get_cv_score(self, sset, property_name_single, params=None):
        """Get leave-one-out cross-validation score over structures set.

        **Parameters:**

        ``params``: dictionary
            Parameters to pass to the fit method of the estimator.
        """
        from sklearn.model_selection import (
            LeaveOneOut,
            cross_val_predict,
            cross_val_score,
        )

        x_mat = self.corrc.get_correlation_matrix(sset)
        y = sset.get_property_values(property_name=property_name_single)

        # cross_val_score internally clones the estimator, so the optimal one in Model is not changed.
        cvs = cross_val_score(
            self.estimator,
            x_mat,
            y,
            params=params,
            cv=LeaveOneOut(),
            scoring="neg_mean_squared_error",
        )
        pred_cv = cross_val_predict(self.estimator, x_mat, y, params=params, cv=LeaveOneOut())

        absolute_errors = np.sqrt(-cvs)
        r2 = 1 - np.sum(absolute_errors**2) / np.sum((y - np.mean(y)) ** 2)
        cv = np.sqrt(-np.mean(cvs))
        maxae = np.amax(absolute_errors)
        mae = np.mean(absolute_errors)

        return {
            "RMSE-CV": cv,
            "R2-CV": r2,
            "MAE-CV": mae,
            "MaxAE-CV": maxae,
            "Predictions-CV": pred_cv,
        }


class ModelBuilder:
    """Model class

    Objects of this class represent a cluster expansion model.

    **Parameters:**

    ``basis``: string (Default: ``"trigonometric"``)
        Basis set used to calculate the matrix of cluster correlations. For a detailed
        description and possible values of this parameter, look at the documentation
        for parameter ``basis`` in :class:`CorrelationsCalculator <clusterx.correlations.CorrelationsCalculator>`

    ``selector_type``: string
        Cluster selector type. For the possible values, look at the
        documentation for the parameter ``selector_type`` in the
        :class:`ClustersSelector <clusterx.clusters_selector.ClustersSelector>` class.

    ``selector_opts``: dictionary
        Cluster selector options. For the possible values, look at the
        documentation for the parameter ``selector_opts`` in the
        :class:`ClustersSelector <clusterx.clusters_selector.ClustersSelector>` class.

    ``estimator_type``: string
        Estimator type. For the possible values, look at the documentation
        for the parameter ``estimator_type`` in the
        :class:`EstimatorFactory <clusterx.estimators.estimator_factory.EstimatorFactory>` class.

    ``estimator_opts``: dictionary
        Estimator options. For the possible values, look at the documentation
        for the parameter ``estimator_opts`` in the
        :class:`EstimatorFactory <clusterx.estimators.estimator_factory.EstimatorFactory>` class.

    ``filepath``: string (default: None)
        If not None, it must be the path to a pickle file generated with ModelBuilder.serialize(). All other
        parameters are overriden

    ``standardize``: Boolean (defaut:``False``)
        Whether to standardize the input. Uses StandardScaler of scikit-learn.
    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            inst = super(ModelBuilder, cls).__new__(cls, *args, **kwargs)
            return inst

        elif "filepath" in kwargs:
            filepath = kwargs["filepath"]

            with open(filepath, "rb") as f:
                inst = pickle.load(f)

            if not isinstance(inst, cls):
                raise TypeError("Unpickled object is not of type {}".format(cls))

            return inst

        else:
            inst = super(ModelBuilder, cls).__new__(cls)
            inst.initialize(*args, **kwargs)
            return inst

    def __init__(
        self,
        basis="trigonometric",
        selector_type="identity",
        selector_opts={},
        estimator_type="skl_LinearRegression",
        estimator_opts={},
        filepath=None,
        standardize=False,
    ):
        pass

    def initialize(
        self,
        basis="trigonometric",
        selector_type="identity",
        selector_opts={},
        estimator_type="skl_LinearRegression",
        estimator_opts={},
        filepath=None,
        standardize=False,
    ):
        self.basis = basis
        self.selector_type = selector_type
        self.selector_opts = selector_opts
        self.estimator_type = estimator_type
        self.estimator_opts = estimator_opts
        self.standardize = standardize
        if "standardize" not in selector_opts.keys():
            selector_opts["standardize"] = standardize

        self.ini_comat = None

        self.opt_cpool = None
        self.opt_corrc = None
        self.opt_comat = None
        self.opt_estimator = None
        self.selector = None

    def get_selector(self):
        """
        Return selector used in build.

        When the ``build`` method is called, a ``ClustersSelector`` object
        is created to perform the cluster selection task. This selector
        can be obtained by calling this function.
        """
        return self.selector

    def get_initial_comat(self):
        """Return initial correlation matrix

        Return the initial correlation matrix, i.e. the full correlation matrix before
        cluster selection is performed.
        """
        return self.ini_comat

    def get_opt_cpool(self):
        """
        Return optimal clusters pool found in build.

        When the ``build`` method is called, a ``ClustersSelector`` object
        is created to perform the cluster selection task. The selected clusters pool
        can be obtained by calling this function.
        """
        return self.opt_cpool

    def get_estimator(self):
        """Return estimator object used to create the optimal model"""
        return self.opt_estimator

    def build(
        self,
        sset: StructuresSet,
        cpool: ClustersPool,
        prop: str | List[str],
        corrc: CorrelationsCalculator | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Build optimal cluster expansion model

        Acts as a Model factory.

        **Parameters:**

        ``sset``: StructuresSet object
            structures set used for model training.

        ``cpool``: ClustersPool object
            clusters pool from which to select the best model using the method
            indicated in ``selector_type``.

        ``prop``: String | list of Strings
            Property name or list of property names.
            Must be a valid name as stored in ``sset``. The list of names can
            be obtained using ``sset.get_property_names()``.

        ``corrc``: CorrelationsCalculator object (default: None)
            If not None, cpool and basis are overriden

        ``kwargs``: Keyword arguments passed to the fit method of the estimator.
        """
        self.sset = sset
        self.plat = cpool.get_plat()
        self.prop = prop

        if verbose:
            print("ModelBuilder: initialize correlations calculator")

        if corrc is None:
            self.cpool = cpool
            corrc = CorrelationsCalculator(self.basis, self.plat, self.cpool)
        else:
            self.cpool = corrc._cpool
            self.basis = corrc.basis_name

        if verbose:
            print("ModelBuilder: Build correlations matrix")
        self.ini_comat = corrc.get_correlation_matrix(self.sset, verbose=verbose)

        self.target = self.sset.get_property_values(property_name=self.prop)

        self.opt_cpool = self.cpool
        self.opt_comat = self.ini_comat

        # Select optimal clusters using the clusters_selector module
        if self.selector_type != "identity":
            self.selector = ClustersSelector(basis=self.basis, method=self.selector_type, **self.selector_opts)
            self.opt_cpool = self.selector.select_clusters(
                sset,
                cpool,
                prop,
                comat=self.ini_comat,
            )
            self.opt_comat = self.selector.optimal_comat
        self.opt_cpool.print_info()

        self.opt_corrc = CorrelationsCalculator(self.basis, self.plat, self.opt_cpool)

        # Find out the ECIs using an estimator
        if not self.standardize:
            self.opt_estimator = EstimatorFactory.create(self.estimator_type, **self.estimator_opts)
        else:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            self.opt_estimator = make_pipeline(
                StandardScaler(),
                EstimatorFactory.create(self.estimator_type, **self.estimator_opts),
            )

        self.opt_estimator.fit(self.opt_comat, self.target, **kwargs)
        return Model(
            corrc=self.opt_corrc,
            property_name=prop,
            estimator=self.opt_estimator,
            standardize=self.standardize,
        )

    def serialize(self, filepath="MODELBDR.pickle"):
        """Serialize model into pickle file

        Only pickle format is supported.

        **Parameters:**

        ``filepath``: string (default: "MODELBDR.pickle")
            file path of the pickle file to serialize the model builder object
        """
        # from pathlib import Path
        # import os

        with open(filepath, "wb") as f:
            pickle.dump(self, f)
