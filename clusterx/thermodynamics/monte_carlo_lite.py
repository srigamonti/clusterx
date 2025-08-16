# Copyright (c) 2015-2025, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
from __future__ import annotations

import os
import pickle
import random
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from clusterx.utils import _timed


class MonteCarloLite:
    r"""Monte Carlo class for binary single sublattice CE models

    **Description**:
        Objects of this class are used to perform Monte Carlo samplings.

        It is initialized with:

        - a Model object, that enables to calculate the energy of a structure,

        - a SuperCell object, in which the sampling is performed,

        - specification of the thermodynamic ensemble:

            If ``ensemble`` is 'canonical', the composition for the sampling is defined with ``nsubs``. In case
            of multilattices, the sublattice for the sampling can be refined with ``sublattice_indices``.

            If ``ensemble`` is 'grandcanonical', the sublattice is defined with ``sublattices_indices``.

    **Parameters**:

    ``energy_model``: Model object
        Model used for acceptance and rejection. Usually, the Model enables to
        calculate the total energy of a structure.

    ``scell``: SuperCell object
        Simulation cell in which the sampling is performed.

    ``energy_scale_factor``: float or None
        the energy given by energy_model times the energy_scale_factor must give the total
        energy in the simulation supercell. If None, it is assumed that the energy_model
        gives the energy per parent lattice and energy_scale_factor is set to scell.get_index().

    ``boltzmann_constant``: float (optional, default 1.0)
        Boltzmann constant kb. kb times the temperature must have the same units
        as the units of energy that energy model yields.

    .. todo:
        Samplings in the grand canonical ensemble are not yet possible.

    """

    def __init__(
        self,
        energy_model,
        scell,
        energy_scale_factor=None,
        boltzmann_constant=1.0,
    ):
        self._emodel = energy_model
        self._emodel.reset_mc(True)

        self._scell = scell

        self._energy_scale_factor = (
            energy_scale_factor
            if energy_scale_factor is not None
            else scell.get_index()
        )
        self._kb = boltzmann_constant

        if not scell.is_nary(2):
            raise ValueError(
                "MonteCarloLite.init(): MonteCarloLite works only for binary materials."
            )

        self._substitutional_sublattice = None
        for k, v in scell.get_sublattice_types().items():
            if len(v) == 2:
                self._substitutional_sublattice = int(k)
                break

        with _timed("MonteCarloLite.init(): Generating pristine structure"):
            self.structure = self._scell.get_pristine_structure()

        with _timed("MonteCarloLite.init(): Computing energy of pristine structure"):
            self.e_pristine = self._emodel.predict(self.structure)

        with _timed("MonteCarloLite.init(): Initialize interaction dictionaries"):
            self._emodel._initialize_interaction_dictionaries(
                self._scell, [self._substitutional_sublattice]
            )

        print("\n" + "=" * 70)
        print("MonteCarloLite Initialized")
        print("=" * 70)
        print(f"Sublattice index:          {self._substitutional_sublattice}")
        print(f"Energy scale factor:       {self._energy_scale_factor}")
        print(f"Boltzmann constant (kB):   {self._kb}")
        print(f"Supercell size:            {self._scell.get_index()} parent lattices")
        print(f"Energy model:              {type(self._emodel).__name__}")
        print(f"Energy model - num params: {len(self._emodel.ecis)}")
        print("=" * 70 + "\n")

    @classmethod
    def from_file(cls, filepath: str) -> MonteCarloLite:
        """
        Load a MonteCarloLite object from a serialized file.

        Supported formats:
        - `.pickle`: Full Python object serialization using `pickle`.

        Parameters
        ----------
        filepath : str
            Path to the serialized file. The format is inferred from the file extension.

        Returns
        -------
        Structure
            A reconstructed Structure object based on the contents of the file.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        file_ext = os.path.splitext(filepath)[1].lower()
        if file_ext == ".pickle":
            return cls._load_from_pickle(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    @staticmethod
    def _load_from_pickle(filepath: str) -> MonteCarloLite:
        with open(filepath, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _validate_initialization_args_metropolis(
        initial_sigmas: Optional[Any] = None,
        initial_structure: Optional[Any] = None,
        n_substitutions: Optional[Any] = None,
    ) -> None:
        """
        Ensure that at most one of the three initialization parameters is provided.
        Raises ValueError if more than one is given.
        If none is provided, the pristine structure or the one from a previous
        Metropolis run will be used.
        """
        provided = sum(
            x is not None for x in (initial_sigmas, initial_structure, n_substitutions)
        )

        if provided > 1:
            raise ValueError(
                "Only one of 'initial_sigmas', 'initial_structure', or 'n_substitutions' "
                "must be given, not multiple. If none is provided, the pristine structure "
                "or the one from a previous Metropolis run will be used."
            )

    def _load_or_create_mcrun(
        self, mcrun_filepath: Optional[str], temperature: float, ensemble: str
    ):
        """
        Load an MCRun from file if possible, otherwise create a new instance.

        Parameters
        ----------
        mcrun_filepath : str or None
            Path to the MC run file. If None, a new instance is created.
        temperature : float
            Temperature for the MC run (used if creating new).
        ensemble : str
            Ensemble type (used if creating new).

        Returns
        -------
        MCRun
            Loaded or newly created MC run instance.
        """
        path = Path(mcrun_filepath) if mcrun_filepath is not None else None

        if path is None:
            warnings.warn(
                "No file path provided for the MC run. A new run will be created; "
                "trajectory will not be saved to a file.",
                UserWarning,
            )
            return MCRun(
                self._scell.get_parent_lattice(),
                self._scell.get_transformation(),
                temperature,
                ensemble,
            )

        if path.exists():
            mcrun = MCRun.from_file(path)

            # Warn if stored temperature differs from requested one
            if not np.isclose(mcrun.temperature, temperature, rtol=1e-8, atol=0.0):
                warnings.warn(
                    f"Temperature in saved MC run ({mcrun.temperature}) "
                    f"differs from requested temperature ({temperature}).",
                    UserWarning,
                )

            mcrun.resume_from_last_step()  # resume after last recorded step
            return mcrun

        warnings.warn(
            f"MC run file not found at: {path}. Starting a new run instance.",
            UserWarning,
        )
        return MCRun(
            self._scell.get_parent_lattice(),
            self._scell.get_transformation(),
            temperature,
            ensemble,
        )

    def _compute_acceptance_ratio(self, mcrun, i: int, window: int = 100) -> float:
        """
        Compute the acceptance ratio for the Monte Carlo run.

        If `i` <= window, computes ratio = total accepted steps / i.
        If `i` > window, computes the ratio over the last `window` iterations only.

        Parameters
        ----------
        mcrun : MCRun
            The Monte Carlo run object.
        i : int
            Current iteration index.
        window : int, optional
            Number of most recent iterations to use when computing the ratio
            if `i` is greater than this value. Default is 100.

        Returns
        -------
        float
            Acceptance ratio.
        """
        if i <= 0:
            return 0.0  # avoid division by zero

        if i > window:
            lower_bound = i - window
            recent_accepts = 0
            # Iterate backwards until we exit the window range
            for step in reversed(mcrun.accepted_steps):
                if step <= lower_bound:
                    break
                recent_accepts += 1
            return recent_accepts / window
        else:
            return len(mcrun.accepted_steps) / i

    def metropolis(
        self,
        temperature=None,
        n_mc_steps=None,
        n_clics=1,
        ensemble=None,
        n_substitutions=None,
        initial_structure=None,
        initial_sigmas=None,
        chemical_potential=None,
        mcrun_filepath=None,
        n_error_reset=None,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        r"""Perform Monte-Carlo Metropolis simulation

        **Description**:
            Perfom Monte-Carlo Metropolis sampling for
            ``no_of_sampling_steps`` sampling steps.

            During the sampling, a new structure at step i is accepted
            with the probability given by :math:`\min( 1, \exp( - (E_i - E_{i-1})/(k_B T)) )`

            The energy :math:`E_i` of visited structure at step i is calculated from the Model
            ``energy_model``. The factor :math:`k_B T` is the product of the temperature :math:`T`
            and the Boltzmann constant :math:`k_B`.

            Note: The units of the ``energy`` :math:`E` and the factor :math:`k_B T` must be the same.
            With ``scale_factor``, :math:`k_B T` can be adjusted to the correct units (see below).

        **Parameters**:

        ``n_mc_steps``: integer
            Number of sampling steps

        ``temperature``: float
            Temperature at which the sampling is performed.

        ``n_clics``: integer
            Number of species flips (grandcanonical) or species swaps (canonical) per sampling step.
            Defaults to 1.

        ``ensemble``: string (default: ``canonical``)
            ``canonical`` allows for swaps of atoms that conserve the concentration defined with ``nsubs``.

            ``grandcanonical`` allows for replacing atoms in the sub-lattices defined with ``sublattice_indices``.
            (So far, ``grandcanonical`` is not yet implemented.)

        ``n_substitutions``: integer, optional
            Defines the number of substituted atoms in the sublattice.

        ``initial_structure``: Structure object, optional
            Initial structure from which the sampling starts.

        ``initial_sigmas``: array, list, or tuple of int, optional
            Initial site occupation values (sigmas) from which the sampling starts.

        ``chemical_potential``: dictionary (default: None)
            Define the chemical potential in the grand canonical ensemble.

        ``mcrun_filepath``: string (default: ``trajectory.json``)
            Name of a Json file in which the trajectory is serialized after the sampling if ``serialize`` is **True**.

        ``n_error_reset``: integer (default: None)
            If not **None*  and ``predict_swap`` equal to **True**, the correlations are calculated as usual (no differences) every n-th step.

        ``random_seed``: Integer or None (default: None)
            Random seed to produce reproducible results

        ``**kwargs``: keyworded argument list, arbitrary length
            These arguments are added to the MonteCarloTrajectory object that is initialized in this method.

        **Returns**: MonteCarloTrajectory object
            Trajectory containing the complete information of the sampling trajectory.

        Notes
        -----
        At most one of `n_substitutions`, `initial_structure`, or `initial_sigmas`
        may be provided. If none are given, the pristine structure or the one from a
        previous Metropolis run will be used.
        """
        import math

        from tqdm import tqdm

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        mcrun = self._load_or_create_mcrun(mcrun_filepath, temperature, ensemble)

        self._validate_initialization_args_metropolis(
            initial_sigmas, initial_structure, n_substitutions
        )

        if len(mcrun.sigmas) > 0:
            sigmas = mcrun.sigmas[-1]
            self.structure.set_arrays(sigmas=sigmas)
        elif n_substitutions is not None:
            _, sigmas = self._scell.gen_random_decoration(
                nsubs={self._substitutional_sublattice: [n_substitutions]}
            )
            self.structure.set_arrays(sigmas=sigmas)
        elif initial_structure is not None:
            sigmas = initial_structure.get_sigmas()
            self.structure.set_arrays(sigmas=sigmas)
        elif initial_sigmas is not None:
            self.structure.set_arrays(sigmas=initial_sigmas)

        else:
            sigmas = self.structure.get_sigmas()

        scaledbeta = self._energy_scale_factor / self._kb / temperature

        with _timed("MonteCarloLite.metropolis: Compute energy of initial structure"):
            e = self._emodel.predict(self.structure)

        mcrun.add_step(0, sigmas, e)

        progress = tqdm(range(1, n_mc_steps + 1), total=n_mc_steps, desc="MMC sim.")

        use_arrays_backup_for_rejected_moves = (
            False  # Very slow, use only for benchmarking
        )

        i_reset = 0
        e_error = 0
        with _timed("Metropolis: MC steps"):
            for i in progress:
                atom_indices = []
                new_sigmas = []
                old_sigmas = []
                for j in range(n_clics):
                    if ensemble == "grandcanonical":
                        atom_index, sigma_initial, sigma_final = (
                            self.structure.flip_random(self._substitutional_sublattice)
                        )
                        atom_indices.append(atom_index)
                        old_sigmas.append(sigma_initial)
                        new_sigmas.append(sigma_final)
                    elif ensemble == "canonical":
                        atom_index1, sigma_initial1, sigma_final1 = (
                            self.structure.flip_random(self._substitutional_sublattice)
                        )
                        atom_index2, sigma_initial2, sigma_final2 = (
                            self.structure.flip_random(
                                self._substitutional_sublattice,
                                sigma_initial=sigma_final1,
                                sigma_final=sigma_initial1,
                            )
                        )

                        if (
                            atom_index1 not in atom_indices
                            and atom_index2 not in atom_indices
                        ):
                            atom_indices.append(atom_index1)
                            old_sigmas.append(sigma_initial1)
                            new_sigmas.append(sigma_final1)

                            atom_indices.append(atom_index2)
                            old_sigmas.append(sigma_initial2)
                            new_sigmas.append(sigma_final2)

                # compute new energy

                if use_arrays_backup_for_rejected_moves:
                    self.structure.backup_arrays()
                de = 0.0
                for atom_index, sigma in zip(atom_indices, new_sigmas):
                    de += self._emodel.predict_flip(
                        self.structure,
                        atom_index=atom_index,
                        new_sigma=sigma,
                        site_types=[self._substitutional_sublattice],
                    )
                    self.structure.update_arrays(
                        atom_indices=[atom_index], new_sigmas=[sigma]
                    )

                e1 = e + de

                if de <= 0:
                    accept_clic = True
                    boltzmann_factor = 0
                else:
                    boltzmann_factor = math.exp(-de * scaledbeta)

                    if np.random.uniform(0, 1) <= boltzmann_factor:
                        accept_clic = True
                    else:
                        accept_clic = False

                if accept_clic:
                    e = e1

                    sigmas = self.structure.get_sigmas()
                    mcrun.add_step(i, sigmas, e)

                else:
                    if use_arrays_backup_for_rejected_moves:
                        self.structure.restore_arrays()
                    else:
                        self.structure.update_arrays(
                            atom_indices=atom_indices[::-1], new_sigmas=old_sigmas[::-1]
                        )

                if (
                    n_error_reset is not None
                    and n_error_reset > 0
                    and i % n_error_reset == 0
                ):
                    e0 = e
                    e = self._emodel.predict(self.structure)
                    mcrun.energies[-1] = e
                    e_error = e - e0
                    i_reset = i

                ratio = self._compute_acceptance_ratio(mcrun, i, window=1000)
                progress.set_description(
                    f"MMC sim. | Acc. ratio: {ratio:.4f} | E-reset@{i_reset}: {e_error:.3e}"
                )

        mcrun.add_step(i, sigmas, e)  # record last step

        with _timed("Metropolis: serialization"):
            if mcrun_filepath is not None:
                mcrun.serialize(filepath=mcrun_filepath)

        return mcrun

    def serialize(self, filepath="mcsetup.pickle", fmt="pickle"):
        """Save the structure to a file in the specified format.

        Parameters
        ----------
        fmt : str, optional
            File format for output (default is "pickle").
        filepath : str, optional
            Path to the output file (default is "mcsetup.pickle").

        """
        file_ext = (
            os.path.splitext(filepath)[1].lower().lstrip(".")
        )  # remove leading dot
        fmt = fmt or file_ext  # use file extension as format if fmt is not provided

        if fmt == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
        else:
            raise NotImplementedError(f"Serialization format '{fmt}' is not supported.")


class MCRun:
    def __init__(self, plat, scshape, temperature, ensemble, n_steps):
        # Core metadata
        self.plat = plat
        self.scshape = scshape
        self.temperature = temperature
        self.ensemble = ensemble
        self.n_steps = n_steps

        # Trajectory data
        self.accepted_steps: List[int] = []
        self.sigmas: List[np.ndarray] = []
        self.energies: List[float] = []

        # Index offset for continuing runs
        self._step_offset: int = 0

    # --- Offset management -------------------------------------------------
    def set_step_offset(self, offset: int) -> None:
        """Manually set the step index offset used by `add_step`."""
        if offset < 0:
            raise ValueError("step offset must be non-negative")
        self._step_offset = offset

    def resume_from_last_step(self) -> None:
        """
        Set the step offset based on the last recorded step.
        """
        if not self.accepted_steps:
            self._step_offset = 0
            return
        self._step_offset = self.accepted_steps[-1] + 1

    @property
    def step_offset(self) -> int:
        """Current step index offset used by `add_step`."""
        return self._step_offset

    # --- Recording steps ---------------------------------------------------
    def add_step(self, step_index: int, sigmas, energy: float) -> None:
        """
        Append a Monte Carlo step to the run history.

        The stored step index is `step_index + step_offset`.

        Parameters
        ----------
        step_index : int
            Index relative to the *current segment* (e.g., 0..N during this run).
        sigmas : array_like
            Sigma values for the structure at this step.
        energy : float
            Energy associated with the step.
        """
        global_index = self._step_offset + int(step_index)
        self.accepted_steps.append(global_index)
        self.sigmas.append(np.array(sigmas, dtype=np.uint8))
        self.energies.append(float(energy))

    def serialize(self, filepath: str = None):
        """Serialize the MCRun object using pickle."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, filepath: str) -> MCRun:
        """Deserialize an MCRun object from a pickle file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        return (
            f"<MCRun(ensemble={self.ensemble}, T={self.temperature}, "
            f"n_steps={len(self.clics)}, n_accepted_steps={len(self.accepted_steps)})>"
            f"acceptance_ratios={len(self.clics) / len(self.accepted_steps) if len(self.accepted_steps) != 0 else 'Undefined'})>"
        )


def specific_heat(mc_setup, mc_run, n_eq: int = 1) -> Dict[str, float]:
    """
    Compute per-site specific heat from a Monte Carlo trajectory where only ACCEPTED
    steps were logged, using a numerically stable weighted Welford update.

    We treat each accepted energy E[j] as constant over its dwell time (number of MC
    steps until the next acceptance). Dwell lengths are the *weights* in the averages.

    Returns a dict with:
        - 'C'      : specific heat per site
        - 'E_mean' : mean energy over the production window
        - 'E_var'  : variance of energy over the production window (population variance)
        - 'n_prod' : number of MC steps in the production window
    """
    # Pull inputs
    accepted_energies = mc_run.energies
    accepted_indices = mc_run.accepted_steps
    T = mc_run.temperature
    kB = mc_setup._kb
    N = mc_setup._energy_scale_factor

    # Use highest practical precision available on this platform
    # (np.longdouble is >= float64 on many Unix platforms; on Windows it may equal float64)
    work_dtype = (
        np.longdouble
        if np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
        else np.float64
    )

    idx = np.asarray(accepted_indices, dtype=np.int64)
    E = np.asarray(accepted_energies, dtype=work_dtype)

    # Basic validation
    if idx.size == 0 or E.size == 0 or idx.size != E.size:
        raise ValueError(
            "accepted_indices and accepted_energies must be non-empty and of equal length."
        )
    if np.any(np.diff(idx) <= 0):
        raise ValueError("accepted_indices must be strictly increasing.")

    n_steps = int(
        idx[-1]
    )  # assume last logged accepted step equals total steps counter
    if n_eq < 0 or n_steps <= 0:
        raise ValueError("n_eq must be >= 0 and n_steps must be > 0.")
    if n_eq >= n_steps:
        raise ValueError("No production steps: n_eq must be less than n_steps.")
    if T <= 0 or N <= 0 or kB <= 0:
        raise ValueError("T, N, and kB must be positive.")

    # Production window [start_prod, end_prod)
    start_prod = int(n_eq)
    end_prod = int(n_steps)

    # Index of last acceptance at or before start_prod (could be -1 if none)
    j0 = np.searchsorted(idx, start_prod, side="right") - 1

    # Weighted Welford state
    W = work_dtype(0.0)  # total weight (production steps)
    mean = work_dtype(0.0)  # running mean
    M2 = work_dtype(0.0)  # running sum of weighted squared deviations
    M2_c = work_dtype(0.0)  # Kahan compensator for M2 (optional but cheap)

    def kahan_add(total, c, x):
        y = x - c
        t = total + y
        c_new = (t - total) - y
        return t, c_new

    def accumulate(dwell_len: int, energy: work_dtype):
        nonlocal W, mean, M2, M2_c
        w = work_dtype(dwell_len)
        if w <= 0:
            return
        W_new = W + w
        # Welford weighted update
        delta = energy - mean
        mean += (w * delta) / W_new
        delta2 = energy - mean
        incr = w * delta * delta2
        M2, M2_c = kahan_add(M2, M2_c, incr)
        W = W_new

    # Case A: we know the energy at start_prod
    if j0 >= 0:
        next_idx = int(idx[j0 + 1]) if (j0 + 1) < idx.size else end_prod
        dwell = max(0, min(next_idx, end_prod) - start_prod)
        accumulate(dwell, E[j0])

        # Subsequent segments within [start_prod, end_prod)
        for j in range(j0 + 1, idx.size):
            seg_start = int(idx[j])
            if seg_start >= end_prod:
                break
            seg_end = int(idx[j + 1]) if (j + 1) < idx.size else end_prod
            if seg_end <= start_prod:
                continue
            dwell = min(seg_end, end_prod) - max(seg_start, start_prod)
            accumulate(dwell, E[j])
    else:
        # Case B: no acceptance before start_prod → start at first acceptance ≥ start_prod
        j1 = np.searchsorted(idx, start_prod, side="left")
        for j in range(j1, idx.size):
            seg_start = int(idx[j])
            if seg_start >= end_prod:
                break
            seg_end = int(idx[j + 1]) if (j + 1) < idx.size else end_prod
            dwell = min(seg_end, end_prod) - seg_start
            accumulate(dwell, E[j])

    if W <= 0:
        raise ValueError(
            "No production data after applying n_eq; check indices and n_steps."
        )

    # Population (time-average) variance over production window
    E_mean = float(mean)
    E_var = float(max(work_dtype(0.0), M2 / W))
    C = E_var / (N * kB * (T**2))

    return {"C": float(C), "E_mean": E_mean, "E_var": E_var, "n_prod": float(W)}
