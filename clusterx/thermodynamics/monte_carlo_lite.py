# Copyright (c) 2015-2025, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
from __future__ import annotations

import os
import pickle
import random
import warnings
from typing import Optional

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

    def metropolis(
        self,
        temperature=None,
        n_mc_steps=None,
        n_clics=1,
        ensemble=None,
        n_substitutions=None,
        chemical_potential=None,
        mcrun_filepath=None,
        n_error_reset=None,
        initial_structure=None,
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


        ``n_substitutions``: integer (default = None)
            Defines the number of substituted atoms in the sublattice

        ``chemical_potential``: dictionary (default: None)
            Define the chemical potential in the grand canonical ensemble.

        ``mcrun_filepath``: string (default: ``trajectory.json``)
            Name of a Json file in which the trajectory is serialized after the sampling if ``serialize`` is **True**.

        ``n_error_reset``: integer (default: None)
            If not **None*  and ``predict_swap`` equal to **True**, the correlations are calculated as usual (no differences) every n-th step.

        ``initial_structure``: Structure object
            Initial structure, from which the sampling starts.
            If ``None``, sampling starts from a random structure with ``n_substitutions`` substitutions.

        ``random_seed``: Integer or None (default: None)
            Random seed to produce reproducible results

        ``**kwargs``: keyworded argument list, arbitrary length
            These arguments are added to the MonteCarloTrajectory object that is initialized in this method.

        **Returns**: MonteCarloTrajectory object
            Trajectory containing the complete information of the sampling trajectory.

        """
        import math

        from tqdm import tqdm

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        with _timed("MonteCarloLite.metropolis: Generating initial random structure"):
            if initial_structure is None and n_substitutions is not None:
                _, sigmas = self._scell.gen_random_decoration(
                    nsubs={self._substitutional_sublattice: [n_substitutions]}
                )
                self.structure.set_arrays(sigmas=sigmas)
            elif initial_structure is not None and n_substitutions is None:
                sigmas = initial_structure.get_sigmas()
                self.structure.set_arrays(sigmas=sigmas)
            else:
                pass

            # if initial_structure is None:
            #     if n_substitutions is not None:
            #         _, sigmas = self._scell.gen_random_decoration(
            #             nsubs={self._substitutional_sublattice: [n_substitutions]}
            #         )
            #     else:
            #         raise ValueError(
            #             "MonteCarloLite.metropolis: please provide n_substitutions or initial_structure."
            #         )
            # else:
            #     sigmas = initial_structure.get_sigmas()

            # self.structure.set_arrays(sigmas=sigmas)

        scaledbeta = self._energy_scale_factor / self._kb / temperature

        with _timed(
            "MonteCarloLite.metropolis: Compute energy of initial random structure"
        ):
            e = self._emodel.predict(self.structure)

        mcrun_filepath
        if mcrun_filepath is None:
            warnings.warn(
                "No file path provided for storing the MC run. Trajectory will not be saved to a file.",
                UserWarning,
            )

        mcrun = MCRun(
            self._scell.get_parent_lattice(),
            self._scell.get_transformation(),
            temperature,
            ensemble,
            n_steps=n_mc_steps,
        )

        mcrun.accepted_steps.append(0)
        mcrun.sigmas.append(tuple(self.structure.get_sigmas()))
        mcrun.energies.append(e)

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

                    mcrun.accepted_steps.append(i)
                    mcrun.sigmas.append(
                        np.array(self.structure.get_sigmas(), dtype=np.uint8)
                    )
                    mcrun.energies.append(e)
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

                ratio = len(mcrun.accepted_steps) / i
                progress.set_description(
                    f"MMC sim. | Acc. ratio: {ratio:.4f} | E-reset@{i_reset}: {e_error:.3e}"
                )

        with _timed("Metropolis: serialization"):
            if mcrun_filepath is not None:
                mcrun.serialize(filepath=mcrun_filepath)

        return mcrun, self.structure

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
        # self.scell = scell
        self.plat = plat
        self.scshape = scshape
        self.temperature = temperature
        self.ensemble = ensemble
        self.n_steps = n_steps

        self.accepted_steps = []
        self.sigmas = []
        self.energies = []
        self.clics = []

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


from typing import Dict


def specific_heat(
    mc_setup,
    mc_run,
    n_eq: int = 1,
    n_steps: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute the (per-site) specific heat from a Monte Carlo trajectory where
    only ACCEPTED steps were logged.

    The energy after an accepted move is assumed to remain constant until the
    next accepted move (i.e., rejected steps do not change the energy).
    We reconstruct the production-time averages by weighting each accepted
    energy by its 'dwell time' (the number of MC steps it persisted).

    Parameters
    ----------
    mc_setup: MonteCarloLite
        an MC initialized instance
    mc_run: MCRun
        the output of mc_setup.metropolis
    n_eq : int
        Number of initial MC steps to discard as equilibration (i.e., discard steps [0, n_eq)).

    Returns
    -------
    dict with keys:
        - 'C'      : specific heat per site
        - 'E_mean' : mean energy over the production window
        - 'E_var'  : variance of energy over the production window
        - 'n_prod' : number of MC steps in the production window (after discarding n_eq)

    Notes
    -----
    Specific heat is computed as:
        C = (⟨E²⟩ - ⟨E⟩²) / (N * kB * T²)

    The averages ⟨·⟩ are time averages over MC steps in the production window,
    reconstructed via dwell-time weighting between accepted moves.

    Edge cases handled:
      * If there is an accepted move before n_eq, its energy is used as the
        starting energy at step n_eq.
      * If the first accepted move occurs after n_eq and there is no earlier
        accepted move, production starts at that first accepted move (since the
        prior energy is unknown from the provided data).
      * If n_eq >= n_steps, an error is raised.
    """
    accepted_energies = mc_run.energies
    accepted_indices = mc_run.accepted_steps
    T = mc_run.temperature
    kB = mc_setup._kb
    N = mc_setup._energy_scale_factor
    idx = np.asarray(accepted_indices, dtype=np.int64)
    E = np.asarray(accepted_energies, dtype=np.float64)
    if idx.size == 0 or E.size == 0 or idx.size != E.size:
        raise ValueError(
            "accepted_indices and accepted_energies must be non-empty and of equal length."
        )
    if np.any(np.diff(idx) <= 0):
        raise ValueError("accepted_indices must be strictly increasing.")

    if n_steps is None:
        if hasattr(mc_run, "n_steps"):
            n_steps = mc_run.n_steps
        else:
            raise ValueError("n_steps (total number of MC steps) must be provided.")

    if n_eq < 0 or n_steps <= 0:
        raise ValueError("n_eq must be >= 0 and n_steps must be > 0.")
    if n_eq >= n_steps:
        raise ValueError("No production steps: n_eq must be less than n_steps.")
    if T <= 0 or N <= 0 or kB <= 0:
        raise ValueError("T, N, and kB must be positive.")

    # Determine 0-based vs 1-based indexing automatically:
    # We'll interpret indices as they are, but all step ranges are half-open [start, end).
    # User must ensure n_eq and n_steps are on the same convention as 'idx'.
    # (If you used 1-based steps in logging, pass matching n_eq and n_steps.)
    start_prod = n_eq
    end_prod = n_steps

    # Find the index of the last accepted move at or before start_prod.
    j0 = (
        np.searchsorted(idx, start_prod, side="right") - 1
    )  # could be -1 if none before start_prod

    sum_w = 0.0  # total production steps
    sum_E = 0.0  # Σ (dwell * E)
    sum_E2 = 0.0  # Σ (dwell * E^2)

    def accumulate(dwell_len: int, energy: float):
        nonlocal sum_w, sum_E, sum_E2
        if dwell_len <= 0:
            return
        sum_w += dwell_len
        sum_E += dwell_len * energy
        sum_E2 += dwell_len * (energy * energy)

    # Case A: we have an accepted move before or exactly at start_prod
    if j0 >= 0:
        # The energy at start_prod is E[j0], and it lasts until the next acceptance (or end).
        next_idx = idx[j0 + 1] if (j0 + 1) < idx.size else end_prod
        dwell = max(0, min(next_idx, end_prod) - start_prod)
        accumulate(dwell, E[j0])

        # Now process subsequent full segments between acceptances in [start_prod, end_prod)
        for j in range(j0 + 1, idx.size):
            seg_start = idx[j]
            if seg_start >= end_prod:
                break
            seg_end = idx[j + 1] if (j + 1) < idx.size else end_prod
            if seg_end <= start_prod:
                continue
            dwell = min(seg_end, end_prod) - max(seg_start, start_prod)
            accumulate(dwell, E[j])

    else:
        # Case B: no accepted move before start_prod.
        # We don't know the energy at start_prod, so start at the first acceptance >= start_prod.
        j1 = np.searchsorted(idx, start_prod, side="left")
        for j in range(j1, idx.size):
            seg_start = idx[j]
            if seg_start >= end_prod:
                break
            seg_end = idx[j + 1] if (j + 1) < idx.size else end_prod
            dwell = min(seg_end, end_prod) - seg_start
            accumulate(dwell, E[j])

    if sum_w <= 0:
        raise ValueError(
            "No production data after applying n_eq; check indices and n_steps."
        )

    E_mean = sum_E / sum_w
    E_var = max(0.0, (sum_E2 / sum_w) - E_mean**2)  # numerical safety
    C = E_var / (N * kB * (T**2))

    return {"C": C, "E_mean": E_mean, "E_var": E_var, "n_prod": float(sum_w)}
