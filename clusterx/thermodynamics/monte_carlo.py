# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

## packages needed for MonteCarloTrajectory
import json
from io import StringIO
from copy import deepcopy
import math
import os
import operator
import random

from tqdm import tqdm
import numpy as np
from ase.cell import Cell
from ase.io import Trajectory, iread, read, write

from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.model import Model
from clusterx.utils import dict_compare, RunningAverage


class MonteCarlo:
    r"""Monte Carlo class

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

    ``ensemble``: string (default: ``canonical``)
        ``canonical`` allows for swaps of atoms that conserve the concentration defined with ``nsubs``.

        ``semigrand-canonical`` allows replacing atoms in sub-lattices defined with ``sublattice_indices``
        with other atoms in said sub-lattice.

    ``nsubs``: dictionary (default = None)
        Defines the number of substituted atoms in each sub-lattice of the
        Supercell in which the sampling is performed.

        The format of the dictionary is as follows::

            {site_type1:[n11,n12,...], site_type2:[n21,n22,...], ...}

        It indicates how many substitutional atoms of every kind (n#1, n#2, ...)
        may replace the pristine species for sites of site_type#
        (see related documentation in SuperCell object).

    ``sublattice_indices``: list of integers (default = None)
        Defines the sublattices for the grand canonical sampling.
        Furthermore, it can be used to limit the canonical sampling
        to a reduced number of sublattices. E.g. in the case of nsubs = {0:[4,6], 1:[4]}. Here, sublattices 0 and 1
        contain substitutional sites, but only a sampling in sublattice 0 is wanted.
        Then, put ``sublattice_indices`` = [0].

    ``chemical_potentials``: dictionary (default: None)
        Define the chemical potentials used for samplings in the grand canonical ensemble.

        The format of the dictionary is as follows

        {site_type1:[:math:`\Delta \mu_{11}`, :math:`\Delta \mu_{12}`,...],
         site_type2:[:math:`\Delta \mu_{21}`, :math:`\Delta \mu_{22}`,...],
         ...}

        Here, :math:`\Delta \mu_{\#i}` is the chemical potential difference of substitutional species i relative
        to the pristine species in sub-lattice with site_type#.

    ``models``: List of Model objects
        The properties returned from these Model objects are additionally
        calculated during the sampling and stored with their corresponding ``prop_name``
        to the dictionary ``key_value_pairs`` for each visited structure during the sampling.

        Properties from Model obejects can also be calculated after the
        sampling by using the MonteCarloTrajectory class.

    ``predict_swap``: boolean (default: False)
       If set to **True**, this parameter makes the sampling faster by calculating the correlation difference of the
       proposed structure with respect to the previous structure.

    ``reduce_structure``: bool (default: False)
        If true, cluster orbits/correlations are computed and evaluated on
        subset of sigmas, in a box around central site in the structure.
        Incompatible with non-diagonal supercell shapes.

    ``energy_key``: string (default: ``energy``)
        For a multi-linear model, one of the predictions needs to be the energy prediction,
        energy_key needs to match one of the strings in energy_model.property_names, so that the energy
        prediction can be indexed from prediction vector.

    ``filename``: string (default: ``trajectory.json``)
        Path at which to store the MonteCarloTrajectory.

    ``seed``: int or None (default: None)
        seed value for random number generators. For reproducible results,
        pass integer seed, otherwise seed=None

    ``resume_from``: string or path-like (default: None)
        Start from the final complete frame of an XYZ/extxyz trajectory. The
        physical configuration is resumed with the random streams initialized
        from ``seed``; the original random stream is not restored.

    ``save_atoms_endpoints``: boolean (default: False)
        Write the initial configuration and each metropolis call's endpoint.
        Continuing calls share one stage boundary frame. Periodic checkpoints
        set by ``save_atoms_every`` follow the chain's proposal counter.

    ``atoms_mode``: string (default: "auto")
        Apply once per chain: "x" requires a new output file, "w" overwrites,
        "a" appends, and "auto" overwrites fresh runs and appends resumes.
        XYZ counters start at zero for each fresh start or file-based resume.

    """

    def __init__(
        self,
        energy_model: Model,
        scell: SuperCell,
        nsubs: dict | None = None,
        ensemble: str = "canonical",
        sublattice_indices: list | None = None,
        chemical_potentials: dict | None = None,
        models: list = [],
        predict_swap: bool = False,
        reduce_structure: bool = False,
        energy_key: str = "energy",
        save_atoms_every: int = 0,
        atoms_filename: str = "metropolis.traj",
        filename: str = "trajectory.json",
        seed: int | None = None,
        resume_from: str | os.PathLike | None = None,
        save_atoms_endpoints: bool = False,
        atoms_mode: str = "auto",
    ):
        if (ensemble == "semigrand-canonical") and (chemical_potentials is None):
            raise ValueError("`chemical_potentials` have to be given when ensemble is 'semigrand-canonical")
        self._emodel = energy_model
        self._emodel.reset_mc(mc=True)
        self._scell = scell
        self._nsubs = nsubs
        self._filename = filename
        self._sublattice_indices = sublattice_indices
        self._chemical_potentials = chemical_potentials
        self._models = models
        self._ensemble = ensemble
        self._predict_swap = predict_swap
        self._reduce_structure = reduce_structure
        self._save_atoms_every = save_atoms_every
        self._atoms_filename = os.fspath(atoms_filename)
        if atoms_mode not in {"auto", "w", "a", "x"}:
            raise ValueError("atoms_mode must be 'auto', 'w' (overwrite), 'a' (append), or 'x' (new file)")
        if save_atoms_every < 0:
            raise ValueError("save_atoms_every must be nonnegative")
        self._save_atoms_endpoints = save_atoms_endpoints
        self._atoms_mode = atoms_mode
        self._atoms_output_path = None
        self._atoms_last_step = None
        self._resume_from = resume_from
        self._resume_source = os.path.abspath(os.fspath(resume_from)) if resume_from is not None else None
        self._chain_state = None
        self._pending_initial_decoration = None
        # Use two rngs, numpy.rng produces MC acceptance probability,
        # random.Random produces hashable values for random flip sites
        # SFC64 should be fastest bit generator in numpy at time of writing
        self._rng_np = np.random.Generator(np.random.SFC64(seed))
        # self._rng_np = np.random.default_rng(seed) # fall-back option
        self._rng_std: random.Random = random
        self._rng_std.seed(seed)

        prop = energy_model.property_name
        if prop is None or len(prop) == 0 or isinstance(prop, str):
            # Note: indexing with None only works for numpy arrays, not lists
            self._energy_prediction_index = None
        else:
            self._energy_prediction_index = prop.index(energy_key)

    def metropolis(
        self,
        no_of_sampling_steps: int = 100,
        scale_factor: list[float] = [1.0],
        temperature: float | list[float] = 1.0,
        boltzmann_constant: float = 1.0,
        initial_decoration: list[int] | None = None,
        target_rate: float | None = None,
        n_clics: int = 1,
        serialize: bool = False,
        filename: str = "trajectory.json",
        nonlinear: bool = False,
        max_steps_per_batch: int = 100_000,
        continue_chain: bool = False,
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

        ``no_of_sampling_steps``: integer
            Number of sampling steps

        ``temperature``: float or list of float
            Temperature at which the sampling is performed. If list, length of
            list must be equal to `no_of_sampling_steps`.

        ``boltzmann_constant``: float
            Boltzmann constant

        ``scale_factor``: list of floats
            List is used to adjust the factor :math:`k_B T` to the same units as the energy from ``energy_model``.

            All floats in list are multiplied by the factor :math:`k_B T`.
            If list is empty, the factor :math:`k_B T` remains unchanged.

        ``initial_decoration``: list of integers
            Atomic numbers of the initial structure, from which the sampling starts.
            If ``None``, sampling starts with a structure randomly generated.

        ``target_rate``: float (default: None)
            Real number between 0 and 1. Represents the targeted acceptance rate.
            If not ``None``, the initial temperature will be adjusted to match the given
            acceptance rate. The acceptance ratio during the simulation is computed using
            the last 1000 moves.

        ``n_clics``: integer
            Number of species flips (grandcanonical) or species swaps (canonical) per sampling step.
            Defaults to 1.

        ``serialize``: boolean (default: False)
            Serialize the MonteCarloTrajectory object into a Json file after the sampling.

        ``filename``: string (default: ``trajectory.json``)
            Name of a Json file in which the trajectory is serialized after the sampling if ``serialize`` is **True**.

        ``nonlinear``: boolean (default: False)
            Whether or not the model is nonlinear with respect to cluster
            correlations. If True, cache global correlations and the current
            model energy, evaluating the trial correlations once per proposal.
            Compatible with reduce_structure. Recognized affine models retain
            their direct energy-difference path even when this is True. If
            False, use the direct linear energy-difference path.

        ``max_steps_per_batch``: integer (default: 100000)
            Maximum number of proposal temperatures held in a temporary array at
            once. The Markov chain and random-number streams continue across these
            internal batches.

        ``continue_chain``: boolean (default: False)
            Continue the configuration and cached nonlinear correlation state from
            the previous call. The default preserves the historical behavior of
            starting a new configuration for each call.

        ``**kwargs``: keyworded argument list, arbitrary length
            These arguments are added to the MonteCarloTrajectory object that is initialized in this method.

        **Returns**: MonteCarloTrajectory object
            Trajectory containing the complete information of the sampling trajectory.

        """
        scale_factor = scale_factor if scale_factor is not None else [self._scell.get_index()]

        try:
            max_steps_per_batch = operator.index(max_steps_per_batch)
        except TypeError as exc:
            raise ValueError("max_steps_per_batch must be a positive integer") from exc
        if max_steps_per_batch < 1:
            raise ValueError("max_steps_per_batch must be a positive integer")
        temperature_array = None if np.ndim(temperature) == 0 else temperature
        if temperature_array is not None and len(temperature_array) != no_of_sampling_steps:
            raise ValueError("temperature must be a scalar or have one value per sampling step")
        if np.ndim(temperature) == 0 and (not np.isfinite(temperature) or temperature <= 0):
            raise ValueError("temperature must be finite and positive")
        scale_factor_product = np.prod(scale_factor)

        continuing = continue_chain and self._chain_state is not None and initial_decoration is None
        resume_frame = None
        if not continuing:
            if initial_decoration is None and self._pending_initial_decoration is not None:
                initial_decoration = self._pending_initial_decoration
                self._pending_initial_decoration = None
            if self._resume_from is not None and initial_decoration is None:
                resume_frame = self._read_resume_frame(self._resume_from)
                initial_decoration = resume_frame.numbers
                self._validate_resume_frame(resume_frame)
                self._resume_from = None
            elif self._resume_from is not None:
                raise ValueError("Specify either resume_from or initial_decoration, not both")
            self.initialize_structure(initial_decoration, self._nsubs)
        structure = self.structure
        sigmas = structure.get_sigmas()
        use_correlations = nonlinear and self._predict_swap
        if use_correlations and (self._emodel.estimator is None or self._emodel._get_affine_parameters(
                len(self._emodel.corrc.get_cpool())) is not None):
            use_correlations = False

        if continuing:
            use_correlations = self._chain_state["use_correlations"]
            correlations = self._chain_state.get("correlations")
            current_energy = self._chain_state["current_energy"]
            e_last = current_energy
        elif self._reduce_structure:
            self._emodel.init_reduced_model(structure)
            self._emodel._init_interaction_dict(self._emodel.scell_reduced)
            structure.init_reduced_structure(self._emodel.scell_reduced)
            if not use_correlations:
                current_energy = 0.0
            if self._predict_swap:
                self._check_reduced_local_geometry()
        elif not use_correlations:
            e_last = self._energy_value(self._emodel.predict(structure))
            current_energy = e_last
            self._emodel._init_interaction_dict(self._scell)

        if use_correlations and not continuing:
            correlations = self._emodel.get_correlations(structure, reduce_structure=self._reduce_structure)
            # Estimator transforms may operate in place; keep MC's state private.
            current_energy = self._energy_value(self._emodel.predict_on_correlations(correlations.copy()))
            e_last = current_energy
        elif not continuing and not self._predict_swap:
            # Keep an absolute prediction separately from reduced trajectories,
            # whose recorded energies remain relative to their initial state.
            if self._reduce_structure:
                current_energy = self._energy_value(self._emodel.predict(structure, reduce_structure=True))
            else:
                current_energy = e_last
        if self._reduce_structure:
            e_last = np.float64(0.0)

        global_step_offset = self._chain_state["global_step"] if continuing else 0
        stage = self._chain_state.get("stage", 1) + 1 if continuing else 1
        counter_origin = self._chain_state["counter_origin"] if continuing else (
            "xyz_resume" if resume_frame is not None else "fresh_start"
        )
        save_atoms = self._save_atoms_every > 0 or self._save_atoms_endpoints
        if save_atoms:
            write_initial = self._prepare_atoms_output(continuing, resume_frame is not None)
            initial_temperature = temperature if temperature_array is None else (
                temperature_array[0] if no_of_sampling_steps else None
            )
            if write_initial:
                self._write_atoms_checkpoint(global_step_offset, stage, 0, initial_temperature, counter_origin)

        traj = MonteCarloTrajectory(
            self._scell,
            filename=filename,
            models=self._models,
            no_of_sampling_steps=no_of_sampling_steps,
            temperature=temperature,
            boltzmann_constant=boltzmann_constant,
            scale_factor=scale_factor,
            target_rate=target_rate,
            **kwargs,
        )
        key_value_pairs = {}
        for m, mo in enumerate(self._models):
            key_value_pairs.update({mo.property_name: mo.predict(structure)})
        traj.add_step(
            step=0,
            energy=e_last,
            indices_list=[],
            new_sigmas=[],
            sigmas=sigmas.copy(),
            key_value_pairs=key_value_pairs,
        )

        progress = tqdm(
            range(1, no_of_sampling_steps + 1),
            total=no_of_sampling_steps,
            desc="MMC sim.",
        )
        rave = self._chain_state["running_average"] if continuing else RunningAverage(1000)
        beta_scale = self._chain_state["beta_scale"] if continuing else 1.0
        beta_chunks = iter(self._iter_beta_chunks(
            no_of_sampling_steps, temperature, temperature_array, boltzmann_constant, max_steps_per_batch
        ))
        beta_chunk_start, beta_chunk = next(beta_chunks, (0, np.empty(0)))
        for step in progress:
            beta_index = step - 1
            if beta_index >= beta_chunk_start + len(beta_chunk):
                beta_chunk_start, beta_chunk = next(beta_chunks)
            flip_tuples = []  # store tuples with (index, new_sigma, old_sigma, site_type)

            if self._ensemble == "canonical":
                for _ in range(n_clics):
                    i, j, site_type = structure.get_random_swap(self._sublattice_indices, self._rng_std)
                    flip_tuples.append((i, sigmas[j], sigmas[i], site_type))
                    flip_tuples.append((j, sigmas[i], sigmas[j], site_type))
            elif self._ensemble == "semigrand-canonical":
                for _ in range(n_clics):
                    i, sigma_initial, sigma_final, site_type = structure.get_random_flip(
                        self._sublattice_indices, rng=self._rng_std
                    )
                    flip_tuples.append((i, sigma_final, sigma_initial, site_type))
            else:
                raise NotImplementedError(f"Unknown ensemble: {self._ensemble}")

            de = 0.0
            if self._predict_swap:
                if use_correlations:
                    correlations_diff = np.zeros_like(correlations)
                for i, new_sigma, old_sigma, site_type in flip_tuples:
                    if use_correlations:
                        correlations_diff += self._emodel.predict_correlations_diff(
                            structure, i, old_sigma, new_sigma, reduce_structure=self._reduce_structure,
                        )
                    else:
                        de += self._emodel.predict_flip(
                            structure=structure,
                            index=i,
                            old_sigma=old_sigma,
                            new_sigma=new_sigma,
                            reduce_structure=self._reduce_structure,
                        )
                    structure.flip(i, old_sigma, new_sigma, site_type)
                if use_correlations:
                    trial_correlations = correlations + correlations_diff
                    trial_energy = self._energy_value(
                        self._emodel.predict_on_correlations(trial_correlations.copy())
                    )
                    de = trial_energy - current_energy
            else:
                for i, new_sigma, old_sigma, site_type in flip_tuples:
                    structure.flip(i, old_sigma, new_sigma, site_type)
                trial_energy = self._energy_value(
                    self._emodel.predict(structure, reduce_structure=self._reduce_structure)
                )
                de = trial_energy - current_energy

            dmu = 0.0  # sum up chemical potential change
            if self._ensemble == "semigrand-canonical":
                for i, new_sigma, old_sigma, site_type in flip_tuples:
                    dmu += (
                        self._chemical_potentials[site_type][new_sigma]
                        - self._chemical_potentials[site_type][old_sigma]
                    )

            # handle multi-linear model, make sure that `de` is np.float64
            de = de if len(de.shape) == 0 else de[self._energy_prediction_index]

            if de + dmu <= 0:
                accept_step = True
            elif self._rng_np.random() <= np.exp(-beta_chunk[beta_index - beta_chunk_start]
                                                 * beta_scale * (de * scale_factor_product + dmu)):
                accept_step = True
            else:
                accept_step = False

            if accept_step:
                e_last += de
                if use_correlations:
                    correlations = trial_correlations
                if use_correlations or not self._predict_swap:
                    current_energy = trial_energy
                elif not self._reduce_structure:
                    current_energy += de

                key_value_pairs = {}
                for m, mo in enumerate(self._models):
                    key_value_pairs.update({mo.property_name: mo.predict(structure)})
                if serialize:
                    traj.add_step(
                        step=step,
                        energy=e_last,
                        indices_list=[i for i, _, _, _ in flip_tuples],
                        new_sigmas=[new_sigma for _, new_sigma, _, _ in flip_tuples],
                        key_value_pairs=key_value_pairs,
                    )
            else:
                # revert flips
                for i, new_sigma, old_sigma, site_type in flip_tuples[::-1]:
                    structure.flip(i, new_sigma, old_sigma, site_type)

            rate = rave.update_and_get_average(1.0 * accept_step)
            if target_rate is not None:
                global_step = global_step_offset + step
                if global_step % 10 == 0 and global_step >= 1000:
                    beta_scale /= math.exp((target_rate - rate) / 10.0)

            if self._save_atoms_every > 0 and (global_step_offset + step) % self._save_atoms_every == 0:
                frame_temperature = temperature if temperature_array is None else temperature_array[step - 1]
                self._write_atoms_checkpoint(
                    global_step_offset + step, stage, step, frame_temperature, counter_origin, de
                )

            if step % 1000 == 0 or no_of_sampling_steps < 1000:
                # if updated too often, progress display might delay steps
                progress.set_description(f"MMC sim. | Acc. rate: {rate:.4f} | delta_E: {de:.1e}")

        if self._save_atoms_endpoints:
            final_temperature = temperature if temperature_array is None else (
                temperature_array[-1] if no_of_sampling_steps else None
            )
            self._write_atoms_checkpoint(
                global_step_offset + no_of_sampling_steps, stage, no_of_sampling_steps,
                final_temperature, counter_origin,
            )
        if serialize:
            traj.serialize()

        self._chain_state = {
            "use_correlations": use_correlations,
            "correlations": correlations.copy() if use_correlations else None,
            "current_energy": current_energy,
            "running_average": rave,
            "beta_scale": beta_scale,
            "global_step": global_step_offset + no_of_sampling_steps,
            "stage": stage,
            "counter_origin": counter_origin,
        }

        return traj

    def _prepare_atoms_output(self, continuing, resumed):
        """Apply the output mode once per chain, then append subsequent stages."""
        path = os.path.abspath(self._atoms_filename)
        if continuing and path == self._atoms_output_path:
            return False
        mode = self._atoms_mode
        if mode == "auto":
            mode = "a" if continuing or resumed else "w"
        if mode == "x" and os.path.exists(path):
            raise FileExistsError(f"Checkpoint output already exists: {path}. Use atoms_mode='a' or 'w' explicitly.")
        self._atoms_write_mode = mode
        self._atoms_output_path = path
        self._atoms_last_step = None
        if resumed and path == self._resume_source and mode == "a":
            # The source's final frame already contains this initial configuration.
            self._atoms_last_step = 0
            if getattr(self, "_resume_complete_bytes", None) is not None:
                with open(path, "r+b") as output:
                    output.truncate(self._resume_complete_bytes)
        return True

    def _write_atoms_checkpoint(self, step, stage, stage_step, temperature, origin, energy=None):
        """Snapshot sigmas without changing index maps, correlations, or random streams."""
        if step == self._atoms_last_step:
            return
        atoms = self.structure.get_atoms()
        atoms.numbers = self.structure.get_decoration()
        nsites = self._scell.get_n_sub_sites(unique=False)
        atoms.info.update(mc_step=int(step), mc_stage=int(stage), mc_stage_step=int(stage_step),
                          mc_counter_origin=origin)
        if nsites:
            atoms.info["mc_sweeps"] = float(step / nsites)
            atoms.info["mc_sweep_sites"] = int(nsites)
        if temperature is not None:
            atoms.info["temperature"] = float(temperature)
        append = self._atoms_write_mode == "a"
        if self._atoms_filename.endswith((".xyz", ".extxyz")):
            write(self._atoms_filename, atoms, format="extxyz", append=append)
        else:
            # Preserve the historical ASE trajectory convention of omitting spectators.
            atoms = atoms[self._scell.get_substitutional_sites()]
            with Trajectory(self._atoms_filename, mode="a" if append else "w") as trajectory:
                if energy is None:
                    trajectory.write(atoms)
                else:
                    trajectory.write(atoms, energy=energy)
        self._atoms_write_mode = "a"
        self._atoms_last_step = step

    @staticmethod
    def _iter_beta_chunks(no_steps, temperature, temperature_array, boltzmann_constant, batch_size):
        """Yield proposal-temperature arrays whose length is bounded by batch_size."""
        for start in range(0, no_steps, batch_size):
            stop = min(start + batch_size, no_steps)
            values = (np.full(stop - start, temperature, dtype=float) if temperature_array is None
                      else np.asarray(temperature_array[start:stop], dtype=float))
            if np.any(~np.isfinite(values)) or np.any(values <= 0):
                raise ValueError("temperature values must be finite and positive")
            yield start, 1.0 / (boltzmann_constant * values)

    def reset_chain(self, initial_decoration=None, seed=None, resume_from=None):
        """Reset the Markov chain, optionally selecting a new start or XYZ frame."""
        self._chain_state = None
        self._atoms_output_path = None
        self._atoms_last_step = None
        self._resume_from = resume_from
        self._resume_source = os.path.abspath(os.fspath(resume_from)) if resume_from is not None else None
        self._pending_initial_decoration = (
            list(initial_decoration) if initial_decoration is not None else None
        )
        self._emodel.reset_mc(mc=True)
        self._rng_np = np.random.Generator(np.random.SFC64(seed))
        self._rng_std.seed(seed)

    def _read_resume_frame(self, filename):
        """Read only the last complete XYZ frame, tolerating an interrupted final write."""
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Resume checkpoint not found: {filename}")
        self._resume_complete_bytes = None
        try:
            if os.fspath(filename).endswith((".xyz", ".extxyz")):
                # Scan with constant storage, then parse just the final complete
                # frame. ASE's whole-file indexing can fail on a partial tail.
                last_bounds = previous_bounds = None
                with open(filename, "rb") as source:
                    while True:
                        start = source.tell()
                        line = source.readline()
                        try:
                            count = int(line)
                        except ValueError:
                            break
                        if count < 0:
                            break
                        for _ in range(count + 1):  # Comment plus atom records.
                            if not source.readline():
                                break
                        else:
                            previous_bounds, last_bounds = last_bounds, (start, source.tell())
                            continue
                        break
                    if last_bounds is None:
                        raise ValueError("no complete frame")
                    for bounds in (last_bounds, previous_bounds):
                        if bounds is None:
                            continue
                        start, end = bounds
                        source.seek(start)
                        try:
                            atoms = read(StringIO(source.read(end - start).decode()), format="extxyz")
                        except (OSError, ValueError, IndexError):
                            if bounds == previous_bounds or previous_bounds is None:
                                raise
                        else:
                            self._resume_complete_bytes = end
                            return atoms
            last = None
            for frame in iread(filename, index=":"):
                last = frame
            if last is None:
                raise ValueError("no complete frame")
            return last
        except (OSError, ValueError, IndexError) as exc:
            raise ValueError(f"Cannot resume checkpoint {filename!r}: {exc}") from exc

    def _validate_resume_frame(self, atoms):
        if self._emodel.get_parent_lattice() != self._scell.get_parent_lattice():
            raise ValueError("Monte Carlo model parent lattice does not match the requested supercell")
        if len(atoms) != len(self._scell):
            raise ValueError("Resume frame site count does not match the Monte Carlo supercell")
        if not np.allclose(atoms.cell.array, self._scell.get_cell().array, atol=1e-6, rtol=0):
            raise ValueError("Resume frame cell does not match the Monte Carlo supercell")
        if not np.array_equal(atoms.pbc, self._scell.get_pbc()):
            raise ValueError("Resume frame periodicity does not match the Monte Carlo supercell")
        if "tags" in atoms.arrays and not np.array_equal(atoms.get_tags(), self._scell.get_tags()):
            raise ValueError("Resume frame sublattice tags do not match the Monte Carlo supercell")
        if not np.allclose(atoms.positions, self._scell.get_positions(), atol=1e-5, rtol=0):
            raise ValueError("Resume frame positions or site ordering do not match the Monte Carlo supercell")
        tmat = atoms.info.get("tmat")
        if tmat is not None and not np.array_equal(np.asarray(tmat), self._scell.get_transformation()):
            raise ValueError("Resume frame transformation does not match the Monte Carlo supercell")
        # Structure validates that each loaded atomic number is allowed at its site.
        try:
            structure = Structure(self._scell, decoration=atoms.numbers, track_indices=True)
        except AttributeError as exc:
            raise ValueError("Resume frame contains species not allowed by the model parent lattice") from exc
        if self._ensemble == "canonical" and self._nsubs is not None:
            conc = structure.get_fractional_concentrations()
            nsites = structure.get_nsites_per_type()
            composition = {key: [int(round(value * nsites[key])) for value in values[1:]]
                           for key, values in conc.items()}
            if not dict_compare(composition, self._nsubs):
                raise ValueError("Resume frame composition does not match canonical nsubs")

    def _energy_value(self, prediction):
        prediction = np.asarray(prediction)
        return prediction[()] if prediction.ndim == 0 else prediction[self._energy_prediction_index]

    def _check_reduced_local_geometry(self):
        """Reject local updates that would miss other images of the flipped atom."""
        full = tuple(self._scell.get_transformation().diagonal())
        reduced = tuple(self._emodel.scell_reduced.get_transformation().diagonal())
        if all(n >= r for n, r in zip(full, reduced)):
            return
        n_parent_sites = len(self._emodel.get_parent_lattice())
        first_site = ((reduced[0] // 2 * reduced[1] + reduced[1] // 2) * reduced[2] + reduced[2] // 2) * n_parent_sites
        # Only clusters touching the central parent cell enter local updates.
        # Boundary clusters can wrap around the reduced box rather than the full one.
        for center in range(first_site, first_site + n_parent_sites):
            for cluster_id in self._emodel._site_clusters[center]:
                seen = set()
                for index in self._emodel._cluster_orbits_array[cluster_id].get_idxs():
                    cell, site = divmod(index, n_parent_sites)
                    x, yz = divmod(cell, reduced[1] * reduced[2])
                    y, z = divmod(yz, reduced[2])
                    wrapped = (x % full[0], y % full[1], z % full[2], site)
                    if wrapped in seen:
                        raise ValueError(
                            "A cluster wraps onto the same atom more than once in this supercell. "
                            "Increase its periodic repeat lengths to use reduced local MC updates."
                        )
                    seen.add(wrapped)

    def initialize_structure(self, initial_decoration: list[int] | None = None, nsubs: dict | None = None):
        if initial_decoration is not None:
            self.structure = Structure(self._scell, initial_decoration, track_indices=True)
            if self._nsubs is None or self._ensemble != "canonical":
                return
            conc = self.structure.get_fractional_concentrations()
            nsites = self.structure.get_nsites_per_type()
            check_dict = {}
            for key in conc.keys():
                ns = nsites[key]
                nl = []
                for i, cel in enumerate(conc[key]):
                    if i == 0:
                        continue
                    else:
                        nl.append(int(round(cel * ns, 0)))
                check_dict.update({key: nl})

            bol = dict_compare(check_dict, self._nsubs)
            if not bol:
                raise ValueError("Number of substitutents does not coincides with them from the inital decoration.")
        else:
            self.structure = self._scell.gen_random_structure(nsubs=self._nsubs, rng=self._rng_np, track_indices=True)

    def propose_step(self):
        """Propose a new structure by performing a random flip or swap."""
        raise NotImplementedError("Stepwise sampling is not yet implemented.")

    def accept_step(self, de: float, dmu: float, beta: float) -> bool:
        """Accept or reject the proposed step based on the energy difference,
        chemical potential difference, and beta."""
        if de + dmu <= 0:
            return True
        elif self.rng.uniform(0, 1) <= np.exp(-beta * (de + dmu)):
            return True
        else:
            return False

    def update_state(self):
        """Update the state of the system after accepting a step."""
        raise NotImplementedError("Stepwise sampling is not yet implemented.")

    def step(self) -> bool:
        """Perform one step of the Metropolis algorithm."""
        raise NotImplementedError("Stepwise sampling is not yet implemented.")

    def run(
        self,
        n_steps: int,
        temperature: float | list[float] = 1.0,
    ):
        """Run the Metropolis sampling."""
        if len(np.shape(temperature)) == 0:
            beta = np.ones((n_steps,), dtype=float) / (self.kB * temperature)
        else:
            beta = 1 / (self.kB * temperature)
        self.rave = RunningAverage(1000)  # keep track of acceptance rate
        progress = tqdm(
            range(1, n_steps + 1),
            total=n_steps,
            desc="MMC sim.",
        )
        for i in progress:
            self.beta = beta[i - 1]
            accept = self.step()
            self.rave.update_and_get_average(1.0 * accept)


class MonteCarloTrajectory:
    """MonteCarloTrajectory class

    **Description**:
        Objects of this class are used to store and access information of the trajectory generated from
        a Monte Carlo sampling performed with the MonteCarlo class.

        It is initialized with the SuperCell object.
        Alternative: If ``read`` is **True**, it is initalized from a Json file with name ``filename`` that
        was created from a MonteCarloTrajectory object before by ``MonteCarloTrajectory.serialize()``.

    **Parameters**:

    ``scell``: SuperCell object
        Super cell in which the sampling is performed.

    ``filename``: string (default: trajectory.json)
        The trajectoy can be stored in a json file with the path given by ``filename``.

    ``read``: boolean (default: False)
        If **True**, the trajectory is read from the Json file ``filename``.

    ``**kwargs``: keyword arguments

        ``models``: List of Model objects

        Further keyword arguments can be used to store additional information about the parameters used for
        the MonteCarloTrajectory.metropolis rountine. This will be saved in the Json file ``filename``
        under ``sampling_info``, if the object is serialized.

    """

    def __init__(self, scell, filename="trajectory.json", read=False, **kwargs):
        # Load Boltzmann constant in eV / K from ASE
        from ase.units import kB

        self._filename = filename
        self._scell = scell

        if read:
            self.read()
            self._models = kwargs.pop("models", [])

        else:
            self._trajectory = []

            self._scell = scell
            self._save_nsteps = kwargs.pop("save_nsteps", 10)
            self._write_no = 0

            self._models = kwargs.pop("models", [])

            self._nmc = kwargs.pop("no_of_sampling_steps", None)
            self._temperature = kwargs.pop("temperature", None)
            self._boltzmann_constant = kwargs.pop("boltzmann_constant", kB)
            self._scale_factor = kwargs.pop("scale_factor", None)
            self._target_rate = kwargs.pop("target_rate", None)
            self._keyword_arguments = kwargs
            self._reduce_structure = kwargs.pop("reduce_structure", False)

    def calculate_properties(self, models=[], prop_func=None, prop_name=None, **kwargs):
        """Calculate the property for all decorations in the trajectory. The property can be
           calculated by a Model object or an external function ``prop_func``.

        **Parameters**:
            ``models``: list (default: empty list)

            ``prop_func``: function (default: None)
                This function recieves as arguments the Structure object and the dictionary
                of trajectory entry at step i, and additional keyword arguments given by ``**kwargs``.

            ``prop_name``: string (default: None)
                Name of property which is calculated by ``prop_func``.

            ``**kwargs``: keyworded argument list, arbitrary length
                Additional parameters for the function ``prop_func``

        """
        for mo in models:
            if mo not in self._models:
                self._models.append(mo)

        predict_swap = kwargs.pop("predict_swap", False)

        structure = Structure(
            self._scell,
            decoration=self._trajectory[0]["decoration"],
            track_indices=True,
        )
        for m, mo in enumerate(models):
            mo.corrc.reset_mc(mc=True)

        for t, tr in enumerate(self._trajectory):
            indices_list = tr["swapped_positions"]
            for j in range(len(indices_list)):
                structure.swap(indices_list[j][0], indices_list[j][1])

            sdict = {}
            if predict_swap:
                if t == 0:
                    movalue = np.zeros(len(models))
                    for m, mo in enumerate(models):
                        movalue[m] = mo.predict(structure)
                        sdict.update({mo.property_name: movalue[m]})
                else:
                    for m, mo in enumerate(models):
                        # only works for single swaps
                        dmo = mo.predict_swap(
                            structure,
                            indices_list[0][0],
                            indices_list[0][1],
                        )
                        movalue[m] = movalue[m] + dmo
                        sdict.update({mo.property_name: movalue[m]})

            else:
                for m, mo in enumerate(models):
                    sdict.update({mo.property_name: mo.predict(structure)})

            if prop_func is not None:
                sdict.update({prop_name: prop_func(structure, tr, **kwargs)})

            self._trajectory[t]["key_value_pairs"].update(sdict)

    def add_step(
        self,
        step: int,
        energy: float,
        indices_list: list[int],
        new_sigmas: list[int],
        sigmas: list[int] | None = None,
        key_value_pairs: dict = {},
    ):
        """Add entry of the structure visited in the sampling to the trajectory."""
        step_dict = {
            "sampling_step_no": int(step),
            "energy": energy,
            "indices": indices_list,
            "new_sigmas": new_sigmas,
            "sigmas": sigmas,
            "key_value_pairs": key_value_pairs,
        }
        self._trajectory.append(step_dict)

    def get_sampling_step_entry_at_step(self, nstep):
        """Return the entry, e.g. the dictionary stored for the n-th samplig step (``nstep``) in trajectory."""
        return self.get_sampling_step_entry(self.get_nid_sampling_step(nstep))

    def get_sampling_step_entry(self, nid):
        """Return the entry, e.g. the dictionary stored at index ``nid`` in trajectory."""
        return self._trajectory[nid]

    def get_structure_at_step(self, nstep):
        """Return structure in form of a Structure object, at the n-th sampling step (``nstep``) in trajectory."""
        return self.get_structure(self.get_nid_sampling_step(nstep))

    def get_structure(self, nid):
        """Return structure in form of a Structure object, at index ``nid`` in trajectory."""
        traj = self._trajectory
        if nid < 0:
            _trajlength = len(traj)
            nid = int(_trajlength + nid)

        sigmas = deepcopy(traj[0]["sigmas"])
        for step_dict in traj:
            for i, new_sigma in zip(step_dict["indices"], step_dict["new_sigmas"]):
                sigmas[i] = new_sigma

        structure = Structure(self._scell, sigmas=sigmas)
        return structure

    def get_lowest_energy_structure(self):
        """Return Structure object with the lowest energy in trajectory."""
        return self.get_structure(np.argmin(self.get_energies()))

    def get_sampling_step_nos(self):
        """Return sampling step numbers of all entries in trajectory as array."""
        steps = []
        for tr in self._trajectory:
            steps.append(tr["sampling_step_no"])

        return np.int_(steps)

    def get_sampling_step_no(self, nid):
        """Return sampling step number of entry at index ``nid`` in trajectory."""
        return self._trajectory[nid]["sampling_step_no"]

    def get_nid_sampling_step(self, nstep):
        """Return entry index at the n-th sampling (``nstep``) step in trajectory."""
        steps = self.get_sampling_step_nos()
        nid = 0
        try:
            nid = np.where(steps == nstep)[0][0]
        except IndexError:
            if nstep > steps[-1]:
                nid = len(steps) - 1
            else:
                for i, s in enumerate(steps):
                    if s > nstep:
                        nid = i - 1
                        break
        return nid

    def get_energies(self):
        """Return energies of all entries in trajectory as array."""
        energies = []
        for tr in self._trajectory:
            energies.append(tr["energy"])

        return np.asarray(energies)

    def get_model_total_energies(self):
        return self.get_energies()

    def get_energy(self, nid):
        """Return energy of entry at index ``nid`` in trajectory."""
        return self._trajectory[nid]["energy"]

    def get_model_total_energy(self, nid):
        self.get_energy(nid)

    def get_properties(self, prop):
        """Return the property ``prop`` from all entries in the trajectory as array."""
        if prop == "energy":
            return self.get_energies()
        else:
            if prop not in [mo.property_name for mo in self._models]:
                raise KeyError(
                    f"Model for property '{prop}' is not defined. Found keys: "
                    f"{[mo.property_name for mo in self._models]}"
                )
            props = []
            for tr in self._trajectory:
                props.append(tr["key_value_pairs"][prop])
            return np.asarray(props)

    def get_property(self, nid, prop):
        """Return property of entry at index ``nid`` in trajectory."""
        if prop == "energy":
            return self.get_energy(nid)
        else:
            return self._trajectory[nid]["key_value_pairs"][prop]

    def calculate_average_property(
        self,
        prop_name="U",
        no_of_equilibration_steps=0,
        average_func=None,
        props_list=None,
        **kwargs,
    ):
        """Get averaged property of property with name ``prop_name`` after discarding at the start a given number of
           equilibration steps. The average can only be obtained from a property that was already calculated before,
           e.g. by MonteCarlo.Trajectory.calculate_properties(...).

           Alternatively, an external function ``average_func`` can be used to calculate the average of one property
           or several properties. The function ``average_func`` has as arguments an array containing the properties
           used for the average at each sampling index,
           i.e. [[prob1_1,prop1_2,...,prop1_N], [prop2_1,prop2_2,...,prop2_N],...],
           and optional keyword arguments. With ``prop_list``, the property names of prob1, prop_2, ... are defined.

        **Parameters:**

        ``prop_name``: string (default: ``U``)
            Name of property that is averaged.
            If **C_p**, the isobaric specific heat at zero pressure is calculated.
            If **U**, the internal energy is calculated.

        ``no_of_equilibration_steps``: integer
            Number of equilibration steps at the start of the Monte-Carlo sampling that are discarded from the average.

        ``average_func``: function (default: ``None``)
            If not **None**, the averaged property is obtained by this function. It gets as arguments an array and
            optional keyword arguments.

        ``**kwargs``: keyword arguments
           Additional arguments for ``average_func``.

        """
        step = 0
        prop_array = []
        prop_sum = 0

        if prop_name == "C_p" or prop_name == "U":
            prop_n = "energy"
        else:
            prop_n = prop_name

        factor = 1.0
        if self._scale_factor is not None:
            for s in self._scale_factor:
                factor *= float(s)

        ind = 0
        maxind = len(self._trajectory)

        while step < self._nmc + 1:
            if step < no_of_equilibration_steps:
                if ind < maxind:
                    if step == self._trajectory[ind]["sampling_step_no"]:
                        if average_func is not None:
                            pv = []
                            for func_prop in props_list:
                                pv.append(float(self.get_property(ind, func_prop)))
                            prop = pv
                        else:
                            prop = float(self.get_property(ind, prop_n))

                        ind += 1

            else:
                if ind < maxind:
                    if step == self._trajectory[ind]["sampling_step_no"]:
                        if average_func is not None:
                            pv = []
                            for func_prop in props_list:
                                pv.append(float(self.get_property(ind, func_prop)))
                            prop = pv
                        else:
                            prop = float(self.get_property(ind, prop_n))

                        ind += 1

                prop_array.append(prop)
                if average_func is None:
                    prop_sum += prop

            step += 1

        if average_func is not None:
            prop_array = np.asarray(prop_array)
            return average_func(prop_array.T, **kwargs)

        else:
            len_prop = len(prop_array)
            prop_average = np.divide(prop_sum, len_prop)

            if prop_name == "C_p":
                prop_array = np.divide(prop_array, factor)
                prop_average = np.divide(prop_average, factor)

                ediff = 0
                for p in prop_array:
                    ediff += np.subtract(p, prop_average) * np.subtract(p, prop_average)

                const = 1.0 * np.multiply((self._temperature) ** 2, (self._boltzmann_constant) ** 2)

                return np.divide(ediff, (const * len_prop * 1 / (1.0 * factor)))

            else:
                return prop_average

    def get_nids(self, prop, value):
        """Return array of integers, that are the indices of the entries in trajectory for which the property
        ``prop`` has the value ``value``.

        """
        arrayid = []

        if prop in ["sampling_step_no", "swapped_positions", "energy", "decoration"]:
            for i, tr in enumerate(self._trajectory):
                if tr[prop] == value:
                    arrayid.append(i)
        else:
            for i, tr in enumerate(self._trajectory):
                if tr["key_value_pairs"][prop] == value:
                    arrayid.append(i)

        return np.asarray(arrayid)

    def write_to_file(self, filename=None):
        self.serialize(filename=filename)

    def serialize(self, filename=None):
        """Write trajectory to Json file with name ``filename``. If ``filename`` is not defined, it uses
        trajectory file name defined in the initialization of MonteCarloTrajectory object.

        """
        if filename is not None:
            self._filename = filename

        trajdic = {}
        traj_info = {}
        traj_info.update({"number_of_sampling_steps": self._nmc})
        traj_info.update({"temperature": self._temperature})
        traj_info.update({"boltzmann_constant": self._boltzmann_constant})
        if self._scale_factor is not None:
            traj_info.update({"scale_factor": self._scale_factor})
        if self._target_rate is not None:
            traj_info.update({"target_rate": self._target_rate})

        for key in self._keyword_arguments.keys():
            traj_info.update({key: self._keyword_arguments[key]})

        trajdic.update({"sampling_info": traj_info})
        # add info about units of the energy, cluster expansion of the energy

        for j, dec in enumerate(self._trajectory):
            trajdic.update({str(j): dec})

        with open(self._filename, "w+", encoding="utf-8") as outfile:
            json.dump(trajdic, outfile, cls=NumpyEncoder, indent=2, separators=(",", ":"))

    def read(self, filename=None, append=False):
        """Read trajectory from the Json file with name ``filename``. If ``filename`` is not defined, it uses
        trajectory file name defined in the initialization of the MonteCarloTrajectory.

        """
        if filename is not None:
            trajfile = open(filename, "r")
            self._filename = filename
        else:
            trajfile = open(self._filename, "r")

        data = json.load(trajfile)

        if not append:
            self._trajectory = []

        traj_info = data.pop("sampling_info", None)

        if traj_info is not None:
            self._nmc = traj_info.pop("number_of_sampling_steps", None)
            self._temperature = traj_info.pop("temperature", None)
            self._boltzmann_constant = traj_info.pop("boltzmann_constant", None)
            self._scale_factor = traj_info.pop("scale_factor", None)
            self._target_rate = traj_info.pop("target_rate", None)
            self._keyword_arguments = traj_info

        data_keys = sorted([int(el) for el in set(data.keys())])

        if "model_total_energy" in data["0"]:
            exchange = True
        else:
            exchange = False

        for key in data_keys:
            tr = data[str(key)]
            if exchange:
                tr["energy"] = tr.pop("model_total_energy")
            self._trajectory.append(tr)


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types
    https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable/32850511

    """

    def default(self, obj):
        if isinstance(obj, list):
            return obj.tolist()
        elif isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, Cell):
            return obj.array

        return json.JSONEncoder.default(self, obj)
