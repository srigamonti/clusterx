# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

"""Maintained Monte Carlo sampling and trajectory analysis.

MonteCarlo can initialize from the last complete XYZ frame. resume_from alone
always reads the requested checkpoint. For a reusable annealing input, use
start_from="300K.xyz", atoms_filename="250K.xyz", atoms_mode="a": output
absence selects start_from; output existence selects resume_from, which defaults
to atoms_filename in this mode. An explicit resume_from may name another file;
the output's existence remains the switch. The selected file must be valid.
File-based initialization restores the configuration with new seeded random
streams, not the saved run's exact proposal sequence. See MonteCarlo and
reset_chain docstrings for examples and the checkpoint-validation contract.
"""

## packages needed for MonteCarloTrajectory
import json
from io import StringIO
from copy import deepcopy
import math
import os
import operator
import random
from numbers import Real
from pathlib import Path

from tqdm import tqdm
import numpy as np
from ase.cell import Cell
from ase.io import Trajectory, iread, read, write

from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.model import Model
from clusterx.utils import dict_compare, RunningAverage


def _require_linear_energy_model(model: Model):
    """Require an energy model affine in the original cluster correlations."""
    if model.estimator is not None and model._get_affine_parameters(len(model.corrc.get_cpool())) is None:
        raise ValueError(
            "Monte Carlo requires a linear CE energy model with explicit ECIs or a supported affine estimator."
        )


def _positive_scalar(value, name):
    """Validate numerical inputs without inferring their physical units."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be supplied explicitly as one finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be supplied explicitly as one finite positive number")
    return value


def _mc_count(value, name, minimum=0):
    """Require an integer control without accepting booleans or truncating floats."""
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class MonteCarlo:
    r"""Monte Carlo class

        Objects of this class are used to perform Monte Carlo samplings.

        It is initialized with:

        - a Model object, that enables to calculate the energy of a structure,

        - a SuperCell object, in which the sampling is performed,

        - specification of the thermodynamic ensemble:

            If ``ensemble`` is 'canonical', the composition for the sampling is defined with ``nsubs``. In case
            of multilattices, the sublattice for the sampling can be refined with ``sublattice_indices``.

            If ``ensemble`` is 'grandcanonical', the sublattice is defined with ``sublattices_indices``.

    Parameters
    ----------
    energy_model : Model
        Linear CE model used for acceptance and rejection. Its predictions retain the
        fitted target's normalization and units. The user must convert them
        to total supercell energy with metropolis's explicit energy_factor.

    scell : SuperCell
        Simulation cell in which the sampling is performed.

    ensemble : str, default canonical
        ``canonical`` allows for swaps of atoms that conserve the concentration defined with ``nsubs``.

        ``semigrand-canonical`` allows replacing atoms in sub-lattices defined with ``sublattice_indices``
        with other atoms in said sub-lattice.

    nsubs : dict, default None
        Defines the number of substituted atoms in each sub-lattice of the
        Supercell in which the sampling is performed.

        The format of the dictionary is as follows::

            {site_type1:[n11,n12,...], site_type2:[n21,n22,...], ...}

        It indicates how many substitutional atoms of every kind (n#1, n#2, ...)
        may replace the pristine species for sites of site_type#
        (see related documentation in SuperCell object).

    sublattice_indices : list of int, default None
        Defines the sublattices for the grand canonical sampling.
        Furthermore, it can be used to limit the canonical sampling
        to a reduced number of sublattices. E.g. in the case of nsubs = {0:[4,6], 1:[4]}. Here, sublattices 0 and 1
        contain substitutional sites, but only a sampling in sublattice 0 is wanted.
        Then, put ``sublattice_indices`` = [0].

    chemical_potentials : dict, default None
        Define additive chemical-potential costs for semigrand sampling, in the
        same energy unit as k_B*T. Supply one value per allowed species,
        including pristine species (usually reference cost zero). To use
        physical chemical potentials in E - sum(mu*N), supply their negatives.

        The format of the dictionary is as follows

        {site_type1:[:math:`\Delta \mu_{11}`, :math:`\Delta \mu_{12}`,...],
         site_type2:[:math:`\Delta \mu_{21}`, :math:`\Delta \mu_{22}`,...],
         ...}

        Here, :math:`\Delta \mu_{\#i}` is the chemical potential difference of substitutional species i relative
        to the pristine species in sub-lattice with site_type#.

    models : List of Model objects
        The properties returned from these Model objects are additionally
        calculated during the sampling and stored with their corresponding ``prop_name``
        to the dictionary ``key_value_pairs`` for each visited structure during the sampling.

        Properties from Model obejects can also be calculated after the
        sampling by using the MonteCarloTrajectory class.

    predict_swap : bool, default False
        If set to **True**, this parameter makes the sampling faster by calculating the correlation difference of the
        proposed structure with respect to the previous structure.

    reduce_structure : bool, default False
        If true, cluster orbits/correlations are computed and evaluated on
        subset of sigmas, in a box around central site in the structure.
        Incompatible with non-diagonal supercell shapes.

    energy_key : str, default energy
        For a multi-linear model, one of the predictions needs to be the energy prediction,
        energy_key needs to match one of the strings in energy_model.property_names, so that the energy
        prediction can be indexed from prediction vector.

    filename : str, default trajectory.json
        Path at which to store the MonteCarloTrajectory.

    seed : int or None, default None
        seed value for random number generators. For reproducible results,
        pass integer seed, otherwise seed=None

    resume_from : str or path-like, default None
        Start from the final complete frame of an XYZ/extxyz trajectory. The
        physical configuration is resumed with the random streams initialized
        from ``seed``; the original random stream is not restored. Without
        start_from this file is always used, regardless of output existence.

    start_from : str or path-like, default None
        Enable automatic first-run/continuation selection. At the first
        metropolis call, inspect atoms_filename BEFORE opening it: if absent,
        load the final complete frame of start_from; if present, load the final
        complete frame of resume_from. With start_from, omitted resume_from
        defaults to atoms_filename. Even with distinct resume/output paths,
        output existence is the switch. Invalid selected checkpoints raise an
        error, without falling back to the other file. Selection happens once
        after construction or reset_chain; continuing stages reuse memory state.
        Both checkpoint options conflict with an explicit initial_decoration.

        For reusable annealing input, use start_from="300K.xyz",
        atoms_filename="250K.xyz", atoms_mode="a" and an explicit temperature
        and proposal/sweep count. The first invocation reads 300K.xyz; later
        invocations read 250K.xyz. The separate 300 K source remains unchanged.
        Enable save_atoms_every or save_atoms_endpoints to create checkpoints.
        To choose a different continuation source, add resume_from="restart.xyz";
        existence of 250K.xyz still selects between that source and 300K.xyz.
        These options restore configurations, not exact original random streams.

    save_atoms_endpoints : bool, default False
        Write the initial configuration and each metropolis call's endpoint.
        Continuing calls share one stage boundary frame. Periodic checkpoints
        set by ``save_atoms_every`` follow the chain's proposal counter.

    atoms_mode : str, default "auto"
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
        start_from: str | os.PathLike | None = None,
    ):
        if (ensemble == "semigrand-canonical") and (chemical_potentials is None):
            raise ValueError("`chemical_potentials` have to be given when ensemble is 'semigrand-canonical")
        _require_linear_energy_model(energy_model)
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
        self._start_from = start_from
        self._resume_from = atoms_filename if start_from is not None and resume_from is None else resume_from
        self._resume_source = None  # Set to the checkpoint actually selected and loaded.
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
        *,
        energy_factor: float,
        temperature: float | list[float],
        boltzmann_constant: float,
        initial_decoration: list[int] | None = None,
        target_rate: float | None = None,
        n_clics: int = 1,
        serialize: bool = False,
        filename: str = "trajectory.json",
        max_steps_per_batch: int = 100_000,
        continue_chain: bool = False,
        metadata: dict | None = None,
        energy_observer=None,
        observation_interval: int = 1,
    ):
        r"""Perform Monte-Carlo Metropolis simulation

            Perfom Monte-Carlo Metropolis sampling for
            ``no_of_sampling_steps`` sampling steps.

            In the canonical ensemble, accept a proposal with probability
            :math:`\min(1,\exp[-s\Delta E_{model}/(k_B T)])`, where
            :math:`s` is the required scalar ``energy_factor``. The user must
            choose this multiplier so that :math:`s\Delta E_{model}` is the
            TOTAL energy change of the simulation supercell, in the SAME energy
            units as ``boltzmann_constant * temperature``. CELL cannot infer
            model units, model normalization, or the temperature unit.

            For a model in eV/atom and temperature in K, supply
            ``energy_factor=len(scell)`` and k_B in eV/K. For meV/atom with
            k_B in eV/K, supply ``energy_factor=len(scell)/1000``. A model
            already predicting total supercell energy requires an explicit 1
            if its energy unit agrees with k_B*T. No automatic conversion is
            performed. The former ``scale_factor`` list API has been removed.

            Semigrand sampling retains the existing additive chemical-potential
            cost convention: the acceptance change is s*delta_E + delta_mu.
            Chemical-potential costs must already use the total-energy unit of
            k_B*T; they are not multiplied by energy_factor. To express the
            conventional E - sum(mu*N), pass negative physical chemical potentials.

        Parameters
        ----------
        no_of_sampling_steps : int
            Number of sampling steps

        temperature : float or list of float
            Required positive temperature in units chosen by the user. If list,
            its length must equal ``no_of_sampling_steps``. Kelvin is not assumed.

        boltzmann_constant : float
            Required finite positive k_B in energy per temperature unit. Its
            product with temperature must have the unit of the scaled TOTAL
            supercell energy change. There is no default unit system.

        energy_factor : float
            Required finite positive scalar multiplying model energy differences.
            Includes BOTH the extensive normalization (per atom, per parent cell,
            etc. to the full simulation supercell) and any energy-unit conversion.
            No list, reciprocal convention, or inferred atom count is accepted.

        initial_decoration : list of int
            Atomic numbers of the initial structure, from which the sampling starts.
            If ``None``, sampling starts with a structure randomly generated.

        target_rate : float, default None
            Real number between 0 and 1. Represents the targeted acceptance rate.
            If not ``None``, the initial temperature will be adjusted to match the given
            acceptance rate. The acceptance ratio during the simulation is computed using
            the last 1000 moves.
            This changes the effective sampling temperature. Setting it to None
            restores the explicitly supplied temperature for fixed-temperature sampling.

        n_clics : int
            Number of species flips (grandcanonical) or species swaps (canonical) per sampling step.
            Defaults to 1.

        serialize : bool, default False
            Serialize the MonteCarloTrajectory object into a Json file after the sampling.

        filename : str, default trajectory.json
            Name of a Json file in which the trajectory is serialized after the sampling if ``serialize`` is **True**.

        max_steps_per_batch : int, default 100000
            Maximum number of proposal temperatures held in a temporary array at
            once. The Markov chain and random-number streams continue across these
            internal batches.

        continue_chain : bool, default False
            Continue the configuration and cached energy from
            the previous call. The default preserves the historical behavior of
            starting a new configuration for each call.

        metadata : dict or None
            Optional descriptive information saved with the trajectory. Sampling
            options must be explicit arguments; unknown keywords are errors.

        energy_observer : callable or None
            Optional callback ``energy_observer(step, total_energy)`` after each
            observation interval, including rejected moves. Energy is absolute
            model energy multiplied by energy_factor, also for reduced MC.
            No initial-state observation is made. The callback must not mutate
            the chain. This permits averages without accepted-move histories.

        observation_interval : positive int
            Proposal spacing for the observer, counted within this call.

        Returns
        -------
        MonteCarloTrajectory object
            Trajectory containing the complete information of the sampling trajectory.
        """
        _require_linear_energy_model(self._emodel)
        energy_factor = _positive_scalar(energy_factor, "energy_factor")
        boltzmann_constant = _positive_scalar(boltzmann_constant, "boltzmann_constant")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        if energy_observer is not None and not callable(energy_observer):
            raise ValueError("energy_observer must be callable")
        observation_interval = _mc_count(observation_interval, "observation_interval", minimum=1)
        self._sampling_parameters = {"energy_factor": energy_factor, "boltzmann_constant": boltzmann_constant}

        try:
            max_steps_per_batch = operator.index(max_steps_per_batch)
        except TypeError as exc:
            raise ValueError("max_steps_per_batch must be a positive integer") from exc
        if max_steps_per_batch < 1:
            raise ValueError("max_steps_per_batch must be a positive integer")
        temperature_array = None if np.ndim(temperature) == 0 else temperature
        if temperature_array is not None and len(temperature_array) != no_of_sampling_steps:
            raise ValueError("temperature must be a scalar or have one value per sampling step")
        if np.ndim(temperature) == 0:
            temperature = _positive_scalar(temperature, "temperature")

        continuing = continue_chain and self._chain_state is not None and initial_decoration is None
        resume_frame = None
        if not continuing:
            if initial_decoration is None and self._pending_initial_decoration is not None:
                initial_decoration = self._pending_initial_decoration
                self._pending_initial_decoration = None
            if initial_decoration is not None and (self._start_from is not None or self._resume_from is not None):
                raise ValueError("Specify either start_from/resume_from or initial_decoration, not both")
            checkpoint = self._select_initial_checkpoint()
            if checkpoint is not None:
                resume_frame = self._read_resume_frame(checkpoint)
                self._validate_resume_frame(resume_frame)
                initial_decoration = resume_frame.numbers
                self._resume_source = os.path.abspath(os.fspath(checkpoint))
                self._start_from = None
                self._resume_from = None
            self.initialize_structure(initial_decoration, self._nsubs)
        structure = self.structure
        sigmas = structure.get_sigmas()
        if continuing:
            current_energy = self._chain_state["current_energy"]
        elif self._reduce_structure:
            self._emodel.init_reduced_model(structure)
            self._emodel._init_interaction_dict(self._emodel.scell_reduced)
            structure.init_reduced_structure(self._emodel.scell_reduced)
            if self._predict_swap:
                self._check_reduced_local_geometry()
                current_energy = 0.0
            else:
                current_energy = self._energy_value(self._emodel.predict(structure, reduce_structure=True))
        else:
            current_energy = self._energy_value(self._emodel.predict(structure))
            self._emodel._init_interaction_dict(self._scell)

        e_last = np.float64(0.0) if self._reduce_structure else current_energy
        # Reduced local predictions need an absolute origin only for energy observations.
        if self._predict_swap and self._reduce_structure:
            absolute_start = self._chain_state.get("absolute_energy") if continuing else None
        else:
            absolute_start = current_energy
        if energy_observer is not None and absolute_start is None:
            absolute_start = self._energy_value(
                self._emodel.predict(structure, reduce_structure=self._reduce_structure)
            )
        energy_offset = absolute_start - e_last if absolute_start is not None else None

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
            energy_factor=energy_factor,
            target_rate=target_rate,
            metadata=metadata or {},
            ensemble=self._ensemble,
            reduce_structure=self._reduce_structure,
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
        beta_scale = self._chain_state["beta_scale"] if continuing and target_rate is not None else 1.0
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
                for i, new_sigma, old_sigma, site_type in flip_tuples:
                    de += self._emodel.predict_flip(
                        structure=structure,
                        index=i,
                        old_sigma=old_sigma,
                        new_sigma=new_sigma,
                        reduce_structure=self._reduce_structure,
                    )
                    structure.flip(i, old_sigma, new_sigma, site_type)
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

            acceptance_energy = energy_factor * de + dmu
            if acceptance_energy <= 0:
                accept_step = True
            elif self._rng_np.random() <= np.exp(-beta_chunk[beta_index - beta_chunk_start]
                                                 * beta_scale * acceptance_energy):
                accept_step = True
            else:
                accept_step = False

            if accept_step:
                e_last += de
                if not self._predict_swap:
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

            if energy_observer is not None and step % observation_interval == 0:
                energy_observer(step, float(energy_factor * (energy_offset + e_last)))

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
            "current_energy": current_energy,
            "running_average": rave,
            "beta_scale": beta_scale,
            "global_step": global_step_offset + no_of_sampling_steps,
            "stage": stage,
            "counter_origin": counter_origin,
            "absolute_energy": energy_offset + e_last if energy_offset is not None else None,
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
        same_source = resumed and mode == "a" and self._resume_source is not None and (
            path == self._resume_source or (os.path.isfile(path) and os.path.samefile(path, self._resume_source))
        )
        if same_source:
            # The source's final frame already contains this initial configuration.
            self._atoms_last_step = 0
            if getattr(self, "_resume_complete_bytes", None) is not None:
                with open(path, "r+b") as output:
                    output.truncate(self._resume_complete_bytes)
        return True

    def _write_atoms_checkpoint(self, step, stage, stage_step, temperature, origin, energy=None):
        """Snapshot sigmas without changing index maps or random streams."""
        if step == self._atoms_last_step:
            return
        atoms = self.structure.get_atoms()
        atoms.numbers = self.structure.get_decoration()
        nsites = self._scell.get_n_sub_sites(unique=False)
        atoms.info.update(mc_step=int(step), mc_stage=int(stage), mc_stage_step=int(stage_step),
                          mc_counter_origin=origin)
        atoms.info.update(self._sampling_parameters)
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

    def reset_chain(self, initial_decoration=None, seed=None, resume_from=None, start_from=None):
        """Discard the chain and caches, and optionally choose a new XYZ start.

        No checkpoint arguments means a fresh random configuration, or the
        supplied initial_decoration. resume_from alone always loads that file.
        With start_from, select start_from if atoms_filename is absent, otherwise
        resume_from (defaulting to atoms_filename). Selection occurs at the next
        metropolis call, once, before opening the output. Checkpoint options and
        initial_decoration are mutually exclusive. All resets initialize new
        random streams from seed, not the saved run's original random state.

        Examples
        --------
        mc.reset_chain(start_from="300K.xyz", seed=43), with
        mc configured to write 250K.xyz, loads 300K.xyz for a new output or
        resumes 250K.xyz if it exists. Pass resume_from="restart.xyz" to use
        that alternate source when 250K.xyz exists. mc.reset_chain(seed=43)
        explicitly clears file-based initialization and starts fresh instead.
        """
        if initial_decoration is not None and (start_from is not None or resume_from is not None):
            raise ValueError("Specify either start_from/resume_from or initial_decoration, not both")
        self._chain_state = None
        self._atoms_output_path = None
        self._atoms_last_step = None
        self._start_from = start_from
        self._resume_from = self._atoms_filename if start_from is not None and resume_from is None else resume_from
        self._resume_source = None
        self._pending_initial_decoration = (
            list(initial_decoration) if initial_decoration is not None else None
        )
        self._emodel.reset_mc(mc=True)
        self._rng_np = np.random.Generator(np.random.SFC64(seed))
        self._rng_std.seed(seed)

    def _select_initial_checkpoint(self):
        """Resolve the initial file once, using output existence only when start_from is set."""
        if self._start_from is None:
            return self._resume_from
        output_exists = os.path.exists(self._atoms_filename)
        checkpoint = self._resume_from if output_exists else self._start_from
        option = "resume_from" if output_exists else "start_from"
        state = "exists" if output_exists else "does not exist"
        print(f"Monte Carlo: output {self._atoms_filename!r} {state}; loading {option}={os.fspath(checkpoint)!r}")
        return checkpoint

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
        """Stepwise helper: de and dmu must already be TOTAL energy changes.

        beta is the inverse of k_B*T in the same energy unit. This helper
        performs no normalization or unit conversion; metropolis applies the
        required energy_factor before its own acceptance check.
        """
        if de + dmu <= 0:
            return True
        elif self._rng_np.random() <= np.exp(-beta * (de + dmu)):
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
        """The stepwise runner is unfinished; use metropolis with explicit physical inputs."""
        raise NotImplementedError("Use metropolis with explicit energy_factor, temperature, and boltzmann_constant")


class _EnergyMoments:
    """Online moments for independent reference samples or production energies."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value):
        if not math.isfinite(value):
            raise ValueError("The energy model returned a nonfinite energy")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def independent_stderr(self):
        return math.sqrt(max(0.0, self.m2) / (self.count * (self.count - 1))) if self.count > 1 else 0.0


class _BlockEnergyMoments(_EnergyMoments):
    """Retain block averages, not a per-proposal trajectory, for an approximate SEM."""

    def __init__(self, n_samples, n_blocks):
        super().__init__()
        base, remainder = divmod(n_samples, n_blocks)
        self.sizes = [base + (i < remainder) for i in range(n_blocks)]
        self.blocks = []
        self.current = _EnergyMoments()

    def add(self, value):
        super().add(value)
        self.current.add(value)
        if self.current.count == self.sizes[len(self.blocks)]:
            self.blocks.append(self.current.mean)
            self.current = _EnergyMoments()

    def stderr(self):
        # Weighted block means also handle sample counts not divisible by n_blocks.
        residual = sum(n * (value - self.mean)**2 for n, value in zip(self.sizes, self.blocks))
        return math.sqrt(residual / ((len(self.blocks) - 1) * self.count))


def _ti_values(values, name):
    """Normalize a scalar or one-dimensional numerical input without guessing units."""
    values = np.asarray(values)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1 or not values.size or values.dtype.kind not in "ifu":
        raise ValueError(f"{name} must be a number or a nonempty one-dimensional list of numbers")
    values = values.astype(float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def compute_free_energy_of_mixing(
    energy_model: Model,
    scell: SuperCell,
    compositions,
    temperatures,
    *,
    energy_factor: float,
    boltzmann_constant: float,
    n_mc_steps: int = 10_000,
    n_mc_eq: int = 1_000,
    sample_interval: int = 1,
    n_reference_samples: int = 1_000,
    n_blocks: int = 10,
    random_seed: int | None = None,
    predict_swap: bool = True,
    reduce_structure: bool = False,
    energy_key: str = "energy",
    max_steps_per_batch: int = 100_000,
    filename: str | os.PathLike | None = None,
    overwrite: bool = False,
):
    r"""Compute configurational mixing free energies by canonical thermodynamic integration.

    For N binary substitutional sites, NB=x*N, and beta=1/(k_B*T), evaluate

    .. math::

        \beta\Delta f_{mix}=-\log\binom{N}{N_B}/N+
        \int_0^\beta [\langle E_{total}\rangle/N-(1-x)e_A-xe_B]d\beta'.

    Only one binary substitutional site type is supported. Spectators remain
    pristine; N counts alloy sites, not all atoms. Species B is sigma=1 in the
    ordered allowed-species list. Compositions must give integer NB exactly;
    they are never silently rounded to a different composition. Pure limits
    are computed analytically without attempting an impossible unlike-atom swap.

    energy_factor must convert predictions to TOTAL supercell energy, in the
    same energy unit as boltzmann_constant*temperature, for every composition
    including both endmembers.
    Results are per binary site in that user-chosen energy unit. No unit system
    or atom-count normalization is inferred.

    The supplied linear CE is held fixed along the integration path, including the
    endmembers and beta=0 reference. It may represent configuration energies
    or effective configuration free energies at a chosen physical temperature.
    For example, keep a CE fitted at 300 K fixed while integrating the auxiliary
    sampling beta from zero to 1/(k_B*300 K). The endpoint gives the mixing free
    energy for that 300 K model; intermediate temperatures are sampling
    parameters and do not change the physical temperature represented by the CE.
    All entries in temperatures use the same supplied model. To evaluate a
    family of temperature-dependent CEs, repeat the complete integration for
    each physical target using its corresponding CE held fixed. This function
    neither updates the CE with sampling temperature nor includes the derivative
    terms needed for integration along a path with changing model parameters.
    E_total and mean_energy_per_site denote scaled model predictions; for an
    effective free-energy CE, they are not thermodynamic internal energies.

    Positive, distinct temperatures may be supplied in either order. Sampling
    and output are ordered from high to low T (increasing beta). Composite
    trapezoidal quadrature includes beta=0 explicitly. At beta=0, independent
    random decorations are drawn uniformly at fixed NB and their actual model
    predictions averaged. Fixed-composition correlations differ from independent-site
    random correlations in a finite cell. n_reference_samples controls the reference estimate.

    Each positive-temperature node discards n_mc_eq proposals, then measures
    n_mc_steps production proposals at sample_interval spacing, including
    rejected moves. Chains continue across temperatures. n_blocks block means
    estimate production standard errors; require at least two samples per block.
    References use the ordinary independent-sample SEM. Free-energy SEMs propagate
    node errors with trapezoidal weights, neglecting cross-temperature covariance.
    They exclude quadrature bias, poor equilibration, and finite-size errors.
    Check convergence in beta spacing, reference samples, burn-in, run length,
    block length, and supercell size. No target-rate adaptation is used.

    Returns a JSON-compatible dictionary with settings, endmember energies,
    beta=0 reference estimates, and one results row per composition/temperature.
    filename optionally writes the same dictionary as JSON; overwrite defaults
    to False. Added sampling storage depends on n_blocks and the grid size,
    not on the total number of proposals. No accepted-move history is serialized.
    """
    energy_factor = _positive_scalar(energy_factor, "energy_factor")
    boltzmann_constant = _positive_scalar(boltzmann_constant, "boltzmann_constant")
    n_mc_steps = _mc_count(n_mc_steps, "n_mc_steps", minimum=1)
    n_mc_eq = _mc_count(n_mc_eq, "n_mc_eq")
    sample_interval = _mc_count(sample_interval, "sample_interval", minimum=1)
    n_reference_samples = _mc_count(n_reference_samples, "n_reference_samples", minimum=2)
    n_blocks = _mc_count(n_blocks, "n_blocks", minimum=2)
    max_steps_per_batch = _mc_count(max_steps_per_batch, "max_steps_per_batch", minimum=1)
    if random_seed is not None:
        random_seed = _mc_count(random_seed, "random_seed")
    n_samples = n_mc_steps // sample_interval
    if n_samples < 2 * n_blocks:
        raise ValueError("n_mc_steps/sample_interval must yield at least two samples per block")
    xs = _ti_values(compositions, "compositions")
    ts = _ti_values(temperatures, "temperatures")
    if np.any((xs < 0) | (xs > 1)):
        raise ValueError("compositions must lie between 0 and 1")
    if np.any(ts <= 0) or len(np.unique(ts)) != len(ts):
        raise ValueError("temperatures must be positive and distinct")
    temperatures = np.sort(ts)[::-1]
    betas = 1.0 / (boltzmann_constant * temperatures)
    if not np.all(np.isfinite(betas)) or np.any(betas <= 0) or np.any(np.diff(betas) <= 0):
        raise ValueError("k_B*T must give finite positive inverse temperatures")
    if energy_model.get_parent_lattice() != scell.get_parent_lattice():
        raise ValueError("Model and simulation parent lattices differ")
    tags = scell.get_substitutional_tags()
    if len(tags) != 1 or len(scell.get_sublattice_types()[tags[0]]) != 2:
        raise ValueError("Free-energy integration requires exactly one binary substitutional site type")
    tag = int(tags[0])
    active = scell.get_substitutional_sites()
    n_sites = len(active)
    if not n_sites:
        raise ValueError("The simulation supercell must contain binary substitutional sites")
    counts = [int(round(float(x) * n_sites)) for x in xs]
    if any(abs(float(x) * n_sites - count) > 1e-8 for x, count in zip(xs, counts)):
        raise ValueError("Each composition must give an integer number of species-B sites in this supercell")
    if len(set(counts)) != len(counts):
        raise ValueError("compositions must specify distinct species-B counts")
    output = Path(filename) if filename is not None else None
    if not isinstance(overwrite, (bool, np.bool_)):
        raise ValueError("overwrite must be a boolean")
    if output is not None:
        if output.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output}; set overwrite=True explicitly")
        if not output.parent.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {output.parent}")
    _require_linear_energy_model(energy_model)

    # Keep input model artifacts read-only while warming geometry caches.
    cache_filepath = energy_model.corrc.pickle_file
    energy_model.corrc.pickle_file = None
    try:
        selector = MonteCarlo(energy_model, scell, nsubs={tag: [0]}, energy_key=energy_key)
        pristine = Structure(scell, sigmas=np.zeros(len(scell), dtype=int))
        if reduce_structure:
            energy_model.init_reduced_model(pristine)

        def energy_per_site(structure):
            if reduce_structure:
                structure.init_reduced_structure(energy_model.scell_reduced)
            prediction = selector._energy_value(energy_model.predict(structure, reduce_structure=reduce_structure))
            prediction = np.asarray(prediction)
            if prediction.size != 1 or not np.all(np.isfinite(prediction)):
                raise ValueError("The selected energy model must return one finite energy")
            energy = energy_factor * float(prediction.reshape(-1)[0]) / n_sites
            if not math.isfinite(energy):
                raise ValueError("energy_factor must give finite scaled model energies")
            return energy

        e_a = energy_per_site(pristine)
        sigmas_b = np.zeros(len(scell), dtype=int)
        sigmas_b[active] = 1
        e_b = energy_per_site(Structure(scell, sigmas=sigmas_b))
        seed_sequence = np.random.SeedSequence(random_seed)
        seeds = seed_sequence.spawn(2 * len(counts))
        result = {
            "schema_version": 1,
            "method": "canonical_monte_carlo_thermodynamic_integration",
            "settings": {
                "energy_factor": energy_factor, "boltzmann_constant": boltzmann_constant,
                "n_sites": n_sites, "n_atoms": len(scell), "site_type": tag,
                "species_a": int(scell.get_sublattice_types()[tag][0]),
                "species_b": int(scell.get_sublattice_types()[tag][1]),
                "input_temperatures": ts.tolist(), "temperature_order": "decreasing_temperature_increasing_beta",
                "n_mc_steps": n_mc_steps, "n_mc_eq": n_mc_eq, "sample_interval": sample_interval,
                "n_reference_samples": n_reference_samples, "n_blocks": n_blocks,
                "predict_swap": predict_swap, "reduce_structure": reduce_structure, "nonlinear": False,
                "max_steps_per_batch": max_steps_per_batch, "seed_entropy": seed_sequence.entropy,
                "quadrature": "composite_trapezoid_in_beta_including_zero",
                "uncertainty": (
                    "block_SEM; node_error_propagation; excludes_cross_temperature_covariance_and_quadrature_bias"
                ),
            },
            "endmembers": {"energy_a_per_site": e_a, "energy_b_per_site": e_b},
            "references": [], "results": [],
        }
        for index, count in enumerate(counts):
            x = count / n_sites
            baseline = (1 - x) * e_a + x * e_b
            pure = count in (0, n_sites)
            reference_seed = int(seeds[2 * index].generate_state(1, dtype=np.uint64)[0])
            mc_seed = int(seeds[2 * index + 1].generate_state(1, dtype=np.uint64)[0])
            reference = _EnergyMoments()
            mc = MonteCarlo(energy_model, scell, nsubs={tag: [count]}, predict_swap=predict_swap,
                            reduce_structure=reduce_structure, energy_key=energy_key, seed=mc_seed)
            if pure:
                reference.add(baseline)
            else:
                rng = np.random.default_rng(reference_seed)
                for _ in range(n_reference_samples):
                    reference.add(energy_per_site(scell.gen_random_structure(nsubs={tag: [count]}, rng=rng)))
            result["references"].append({
                "composition": x, "n_b": count, "beta": 0.0, "temperature": None,
                "mean_energy_per_site": reference.mean,
                "mean_energy_per_site_stderr": reference.independent_stderr(),
                "n_samples": reference.count,
                "method": "single_configuration" if pure else "uniform_random_fixed_composition",
                "reference_seed": reference_seed, "mc_seed": mc_seed,
            })
            log_omega = math.lgamma(n_sites + 1) - math.lgamma(count + 1) - math.lgamma(n_sites - count + 1)
            means = [reference.mean - baseline]
            errors = [reference.independent_stderr()]
            beta_grid = [0.0]
            continuing = False
            for temperature, beta in zip(temperatures, betas):
                if pure:
                    mean, error, sample_count = baseline, 0.0, 0
                else:
                    sampling = dict(energy_factor=energy_factor, boltzmann_constant=boltzmann_constant,
                                    temperature=float(temperature),
                                    max_steps_per_batch=max_steps_per_batch, serialize=False)
                    if n_mc_eq:
                        mc.metropolis(n_mc_eq, continue_chain=continuing, **sampling)
                        continuing = True
                    statistics = _BlockEnergyMoments(n_samples, n_blocks)

                    def observe(_step, total_energy):
                        statistics.add(total_energy / n_sites)

                    mc.metropolis(n_mc_steps, continue_chain=continuing, energy_observer=observe,
                                  observation_interval=sample_interval, **sampling)
                    continuing = True
                    mean, error, sample_count = statistics.mean, statistics.stderr(), statistics.count
                means.append(mean - baseline)
                errors.append(error)
                beta_grid.append(float(beta))
                widths = np.diff(beta_grid)
                weights = np.zeros(len(beta_grid))
                weights[:-1] += widths / 2
                weights[1:] += widths / 2
                integral = float(np.dot(weights, means))
                beta_free_energy = -log_omega / n_sites + integral
                free_error = float(np.linalg.norm(weights * errors) / beta)
                result["results"].append({
                    "composition": x, "n_b": count, "temperature": float(temperature), "beta": float(beta),
                    "mean_energy_per_site": mean, "mean_energy_per_site_stderr": error,
                    "mean_mixing_energy_per_site": mean - baseline,
                    "free_energy_of_mixing_per_site": beta_free_energy / float(beta),
                    "free_energy_of_mixing_per_site_stderr": free_error,
                    "beta_free_energy_of_mixing_per_site": beta_free_energy,
                    "n_samples": sample_count, "log_configuration_count": log_omega,
                })
        if output is not None:
            with output.open("w" if overwrite else "x", encoding="utf-8") as stream:
                json.dump(result, stream, cls=NumpyEncoder, indent=2, allow_nan=False)
                stream.write("\n")
        return result
    finally:
        energy_model.corrc.pickle_file = cache_filepath


class MonteCarloTrajectory:
    """MonteCarloTrajectory class

        Objects of this class are used to store and access information of the trajectory generated from
        a Monte Carlo sampling performed with the MonteCarlo class.

        It is initialized with the SuperCell object.
        Alternative: If ``read`` is **True**, it is initalized from a Json file with name ``filename`` that
        was created from a MonteCarloTrajectory object before by ``MonteCarloTrajectory.serialize()``.

    Parameters
    ----------
    scell : SuperCell
        Super cell in which the sampling is performed.

    filename : str, default trajectory.json
        The trajectoy can be stored in a json file with the path given by ``filename``.

    read : bool, default False
        If **True**, the trajectory is read from the Json file ``filename``.

    Notes
    -----
    New trajectories require an explicit scalar ``energy_factor`` and
    ``boltzmann_constant``. Reading new JSON obtains these inputs from
    ``sampling_info``. For older files missing energy_factor, the user must
    declare it explicitly. ``metadata`` holds optional descriptive information;
    unknown constructor keywords are errors.
    """

    def __init__(self, scell=None, filename="trajectory.json", read=False, *, energy_factor=None,
                 boltzmann_constant=None, temperature=None, no_of_sampling_steps=None, models=None,
                 target_rate=None, save_nsteps=10, reduce_structure=False, ensemble=None, metadata=None):
        self._filename = filename
        self._scell = scell
        self._energy_factor_override = (
            _positive_scalar(energy_factor, "energy_factor") if energy_factor is not None else None
        )

        if read:
            self.read()
            self._models = models or []

        else:
            self._trajectory = []

            self._scell = scell
            self._save_nsteps = save_nsteps
            self._write_no = 0

            self._models = models or []

            self._nmc = no_of_sampling_steps
            self._temperature = temperature
            self._boltzmann_constant = _positive_scalar(boltzmann_constant, "boltzmann_constant")
            self._energy_factor = _positive_scalar(energy_factor, "energy_factor")
            self._target_rate = target_rate
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError("metadata must be a dictionary")
            self._keyword_arguments = {"metadata": metadata or {}, "ensemble": ensemble,
                                       "reduce_structure": reduce_structure}
            self._reduce_structure = reduce_structure

    def calculate_properties(self, models=[], prop_func=None, prop_name=None, **kwargs):
        """Calculate the property for all decorations in the trajectory. The property can be
           calculated by a Model object or an external function ``prop_func``.

        Parameters
        ----------
        models : list, default empty list

        prop_func : function, default None
            This function recieves as arguments the Structure object and the dictionary
            of trajectory entry at step i, and additional keyword arguments given by ``**kwargs``.

        prop_name : str, default None
            Name of property which is calculated by ``prop_func``.

        **kwargs : keyworded argument list, arbitrary length
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
        """Return raw model energies, before energy_factor (relative for reduced MC)."""
        energies = []
        for tr in self._trajectory:
            energies.append(tr["energy"])

        return np.asarray(energies)

    def get_model_total_energies(self):
        """Return supercell energies in the user-chosen energy unit; retain any reduced-MC offset."""
        return self._energy_factor * self.get_energies()

    def get_energy(self, nid):
        """Return the raw model energy of entry nid, before energy_factor."""
        return self._trajectory[nid]["energy"]

    def get_model_total_energy(self, nid):
        """Return one supercell energy, retaining the reduced-MC relative zero when used."""
        return self._energy_factor * self.get_energy(nid)

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
        """Average a complete accepted-move history (sample with serialize=True).

           Get averaged property of property with name ``prop_name`` after discarding at the start a given number of
           equilibration steps. The average can only be obtained from a property that was already calculated before,
           e.g. by MonteCarlo.Trajectory.calculate_properties(...).

           Alternatively, an external function ``average_func`` can be used to calculate the average of one property
           or several properties. The function ``average_func`` has as arguments an array containing the properties
           used for the average at each sampling index,
           i.e. [[prob1_1,prop1_2,...,prop1_N], [prop2_1,prop2_2,...,prop2_N],...],
           and optional keyword arguments. With ``prop_list``, the property names of prob1, prop_2, ... are defined.

        Parameters
        ----------
        prop_name : str, default U
            Name of property that is averaged.
            If **U**, return mean TOTAL supercell energy after applying energy_factor.
            Reduced MC retains its relative initial-state energy zero.
            **C_p** is the historical name for the TOTAL configurational heat
            capacity Var(E_total)/(k_B*T**2), in energy per temperature unit.
            It requires a canonical, fixed-temperature trajectory without
            target-rate adaptation. It does not include pressure/volume terms
            and is not normalized per atom or reported in units of k_B.

        no_of_equilibration_steps : int
            Number of equilibration steps at the start of the Monte-Carlo sampling that are discarded from the average.

        average_func : function, default None
            If not **None**, the averaged property is obtained by this function. It gets as arguments an array and
            optional keyword arguments.

        **kwargs : keyword arguments
            Additional arguments for ``average_func``.
        """
        step = 0
        prop_array = []
        prop_sum = 0

        if prop_name == "C_p" or prop_name == "U":
            prop_n = "energy"
        else:
            prop_n = prop_name

        if prop_name == "C_p":
            if (np.ndim(self._temperature) != 0 or self._target_rate is not None
                    or self._keyword_arguments.get("ensemble") != "canonical"):
                raise ValueError("C_p requires canonical sampling at fixed temperature without target_rate adaptation")
            _positive_scalar(self._temperature, "temperature")

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
                total_energies = self._energy_factor * np.asarray(prop_array)
                return np.var(total_energies) / (self._boltzmann_constant * self._temperature**2)
            elif prop_name == "U":
                return self._energy_factor * prop_average
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
        traj_info.update({"energy_factor": self._energy_factor})
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
        if traj_info is None:
            raise ValueError(
                "Trajectory lacks sampling_info; declare its physical inputs before thermodynamic analysis"
            )

        if traj_info is not None:
            self._nmc = traj_info.pop("number_of_sampling_steps", None)
            self._temperature = traj_info.pop("temperature", None)
            self._boltzmann_constant = _positive_scalar(traj_info.pop("boltzmann_constant", None), "boltzmann_constant")
            factor = traj_info.pop("energy_factor", self._energy_factor_override)
            if factor is None:
                raise ValueError(
                    "Trajectory has no energy_factor; supply it explicitly when loading older trajectories"
                )
            self._energy_factor = _positive_scalar(factor, "energy_factor")
            if self._energy_factor_override is not None and self._energy_factor_override != self._energy_factor:
                raise ValueError("Supplied energy_factor differs from the trajectory's stored factor")
            self._target_rate = traj_info.pop("target_rate", None)
            self._keyword_arguments = traj_info
            self._reduce_structure = traj_info.get("reduce_structure", False)

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
