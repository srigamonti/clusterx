# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
import json
import random
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import plac

from clusterx.cli.config_utils import cmd_message, get_command_name
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
from clusterx.thermodynamics.monte_carlo import MonteCarlo, MonteCarloTrajectory, _positive_scalar
from clusterx.thermodynamics.monte_carlo_lite import (
    MCRun,
    MonteCarloLite,
    specific_heat,
)
from clusterx.utils import _process_deprecated, _timed
from clusterx.visualization import plot_property

commands = ["metropolis"]


def _boolean(value):
    """Parse explicit boolean option values, including False on the command line."""
    if str(value).lower() not in {"true", "false", "1", "0"}:
        raise ValueError("Use true or false for boolean options")
    return str(value).lower() in {"true", "1"}


def _sampling_schedule(runs, temperature, n_mc_steps, n_mc_sweeps, nsites):
    """Validate an ordered schedule and convert sweep counts to proposal counts."""
    if runs is not None and (n_mc_steps is not None or n_mc_sweeps is not None):
        raise ValueError("With runs, specify lengths inside each stage, not at the top level")
    if runs is None:
        stage = {"temperature": temperature}
        if n_mc_sweeps is not None:
            stage["n_mc_sweeps"] = n_mc_sweeps
        if n_mc_steps is not None:
            stage["n_mc_steps"] = n_mc_steps
        if n_mc_steps is None and n_mc_sweeps is None:
            stage["n_mc_steps"] = 100  # Historical single-run default.
        runs = [stage]
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must be a nonempty ordered list of temperature stages")
    stages = []
    for index, run in enumerate(runs, 1):
        if not isinstance(run, dict):
            raise ValueError(f"Stage {index} must be a table")
        unknown = set(run) - {"temperature", "n_mc_steps", "n_mc_sweeps", "traj_filepath", "ignore"}
        if unknown:
            raise ValueError(f"Unknown stage {index} options: {', '.join(sorted(unknown))}")
        if not isinstance(run.get("ignore", False), bool):
            raise ValueError(f"Stage {index}: ignore must be true or false")
        if run.get("ignore", False):
            continue
        stage_temperature = run.get("temperature")
        if (not isinstance(stage_temperature, (int, float)) or isinstance(stage_temperature, bool)
                or not np.isfinite(stage_temperature) or stage_temperature <= 0):
            raise ValueError(f"Stage {index} needs a finite positive temperature")
        if ("n_mc_steps" in run) == ("n_mc_sweeps" in run):
            raise ValueError(f"Stage {index}: specify exactly one of n_mc_steps or n_mc_sweeps")
        if "n_mc_steps" in run:
            steps = run["n_mc_steps"]
        else:
            if not nsites:
                raise ValueError("Sweeps require at least one substitutional site")
            sweeps = run["n_mc_sweeps"]
            if not isinstance(sweeps, (int, float)) or isinstance(sweeps, bool):
                raise ValueError(f"Stage {index} needs a numeric sweep count")
            steps = sweeps * nsites
        if not isinstance(steps, (int, float)) or isinstance(steps, bool):
            raise ValueError(f"Stage {index} needs an integer proposal count")
        if not np.isfinite(steps) or steps <= 0 or not np.isclose(steps, round(steps), rtol=0, atol=1e-8):
            raise ValueError(f"Stage {index} length must give a positive integer number of proposals")
        stages.append({**run, "n_mc_steps": int(round(steps))})
    if not stages:
        raise ValueError("The schedule contains no active stages")
    return stages


@plac.annotations(
    task=("Task to perform.", "option", "task", str),
    plat_filepath=(
        "Filepath of a serialized ParentLattice object.",
        "option",
        "plfp",
        str,
    ),
    model_filepath=("Output Model file path.", "option", "mofp", str),
    mcsetup_filepath=("MC setup file path.", "option", "sufp", str),
    mcrun_filepath=("MC run file path.", "option", "mcrfp", str),
    mcrun_filepaths=("MC run file path.", "option", "mcrfps", list),
    mcscell_filepath=("MC scell file path.", "option", "scfp", str),
    traj_filepath=(
        "Filename for the output trajectory",
        "option",
        "trfp",
        str,
    ),
    sc_shape=("Supercell shape as a JSON integer, vector, or matrix.", "option", "scsh", json.loads),
    n_substitutions=("Substitution counts as a JSON integer or dictionary", "option", "nsubs", json.loads),
    ensemble=("Thermodynamic ensemble", "option", "ens", str),
    sublattice_indices=("Sublattice indices as a JSON list", "option", "sublat", json.loads),
    chemical_potentials=(
        "Chemical potentials for each sublattice",
        "option",
        "chem_pot",
        json.loads,
    ),
    models_aux_filepaths=(
        "Filepaths of auxiliary models to be used in sampling",
        "option",
        "aux_models",
        json.loads,
    ),
    n_clics=(
        "Number of clics (swaps in canonical, flips in grand-canoncial)",
        "option",
        "n_clics",
        int,
    ),
    predict_swap=(
        "Use predict_swap during energy prediction",
        "option",
        "pred_swap",
        _boolean,
    ),
    reduce_structure=(
        "Use reduced structure, when supercell shape is diagonal",
        "option",
        "reduce",
        _boolean,
    ),
    n_mc_steps=(
        "Number of metropolis steps to perform",
        "option",
        "n_steps",
        int,
    ),
    n_mc_sweeps=("Sweeps; one sweep is one proposal per substitutional site", "option", "sweeps", float),
    runs=("Ordered temperature/length stages as a JSON list", "option", "runs", json.loads),
    resume_from=("Resume the final complete XYZ frame with a new seeded random stream", "option", "resume", str),
    start_from=("Initial XYZ source if output is absent; otherwise use resume_from (default: output)",
                "option", "start", str),
    save_atoms_every=("Checkpoint interval in proposals; zero disables periodic output", "option", "save_every", int),
    atoms_filename=("XYZ/extxyz or ASE checkpoint output", "option", "atoms", str),
    save_atoms_endpoints=("Save initial configuration and each temperature endpoint", "option", "endpoints", _boolean),
    atoms_mode=("Checkpoint mode: x=new file, w=overwrite, a=append, auto=legacy", "option", "atoms_mode", str),
    max_steps_per_batch=("Maximum temporary temperature array length", "option", "batch", int),
    nonlinear=("Cache global correlations for nonlinear models", "option", "nonlinear", _boolean),
    continue_chain=("Continue in memory; schedule stages always continue", "option", "continue", _boolean),
    serialize=("Store accepted-move histories as JSON; use false for large runs", "option", "serialize", _boolean),
    energy_key=("Energy property name for multi-output models", "option", "energy_key", str),
    energy_factor=(
        "Required scalar multiplier: energy_factor * model change must be TOTAL supercell energy in k_B*T units",
        "option",
        "ef",
        float,
    ),
    temperature=(
        "Required temperature in user-chosen units consistent with k_B and the scaled supercell energy",
        "option",
        "temp",
        float,
    ),
    boltzmann_constant=("Required k_B in scaled energy per temperature unit; no default units", "option", "k_B", float),
    metadata=("Optional trajectory descriptions as a JSON dictionary", "option", "metadata", json.loads),
    random_seed=(
        "Random seed to produce reproducible results",
        "option",
        "rs",
        int,
    ),
    initial_decoration=(
        "Initial decoration of the supercell, list of integers",
        "option",
        "init_dec",
        json.loads,
    ),
    target_rate=(
        """Real number between 0 and 1. Represents the rate of accepted moves.
        If not ``None``, the initial temperature will be adjusted to match the given
        acceptance ratio. The acceptance ratio during the simulation is computed using
        the last 1000 moves.""",
        "option",
        "acc_ratio",
        float,
    ),
    filename=(
        "Deprecated: use ``traf_filepath instead``",
        "option",
        "fn",
        str,
    ),
)
def metropolis(
    # class init arguments
    task: str = "usage",
    plat_filepath: str = "plat.json",
    model_filepath: str = "model.pickle",
    mcsetup_filepath: str = "mc-setup.pickle",
    mcrun_filepath: str = None,
    mcrun_filepaths: list[str] = None,
    plotdata_filepath: str = None,
    show_plot=True,
    save_plot=False,
    keep_sigmas: Optional[int] = 1,
    mcscell_filepath: str = "mc-scell.pickle",
    traj_filepath: str = "mc-trajectory.json",
    sc_shape: Optional[Union[int, List[int], List[List[int]]]] = 1,
    n_substitutions: Optional[Union[int, dict]] = None,
    ensemble: str = "canonical",
    sublattice_indices: List[int] = [],
    chemical_potential: float = 0.0,
    chemical_potentials: Optional[dict] = None,
    models_aux_filepaths: List[str] = [],
    predict_swap: bool = True,
    reduce_structure: bool = False,
    # metropolis sampling arguments
    n_mc_steps: int | None = None,
    n_mc_eq: int = 1,
    n_clics: int = 1,
    runs: Optional[list[dict]] = None,
    energy_factor: float | None = None,
    temperature: float | None = None,
    temperatures: Optional[List[float]] = None,
    boltzmann_constant: float | None = None,
    initial_decoration: Optional[List[int]] = None,
    target_rate: Optional[float] = None,
    random_seed: Optional[int] = None,
    # deprecated
    filename: Optional[str] = None,  # use traj_filepath instead
    # Maintained MC constructor and sampling options.
    resume_from: str | None = None,
    save_atoms_every: int = 0,
    atoms_filename: str = "metropolis.traj",
    save_atoms_endpoints: bool = False,
    atoms_mode: str = "auto",
    energy_key: str = "energy",
    n_mc_sweeps: float | None = None,
    max_steps_per_batch: int = 100_000,
    nonlinear: bool = False,
    continue_chain: bool = False,
    serialize: bool = True,
    metadata: dict | None = None,
    start_from: str | None = None,
):
    r"""Sample with an explicit TOTAL-supercell energy multiplier, k_B, and temperature.

    The user is responsible for model normalization and units. For model energy
    eV/atom and T in K, use energy_factor=N_atoms and k_B in eV/K. For meV/atom
    and k_B in eV/K, use N_atoms/1000. A total-energy model still requires an
    explicit factor (1 when its energy unit matches k_B*T). CELL does not infer
    atom counts, unit conversions, or a default temperature/Boltzmann constant.
    Specify temperature directly or explicitly in every runs stage.

    Notes
    -----
    .. rubric:: Checkpoint initialization for task="runmc"

    resume_from alone always loads that file's final complete frame, even when
    atoms_filename does not exist. It never infers a temperature or run length.
    With start_from, the existence of atoms_filename selects the source, BEFORE
    opening or writing output:

    - Output absent: load the last complete frame of start_from.
    - Output present: load the last complete frame of resume_from.

    When start_from is supplied, omitted resume_from defaults to atoms_filename.
    If resume_from names a different file, existence of atoms_filename still
    determines which source is selected. Missing, empty or incompatible selected
    checkpoints cause an error; they never silently fall back to the other file.
    The command prints the selected source and the output-existence decision.
    initial_decoration cannot be combined with either checkpoint option.

    Loading restores cell/site/species/composition-compatible configurations
    and rebuilds model correlations and energy before sampling, including for
    nonlinear reduced models. random_seed initializes a new random stream;
    checkpoint files do not restore the exact original proposal sequence.
    Selection happens once for the new MC object. Later schedule stages reuse
    the chain, random streams and caches without rechecking files.

    .. rubric:: Output and reuse

    Use atoms_mode="a" to create an absent output or append to an existing one.
    With the shorthand resume/output path, appending avoids duplicating the
    last loaded frame and removes an interrupted final XYZ tail before writing.
    A distinct start_from source is only read. atoms_mode="w" explicitly
    replaces output AFTER loading the selected configuration; "x" refuses an
    existing output. Output existence is tested before either action.
    Enable save_atoms_every > 0 and/or save_atoms_endpoints=true to write frames;
    without checkpoints, output cannot mark completion of a previous invocation.
    These options specify another explicit number of proposals/sweeps each
    invocation, not a cumulative target or an inferred remaining run length.

    .. rubric:: Reusable named-step TOML example

    Save as anneal-250K.toml and run ``cell --input_file anneal-250K.toml``.
    This example assumes a one-atom binary parent cell, a 64-site supercell at
    half filling, and a model predicting eV/atom; adjust paths, geometry, counts
    and energy_factor for your model. The same input can be invoked repeatedly::

        do = ["anneal_250K"]

        [anneal_250K]
        command = "metropolis"
        task = "runmc"
        model_filepath = "model.pickle"
        plat_filepath = "plat.json"
        sc_shape = [4, 4, 4]
        n_substitutions = {"0" = [32]}
        ensemble = "canonical"
        temperature = 250.0
        n_mc_sweeps = 10
        energy_factor = 64.0
        boltzmann_constant = 8.617330337217213e-5
        predict_swap = true
        reduce_structure = true
        nonlinear = true
        start_from = "300K.xyz"
        # resume_from = "250K.xyz"  # Defaults to atoms_filename with start_from.
        atoms_filename = "250K.xyz"
        atoms_mode = "a"
        save_atoms_endpoints = true
        save_atoms_every = 64
        serialize = false
        random_seed = 43
        max_steps_per_batch = 100000

    First invocation reads 300K.xyz and creates 250K.xyz. Later invocations
    continue from 250K.xyz. To use another continuation source, set
    resume_from="restart.xyz"; existence of 250K.xyz still selects that branch.

    Equivalent direct CLI invocation (booleans take explicit true/false)::

        cell metropolis -task runmc -mofp model.pickle -plfp plat.json \
            -scsh '[4,4,4]' -nsubs '{"0":[32]}' -temp 250 -sweeps 10 \
            -ef 64 -k_B 8.617330337217213e-5 -pred_swap true -reduce true \
            -nonlinear true -start 300K.xyz -atoms 250K.xyz -atoms_mode a \
            -endpoints true -save_every 64 -serialize false -rs 43

    The Python API provides MonteCarlo(..., start_from=...) and
    mc.reset_chain(start_from=..., seed=...) with the same selection rule.
    mc.reset_chain(seed=...) clears file-based initialization for a fresh run.
    """
    cmd_message("head")

    traj_filepath = _process_deprecated(
        traj_filepath, filename, "traj_filepath", "filename"
    )
    if task not in {"runmc", "runMC", "run-monte-carlo"} and n_mc_steps is None:
        n_mc_steps = 100

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    match task:
        case "runmc" | "runMC" | "run-monte-carlo":
            print(f"Info({get_command_name()}): Initialization")

            energy_factor = _positive_scalar(energy_factor, "energy_factor")
            boltzmann_constant = _positive_scalar(boltzmann_constant, "boltzmann_constant")
            if runs is None and temperature is None:
                raise ValueError("temperature must be supplied explicitly, directly or in an explicit runs schedule")

            if isinstance(n_substitutions, dict):
                n_substitutions = {int(k): v for k, v in n_substitutions.items()}
            elif isinstance(n_substitutions, int):
                n_substitutions = {0: [n_substitutions]}

            plat = ParentLattice(filepath=plat_filepath)
            energy_model = Model(filepath=model_filepath)
            scell = SuperCell(plat, sc_shape)
            if energy_model.get_parent_lattice() != plat:
                raise ValueError("Model and simulation parent lattices differ")
            models_aux = [Model(filepath=fp) for fp in models_aux_filepaths]
            if temperatures is not None:
                raise ValueError("For maintained MC schedules, use runs with an explicit length for each temperature")
            if (start_from is not None or resume_from is not None) and runs is None and (
                temperature is None or (n_mc_steps is None and n_mc_sweeps is None)
            ):
                raise ValueError("Checkpoint initialization requires an explicit runs schedule, "
                                 "or temperature and a step/sweep count")
            stages = _sampling_schedule(
                runs, temperature,
                n_mc_steps, n_mc_sweeps, scell.get_n_sub_sites(unique=False),
            )

            mc = MonteCarlo(
                energy_model=energy_model,
                scell=scell,
                nsubs=n_substitutions,
                ensemble=ensemble,
                sublattice_indices=sublattice_indices or None,
                chemical_potentials=(
                    {int(key): value for key, value in chemical_potentials.items()}
                    if chemical_potentials is not None else None
                ),
                models=models_aux,
                predict_swap=predict_swap,
                reduce_structure=reduce_structure,
                filename=traj_filepath,
                seed=random_seed,
                resume_from=resume_from,
                start_from=start_from,
                save_atoms_every=save_atoms_every,
                atoms_filename=atoms_filename,
                save_atoms_endpoints=save_atoms_endpoints,
                atoms_mode=atoms_mode,
                energy_key=energy_key,
            )
            for index, stage in enumerate(stages):
                output_path = stage.get("traj_filepath", traj_filepath)
                if serialize and len(stages) > 1 and "traj_filepath" not in stage:
                    path = Path(traj_filepath)
                    output_path = str(path.with_name(f"{path.stem}_stage{index + 1:04d}{path.suffix}"))
                trajectory = mc.metropolis(
                    no_of_sampling_steps=stage["n_mc_steps"],
                    energy_factor=energy_factor,
                    temperature=stage["temperature"],
                    boltzmann_constant=boltzmann_constant,
                    initial_decoration=initial_decoration if index == 0 else None,
                    target_rate=target_rate,
                    n_clics=n_clics,
                    serialize=serialize,
                    filename=output_path,
                    max_steps_per_batch=max_steps_per_batch,
                    nonlinear=nonlinear,
                    continue_chain=continue_chain if index == 0 else True,
                    metadata=metadata,
                )
            return trajectory

        case (
            "setup" | "setupmclite" | "setupMClite" | "setup-monte-carlo-lite"
            | "run" | "run-metropolis" | "runmclite" | "runMClite"
            | "run-monte-carlo-lite" | "run-simulated-annealing"
        ):
            raise ValueError(
                "MonteCarloLite sampling tasks are discontinued. Use task='runmc' with "
                "energy_factor, boltzmann_constant, and an explicit temperature or runs schedule."
            )

        case "compute_specific_heat" | "compute-specific-heat":
            mc_setup = MonteCarloLite.from_file(mcsetup_filepath)
            specific_heats = []
            temperatures = []
            for fp in mcrun_filepaths:
                mc_run = MCRun.from_file(fp)
                temperatures.append(mc_run.temperature)
                specific_heat_dict = specific_heat(mc_setup, mc_run, n_eq=n_mc_eq)

                specific_heats.append(specific_heat_dict["C"])

            plot_data = plot_property(
                temperatures,
                specific_heats,
                prop_name="Specific heat",
                xaxis_label="Temperature",
                yaxis_label="Specific heat",
                show_plot=show_plot,
                save_plot=save_plot,
            )

            if plotdata_filepath is not None:
                with open(plotdata_filepath, "w+", encoding="utf-8") as outfile:
                    json.dump(plot_data, outfile, indent=2, separators=(",", ":"))

        case "plot-mc-run":
            # for e, s in zip(mcrun.energies, mcrun.accepted_steps):
            #     energies_accepted.append(e)
            #     steps_accepted.append(s + (itemp + 1) * n_mc_steps)

            if mcrun_filepath is not None and mcrun_filepaths is not None:
                raise ValueError(
                    "Provide either `mcrun_filepath` or `mcrun_filepaths`, not both."
                )

            if mcrun_filepaths is None:
                if mcrun_filepath is not None:
                    mcrun_filepaths = [mcrun_filepath]
                else:
                    mcrun_filepaths = []

            for mcrun_filepath in mcrun_filepaths:
                mcrun = MCRun.from_file(filepath=mcrun_filepath)
                energies_accepted = mcrun.energies
                steps_accepted = mcrun.accepted_steps
                temperature = mcrun.temperature
                with _timed("plot_property"):
                    plot_property(
                        steps_accepted,
                        energies_accepted,
                        prop_name=f"MC Run, T={temperature}",
                        xaxis_label="Step number",
                        yaxis_label="Energy",
                    )

        case "plot-mc-runs":
            """Plot multiple runs in one frame."""
            scale = 1.0
            width = 15.0 * scale
            golden_ratio = (np.sqrt(5) - 0.9) / 2.0
            height = float(width * golden_ratio)
            fig, ax = plt.subplots(figsize=(width, height))
            for i, mcrun_filepath in enumerate(mcrun_filepaths):
                mcrun = MCRun.from_file(filepath=mcrun_filepath)
                energies_accepted = mcrun.energies
                steps_accepted = mcrun.accepted_steps
                temperature = mcrun.temperature
                ax.plot(
                    steps_accepted,
                    energies_accepted,
                    marker=".",
                    markersize=15 * scale,
                    markeredgewidth=2.0 * scale,
                    linewidth=2.2 * scale,
                    label=f"T={temperature}",
                )
            plt.legend()
            plt.show()
            fig.savefig("mc-runs.png")

        case "info-mc-run":
            if mcrun_filepath is not None and mcrun_filepaths is not None:
                raise ValueError(
                    "Provide either `mcrun_filepath` or `mcrun_filepaths`, not both."
                )

            if mcrun_filepaths is None:
                if mcrun_filepath is not None:
                    mcrun_filepaths = [mcrun_filepath]
                else:
                    mcrun_filepaths = []

            for mcrun_filepath in mcrun_filepaths:
                mcrun = MCRun.from_file(filepath=mcrun_filepath)
                energies_accepted = mcrun.energies
                steps_accepted = mcrun.accepted_steps
                temperature = mcrun.temperature
                print("---------------------------------------------------------------")
                print(mcrun_filepath)
                print(temperature)
                print(steps_accepted[-1])
                print(f"{steps_accepted[-1]:e}")

        case "collect-structures" | "create-structures-set-from-mcrun":
            mc_setup = MonteCarloLite.from_file(mcsetup_filepath)
            scell = mc_setup._scell
            plat = scell.get_parent_lattice()
            sset = StructuresSet(parent_lattice=plat)

            if mcrun_filepath is not None and mcrun_filepaths is not None:
                raise ValueError(
                    "Provide either `mcrun_filepath` or `mcrun_filepaths`, not both."
                )

            if mcrun_filepaths is None:
                if mcrun_filepath is not None:
                    mcrun_filepaths = [mcrun_filepath]
                else:
                    mcrun_filepaths = []

            for mcrun_filepath in mcrun_filepaths:
                mcrun = MCRun.from_file(filepath=mcrun_filepath)
                sigmas = mcrun.last_sigma()
                sset.add_structure(Structure(super_cell=scell, sigmas=sigmas))

            sset.serialize(filepath="sset.db", overwrite=True)

        case "compute-property":
            property_model = Model(filepath=model_filepath)
            mc_setup = MonteCarloLite.from_file(mcsetup_filepath)

            scell = mc_setup._scell
            if not scell.is_nary(2):
                raise RuntimeError("Invalid supercell: expected binary (n=2).")

            substitutional_sublattice = scell.get_substitutional_tags()[0]

            if mcrun_filepath is not None and mcrun_filepaths is not None:
                raise ValueError(
                    "Provide either `mcrun_filepath` or `mcrun_filepaths`, not both."
                )

            if mcrun_filepaths is None:
                if mcrun_filepath is not None:
                    mcrun_filepaths = [mcrun_filepath]
                else:
                    mcrun_filepaths = []

            data = {}
            data["property_name"] = property_model.property_name
            data["number_of_substitutional_sites"] = scell.get_n_sub_sites(unique=False)
            data["property_values"] = []
            data["temperatures"] = []
            data["number_of_substituents"] = []
            data["fractional_concentrations"] = []

            for mcrun_filepath in mcrun_filepaths:
                mcrun = MCRun.from_file(filepath=mcrun_filepath)

                sigmas = mcrun.last_sigma()

                structure = Structure(super_cell=scell, sigmas=sigmas)

                flag = {}
                property_value = property_model.predict(structure, flag=flag)

                if flag["computed_orbits_from_scratch"]:
                    print(
                        "Metropolis (task=averaged-property): CE Model computed orbits from scratch"
                    )
                    print(
                        "... serializing Model to save new orbits and save time in later calls"
                    )
                    property_model.serialize(filepath=model_filepath)

                data["property_values"].append(float(property_value))
                data["temperatures"].append(mcrun.temperature)

                data["fractional_concentrations"].append(
                    structure.get_fractional_concentrations()[
                        int(substitutional_sublattice)
                    ][1]
                )
                data["number_of_substituents"].append(np.count_nonzero(sigmas))

            if not np.all(data["temperatures"][0] == data["temperatures"]):
                plot_property(
                    data["temperatures"],
                    data["property_values"],
                    prop_name=data["property_name"],
                    xaxis_label="Temperature",
                    yaxis_label=data["property_name"],
                    show_plot=show_plot,
                    save_plot=save_plot,
                )

            if not np.all(
                data["number_of_substituents"][0]
                == np.array(data["number_of_substituents"])
            ):
                plot_property(
                    data["fractional_concentrations"],
                    data["property_values"],
                    prop_name=data["property_name"],
                    xaxis_label="Temperature",
                    yaxis_label=data["property_name"],
                    show_plot=show_plot,
                    save_plot=save_plot,
                )

            if plotdata_filepath is not None:
                with open(plotdata_filepath, "w+", encoding="utf-8") as outfile:
                    json.dump(data, outfile, indent=2, separators=(",", ":"))

            print(data)

        case "plot-mc-trajectory":
            traj = MonteCarloTrajectory(filename=traj_filepath, read=True)
            energies_accepted = traj.get_energies()
            steps_accepted = traj.get_sampling_step_nos()
            plot_property(
                steps_accepted,
                energies_accepted,
                prop_name="Energy of visited structures",
                xaxis_label="step no.",
                yaxis_label="Raw model energy (model units)",
            )
