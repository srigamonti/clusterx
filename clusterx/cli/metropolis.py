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
from clusterx.thermodynamics.monte_carlo import MonteCarlo, MonteCarloTrajectory
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
    save_atoms_every=("Checkpoint interval in proposals; zero disables periodic output", "option", "save_every", int),
    atoms_filename=("XYZ/extxyz or ASE checkpoint output", "option", "atoms", str),
    save_atoms_endpoints=("Save initial configuration and each temperature endpoint", "option", "endpoints", _boolean),
    atoms_mode=("Checkpoint mode: x=new file, w=overwrite, a=append, auto=legacy", "option", "atoms_mode", str),
    max_steps_per_batch=("Maximum temporary temperature array length", "option", "batch", int),
    nonlinear=("Cache global correlations for nonlinear models", "option", "nonlinear", _boolean),
    continue_chain=("Continue in memory; schedule stages always continue", "option", "continue", _boolean),
    serialize=("Store accepted-move histories as JSON; use false for large runs", "option", "serialize", _boolean),
    energy_key=("Energy property name for multi-output models", "option", "energy_key", str),
    energy_normalization=("Energy normalization: per_atom, per_parent_cell, or total", "option", "energy_norm", str),
    energy_scale_factor=(
        "Divide model energy differences by this factor to obtain supercell energy differences",
        "option",
        "escf",
        float,
    ),
    temperature=(
        "Temperature at which the sampling is performed.",
        "option",
        "temp",
        float,
    ),
    boltzmann_constant=("Boltzmann constant", "option", "k_B", float),
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
    scale_factor=(
        "Deprecated: use ``energy_scale_factor instead``",
        "option",
        "scf",
        list,
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
    energy_scale_factor: float | None = None,
    temperature: float | None = None,
    temperatures: Optional[List[float]] = None,
    boltzmann_constant: float = 1.0,
    initial_decoration: Optional[List[int]] = None,
    target_rate: Optional[float] = None,
    random_seed: Optional[int] = None,
    # deprecated
    scale_factor: Optional[List[float]] = None,  # use energy_scale_factor instead
    filename: Optional[str] = None,  # use traj_filepath instead
    # Maintained MC constructor and sampling options.
    resume_from: str | None = None,
    save_atoms_every: int = 0,
    atoms_filename: str = "metropolis.traj",
    save_atoms_endpoints: bool = False,
    atoms_mode: str = "auto",
    energy_key: str = "energy",
    n_mc_sweeps: float | None = None,
    energy_normalization: str | None = None,
    max_steps_per_batch: int = 100_000,
    nonlinear: bool = False,
    continue_chain: bool = False,
    serialize: bool = True,
    **sampling_kwargs,
):
    """Perform Metropolis sampling
    Deprecated:
    scale_factor: Deprecated. Use energy_scale_factor instead."""
    cmd_message("head")

    energy_scale_factor = _process_deprecated(
        energy_scale_factor, scale_factor, "energy_scale_factor", "scale_factor"
    )
    traj_filepath = _process_deprecated(
        traj_filepath, filename, "traj_filepath", "filename"
    )
    if task not in {"runmc", "runMC", "run-monte-carlo"} and n_mc_steps is None:
        n_mc_steps = 100
    if task not in {"runmc", "runMC", "run-monte-carlo"} and temperature is None:
        temperature = 1.0

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    match task:
        case "runmc" | "runMC" | "run-monte-carlo":
            print(f"Info({get_command_name()}): Initialization")

            aliases = {"seed": "random_seed", "nsubs": "n_substitutions", "scell": "sc_shape",
                       "energy_model": "model_filepath", "models": "models_aux_filepaths",
                       "no_of_sampling_steps": "n_mc_steps"}
            for name in sampling_kwargs.keys() & aliases.keys():
                raise ValueError(f"Use the CLI option {aliases[name]!r} instead of Python API option {name!r}")

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
            if resume_from is not None and runs is None and (
                temperature is None or (n_mc_steps is None and n_mc_sweeps is None)
            ):
                raise ValueError("Resume requires an explicit runs schedule, or temperature and a step/sweep count")
            stages = _sampling_schedule(
                runs, 1.0 if temperature is None else temperature,
                n_mc_steps, n_mc_sweeps, scell.get_n_sub_sites(unique=False),
            )
            if energy_normalization is not None and energy_scale_factor is not None:
                raise ValueError("Specify either energy_normalization or energy_scale_factor")
            normalizations = {"per_atom": len(scell), "per_parent_cell": scell.get_index(), "total": 1}
            if energy_normalization is not None:
                if energy_normalization not in normalizations:
                    raise ValueError("energy_normalization must be per_atom, per_parent_cell, or total")
                sampling_scale = [normalizations[energy_normalization]]
            else:
                # Preserve the CLI convention: energy_scale_factor is a divisor.
                if energy_scale_factor is not None and (
                    not np.isfinite(energy_scale_factor) or energy_scale_factor <= 0
                ):
                    raise ValueError("energy_scale_factor must be finite and positive")
                sampling_scale = [1 / energy_scale_factor] if energy_scale_factor is not None else None

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
                    scale_factor=sampling_scale,
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
                    **sampling_kwargs,
                )
            return trajectory

        case "setup" | "setupmclite" | "setupMClite" | "setup-monte-carlo-lite":
            print(f"Info({get_command_name()}): Setting up MC")

            plat = ParentLattice(filepath=plat_filepath)
            with _timed("CLI.metropolis (task=setupmclite): unpickling model"):
                energy_model = Model(filepath=model_filepath)

            with _timed("CLI.metropolis (task=setupmclite): building supercell"):
                scell = SuperCell(plat, sc_shape)

            with _timed("CLI.metropolis (task=setupmclite): Create MC instance"):
                mclite = MonteCarloLite(
                    energy_model=energy_model,
                    scell=scell,
                    energy_scale_factor=energy_scale_factor,
                    boltzmann_constant=boltzmann_constant,
                )

            with _timed("CLI.metropolis(setupMClite): MonteCarloLite serialization"):
                mclite.serialize(mcsetup_filepath)

        case (
            "run"
            | "run-metropolis"
            | "runmclite"
            | "runMClite"
            | "run-monte-carlo-lite"
        ):
            print(f"Info({get_command_name()}): Read MC setup")

            with _timed("from_file"):
                mclite = MonteCarloLite.from_file(mcsetup_filepath)

            print(f"Info({get_command_name()}): Running Metropolis MC simulation")

            n_substitutions0 = n_substitutions

            if runs is None:
                runs = [{}]

            for irun, run in enumerate(runs):
                temperature = run.get("temperature", temperature)
                n_mc_steps = run.get("n_mc_steps", n_mc_steps)
                mcrun_filepath = run.get("mcrun_filepath", mcrun_filepath)
                ignore = run.get("ignore", False)
                equilibration_run = run.get("equilibration_run", False)
                n_substitutions = run.get("n_substitutions", n_substitutions0)

                if ignore:
                    continue

                mcrun = mclite.metropolis(
                    temperature=temperature,
                    n_mc_steps=n_mc_steps,
                    ensemble=ensemble,
                    n_substitutions=n_substitutions,
                    mcrun_filepath=mcrun_filepath,
                    random_seed=random_seed,
                    n_clics=n_clics,
                    keep_sigmas=keep_sigmas,
                )

        case "run-simulated-annealing":
            print(f"Info({get_command_name()}): Reading MC setup")

            with _timed("from_file"):
                mclite = MonteCarloLite.from_file(mcsetup_filepath)

            print(f"Info({get_command_name()}): Running simulated annealing")

            for irun, run in enumerate(runs):
                temperature = run["temperature"]
                n_mc_steps = run["n_mc_steps"]
                mcrun_filepath = run["mcrun_filepath"]
                ignore = run.get("ignore", False)
                equilibration_run = run.get("equilibration_run", False)

                if ignore:
                    continue

                if equilibration_run:
                    mcrun = mclite.metropolis(
                        temperature=temperature,
                        n_mc_steps=n_mc_steps,
                        ensemble="canonical",
                        n_substitutions=n_substitutions,
                        mcrun_filepath=mcrun_filepath,
                        random_seed=random_seed,
                        n_clics=n_clics,
                        is_simulated_annealing=True,
                        keep_sigmas=keep_sigmas,
                    )
                else:
                    mcrun = mclite.metropolis(
                        temperature=temperature,
                        n_mc_steps=n_mc_steps,
                        ensemble="canonical",
                        mcrun_filepath=mcrun_filepath,
                        n_clics=n_clics,
                        is_simulated_annealing=True,
                        keep_sigmas=keep_sigmas,
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
                yaxis_label="Energy [eV/#sites]",
            )
