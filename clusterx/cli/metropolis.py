# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
import random
from typing import List, Optional, Union

import numpy as np
import plac
import json

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
    sc_shape=("3x3 integer matrix to specify supercell shape.", "option", "scsh", list),
    n_substitutions=("Number of substitutions", "option", "nsubs", dict),
    ensemble=("Thermodynamic ensemble", "option", "ens", str),
    sublattice_indices=("Sublattice indices", "option", "sublat", list),
    chemical_potentials=(
        "Chemical potentials for each sublattice",
        "option",
        "chem_pot",
        dict,
    ),
    models_aux_filepaths=(
        "Filepaths of auxiliary models to be used in sampling",
        "option",
        "aux_models",
        list,
    ),
    no_of_swaps=(
        "Number of swaps in metropolis sampling",
        "option",
        "n_swaps",
        int,
    ),
    predict_swap=(
        "Use predict_swap during energy prediction",
        "option",
        "pred_swap",
        bool,
    ),
    n_error_reset=(
        "Reset error after this many steps for numerical accuracy",
        "option",
        "err_reset",
        int,
    ),
    n_mc_steps=(
        "Number of metropolis steps to perform",
        "option",
        "n_steps",
        int,
    ),
    energy_scale_factor=(
        "Float is used to adjust the energy from ``energy_model`` to give the total energy for the simulation supercell",
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
        str,
    ),
    initial_decoration=(
        "Initial decoration of the supercell, list of integers",
        "option",
        "init_dec",
        list,
    ),
    acceptance_ratio=(
        """Real number between 0 and 100. Represents the percentage of accepted moves.
        If not ``None``, the initial temperature will be adjusted to match the given
        acceptance ratio. The acceptance ratio during the simulation is computed using
        the last 100 moves.""",
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
    plotdata_filepath: str =None,
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
    no_of_swaps: int = 1,
    predict_swap: bool = True,
    n_error_reset: Optional[int] = None,
    # metropolis sampling arguments
    n_mc_steps: int = 100,
    n_mc_eq: int = 1,
    n_clics: int = 1,
    runs: Optional[dict] = None,
    energy_scale_factor: float | None = None,
    temperature: float = 1.0,
    temperatures: Optional[List[float]] = None,
    boltzmann_constant: float = 1.0,
    initial_decoration: Optional[List[int]] = None,
    acceptance_ratio: Optional[float] = None,
    # deprecated
    scale_factor: Optional[List[float]] = None,  # use energy_scale_factor instead
    filename: Optional[str] = None,  # use traj_filepath instead
    random_seed: Optional[int] = None,
    **sampling_kwargs,
):
    """Perform Wang-Landau sampling
    Deprecated:
        scale_factor: Deprecated. Use energy_scale_factor instead."""
    cmd_message("head")

    energy_scale_factor = _process_deprecated(
        energy_scale_factor, scale_factor, "energy_scale_factor", "scale_factor"
    )
    traj_filepath = _process_deprecated(
        traj_filepath, filename, "traj_filepath", "filename"
    )

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    match task:
        case "runmc" | "runMC" | "run-monte-carlo":
            print(f"Info({get_command_name()}): Initialization")

            if isinstance(n_substitutions, dict):
                n_substitutions = {int(k): v for k, v in n_substitutions.items()}
            elif isinstance(n_substitutions, int):
                n_substitutions = {0: [n_substitutions]}

            plat = ParentLattice(filepath=plat_filepath)
            energy_model = Model(filepath=model_filepath)
            scell = SuperCell(plat, sc_shape)
            models_aux = [Model(filepath=fp) for fp in models_aux_filepaths]

            mc = MonteCarlo(
                energy_model=energy_model,
                scell=scell,
                nsubs=n_substitutions,
                ensemble=ensemble,
                sublattice_indices=sublattice_indices,
                chemical_potentials=chemical_potentials,
                models=models_aux,
                no_of_swaps=no_of_swaps,
                predict_swap=predict_swap,
                error_reset=n_error_reset,
                filename=traj_filepath,
            )
            mc.metropolis(
                no_of_sampling_steps=no_of_sampling_steps,
                scale_factor=[1 / energy_scale_factor],
                temperature=temperature,
                boltzmann_constant=boltzmann_constant,
                initial_decoration=initial_decoration,
                acceptance_ratio=acceptance_ratio,
                serialize=True,
                filename=traj_filepath,
                **sampling_kwargs,
            )

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

            for irun, run in enumerate(runs):
                temperature = run["temperature"]
                n_mc_steps = run["n_mc_steps"]
                mcrun_filepath = run["mcrun_filepath"]
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
                        is_simulated_annelaing=True,
                        keep_sigmas=keep_sigmas,
                    )
                else:
                    mcrun = mclite.metropolis(
                        temperature=temperature,
                        n_mc_steps=n_mc_steps,
                        ensemble="canonical",
                        mcrun_filepath=mcrun_filepath,
                        n_clics=n_clics,
                        is_simulated_annelaing=True,
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
                show_plot = show_plot,
                save_plot =save_plot
            )

            if plotdata_filepath is not None:
                with open(plotdata_filepath, "w+", encoding="utf-8") as outfile:
                    json.dump(
                        plot_data, outfile, indent=2, separators=(",", ":")
                    )
                    
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



        case "info-mc-run":
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
