# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
from typing import List, Optional, Union

import plac

from clusterx.cli.config_utils import cmd_message, get_command_name
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.thermodynamics.monte_carlo import MonteCarlo, MonteCarloTrajectory
from clusterx.utils import _process_deprecated

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
    traj_filepath=(
        "Filename for the output trajectory",
        "option",
        "trfp",
        str,
    ),
    sc_shape=("3x3 integer matrix to specify supercell shape.", "option", "scsh", list),
    nsubs=("Number of substitutions", "option", "nsubs", dict),
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
    error_reset=(
        "Reset error after this many steps for numerical accuracy",
        "option",
        "err_reset",
        int,
    ),
    no_of_sampling_steps=(
        "Number of metropolis steps to perform",
        "option",
        "n_steps",
        int,
    ),
    energy_scale_factor=(
        "Float is used to adjust the energy from ``energy_model`` to give the total energy for the simulation supercell",
        "option",
        "scf",
        float,
    ),
    temperature=(
        "Temperature at which the sampling is performed.",
        "option",
        "temp",
        float,
    ),
    boltzmann_constant=("Boltzmann constant", "option", "k_B", float),
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
    filename=(
        "Filename for the output trajectory",
        "option",
        "fn",
        str,
    ),
    scale_factor=(
        "Deprecated: use ``energy_scale_factor instead``",
        "option",
        "scf",
        list,
    ),
)
def metropolis(
    # class init arguments
    task: str = "usage",
    plat_filepath: str = "plat.json",
    model_filepath: str = "model.pickle",
    traj_filepath: str = "trajectory.json",
    sc_shape: Optional[Union[int, List[int], List[List[int]]]] = 1,
    nsubs: Optional[Union[int, dict]] = None,
    ensemble: str = "canonical",
    sublattice_indices: List[int] = [],
    chemical_potentials: Optional[dict] = None,
    models_aux_filepaths: List[str] = [],
    no_of_swaps: int = 1,
    predict_swap: bool = True,
    error_reset: Optional[int] = None,
    # metropolis sampling arguments
    no_of_sampling_steps: int = 100,
    energy_scale_factor: float = 1.0,
    temperature: float = 1.0,
    boltzmann_constant: float = 1.0,
    initial_decoration: Optional[List[int]] = None,
    acceptance_ratio: Optional[float] = None,
    # deprecated
    scale_factor: Optional[List[float]] = None,  # use energy_scale_factor instead
    filename: Optional[str] = None,  # use traj_filepath instead
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
    if isinstance(nsubs, dict):
        nsubs = {int(k): v for k, v in nsubs.items()}
    elif isinstance(nsubs, int):
        nsubs = {0: [nsubs]}

    match task:
        case "runmc" | "runMC" | "run-monte-carlo":
            print(f"Info({get_command_name()}): Initialization")

            plat = ParentLattice(filepath=plat_filepath)
            energy_model = Model(filepath=model_filepath)
            scell = SuperCell(plat, sc_shape)
            models_aux = [Model(filepath=fp) for fp in models_aux_filepaths]

            mc = MonteCarlo(
                energy_model=energy_model,
                scell=scell,
                nsubs=nsubs,
                ensemble=ensemble,
                sublattice_indices=sublattice_indices,
                chemical_potentials=chemical_potentials,
                models=models_aux,
                no_of_swaps=no_of_swaps,
                predict_swap=predict_swap,
                error_reset=error_reset,
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
        case "plot-mc-trajectory":
            from clusterx.visualization import plot_property

            traj = MonteCarloTrajectory(filename=traj_filepath, read=True)
            energies_accepted = traj.get_energies()
            steps_accepted = traj.get_sampling_step_nos()
            print(energies_accepted)
            print(steps_accepted)
            plot_property(
                steps_accepted,
                energies_accepted,
                prop_name="Energy of visited structures",
                xaxis_label="step no.",
                yaxis_label="Energy [eV/#sites]",
            )
