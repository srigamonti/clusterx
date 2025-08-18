# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.
from typing import Optional, Union, List
import math

import plac
from clusterx.cli.config_utils import cmd_message, get_command_name
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.thermodynamics.monte_carlo import MonteCarlo

commands = ["metropolis"]


@plac.annotations(
    plat_filepath=(
        "Filepath of a serialized ParentLattice object.",
        "option",
        "plfp",
        str,
    ),
    model_filepath=("Output Model file path.", "option", "mof", str),
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
    filename=(
        "Filename for the output trajectory",
        "option",
        "fn",
        str,
    ),
    no_of_sampling_steps=(
        "Number of metropolis steps to perform",
        "option",
        "n_steps",
        int,
    ),
    scale_factor=(
        "List is used to adjust the factor :math:`k_B T` to the same units as the energy from ``energy_model``",
        "option",
        "scf",
        list,
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
)
def metropolis(
    # class init arguments
    plat_filepath: str = "plat.json",
    model_filepath: str = "model.pickle",
    sc_shape: Optional[Union[int, List[int], List[List[int]]]] = 1,
    nsubs: dict = {0: [1]},
    ensemble: str = "canonical",
    sublattice_indices: List[int] = [],
    chemical_potentials: Optional[dict] = None,
    models_aux_filepaths: List[str] = [],
    no_of_swaps: int = 1,
    predict_swap: bool = True,
    error_reset: Optional[int] = None,
    filename: str = "trajectory.json",
    # metropolis sampling arguments
    no_of_sampling_steps: int = 100,
    scale_factor: List[float] = [1.0],
    temperature: float = 1.0,
    boltzmann_constant: float = 1.0,
    initial_decoration: Optional[List[int]] = None,
    acceptance_ratio: Optional[float] = None,
    **sampling_kwargs,
):
    """Perform Wang-Landau sampling"""
    cmd_message("head")

    print(f"Info({get_command_name()}): Initialization")
    plat = ParentLattice(filepath=plat_filepath)
    energy_model = Model(filepath=model_filepath)
    scell = SuperCell(plat, sc_shape)
    models_aux = [Model(filepath=fp) for fp in models_aux_filepaths]
    if isinstance(nsubs, dict):
        nsubs = {int(k): v for k, v in nsubs.items()}
    elif isinstance(nsubs, int):
        nsubs = {0: [nsubs]}

    mc = MonteCarlo(
        energy_model=energy_model,
        scell=scell,
        nsubs=nsubs,
        ensemble="canonical",
        sublattice_indices=sublattice_indices,
        chemical_potentials=chemical_potentials,
        models=models_aux,
        no_of_swaps=no_of_swaps,
        predict_swap=predict_swap,
        error_reset=error_reset,
        filename=filename,
    )
    mc.metropolis(
        no_of_sampling_steps=no_of_sampling_steps,
        scale_factor=scale_factor,
        temperature=temperature,
        boltzmann_constant=boltzmann_constant,
        initial_decoration=initial_decoration,
        acceptance_ratio=acceptance_ratio,
        serialize=True,
        filename=filename,
        **sampling_kwargs,
    )
