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
from clusterx.thermodynamics.wang_landau import WangLandau

commands = ["wang_landau"]


@plac.annotations(
    nsubs=("Number of substitutions", "option", "nsubs", dict),
    model_filepath=("Output Model file path.", "option", "mof", str),
    plat_filepath=(
        "Filepath of a serialized ParentLattice object.",
        "option",
        "plfp",
        str,
    ),
    sc_shape=("3x3 integer matrix to specify supercell shape.", "option", "scsh", list),
    energy_range=(
        "Energy range for sampling, list of maximum and minimum",
        "option",
        "er",
        list,
    ),
)
def wang_landau(
    # class init arguments
    plat_filepath: str = "plat.json",
    model_filepath: str = "model.pickle",
    sc_shape: Optional[Union[int, List[int], List[List[int]]]] = 1,
    nsubs: dict = {0: [1]},
    ensemble: str = "canonical",
    sublattice_indices: List[int] = [],
    chemical_potentials: Optional[dict] = None,
    predict_swap: bool = True,
    error_reset: Optional[int] = None,
    # wang-landau sampling arguments
    energy_range: List[float] = [-2.0, 2.0],
    energy_bin_width: float = 0.2,
    f_range: List[float] = [math.exp(1), math.exp(1e-4)],
    update_method: str = "square_root",
    flatness_conditions: List[List[float]] = [
        [0.5, math.exp(1e-1)],
        [0.80, math.exp(1e-3)],
        [0.90, math.exp(1e-5)],
        [0.95, math.exp(1e-7)],
        [0.98, math.exp(1e-8)],
    ],
    initial_decoration: Optional[List[int]] = None,
    serialize: bool = True,
    filename: str = "cdos.json",
    serialize_during_sampling: bool = True,
    restart_from_file: bool = False,
    plot_hist_real_time: bool = True,
    acc_prob_init_structure: float = 1e-3,
    acc_prob_dist_init_structure: str = "gaussian",
    itmax_init_structure: int = int(1e8),
    nproc: int = 0,
    seed: Optional[int] = 1,
    **kwargs,
):
    """Perform Wang-Landau sampling"""
    cmd_message("head")

    print(f"Info({get_command_name()}): Initialization")
    plat = ParentLattice(filepath=plat_filepath)
    model = Model(filepath=model_filepath)
    scell = SuperCell(plat, sc_shape)

    wl = WangLandau(
        energy_model=model,
        energy_factor=1.0,
        scell=scell,
        nsubs=nsubs,
        ensemble=ensemble,
        sublattice_indices=sublattice_indices,
        chemical_potentials=chemical_potentials,
        predict_swap=predict_swap,
        error_reset=error_reset,
    )
    wl.wang_landau_sampling(
        energy_range=energy_range,
        energy_bin_width=energy_bin_width,
        f_range=f_range,
        update_method=update_method,
        flatness_conditions=flatness_conditions,
        initial_decoration=initial_decoration,
        serialize=serialize,
        filename=filename,
        serialize_during_sampling=serialize_during_sampling,
        restart_from_file=restart_from_file,
        plot_hist_real_time=plot_hist_real_time,
        acc_prob_init_structure=acc_prob_init_structure,
        acc_prob_dist_init_structure=acc_prob_dist_init_structure,
        itmax_init_structure=itmax_init_structure,
        nproc=nproc,
        seed=seed,
        **kwargs,
    )
