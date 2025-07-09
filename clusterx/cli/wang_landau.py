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
    plat_filepath=("Filepath of a serialized ParentLattice object.", "option", "plfp", str),
    sc_shape=("3x3 integer matrix to specify supercell shape.", "option", "scsh", list),
    energy_range=("Energy range for sampling, list of maximum and minimum", "option", "er", list),
)
def wang_landau(
    nsubs: dict = {0: [1]},
    model_filepath: str = "model.pickle",
    plat_filepath: str = "plat.json",
    sc_shape: Optional[Union[int, List[int], List[List[int]]]] = 1,
    energy_range: List[float] = [-2., 2.],
):
    """Perform Wang-Landau sampling"""
    cmd_message("head")

    print(f"Info({get_command_name()}): Initialization")
    model = Model(filepath=model_filepath)
    plat = ParentLattice(filepath=plat_filepath)
    scell = SuperCell(plat, sc_shape)

    wl = WangLandau(
        energy_model=model, scell=scell, ensemble="canonical", nsubs=nsubs, predict_swap=True,
    )
    cdos = wl.wang_landau_sampling(
        energy_range=energy_range,
        energy_bin_width=0.2,
        f_range=[math.exp(1), math.exp(1e-4)],
        update_method="square_root",
        flatness_conditions=[
            [0.5, math.exp(1e-1)],
            [0.80, math.exp(1e-3)],
            [0.90, math.exp(1e-5)],
            [0.95, math.exp(1e-7)],
            [0.98, math.exp(1e-8)],
        ],
        initial_decoration=None,
        serialize=True,
        filename="cdos.json",
        serialize_during_sampling=True,
        restart_from_file=False,
        plot_hist_real_time=False,
        acc_prob_init_structure=1e-3,
        acc_prob_dist_init_structure="gaussian",
        itmax_init_structure=int(1e8),
        nproc=0,
        #**kwargs,
    )
    energy_bins, gs = cdos.get_cdos(ln=True, normalization=False)
