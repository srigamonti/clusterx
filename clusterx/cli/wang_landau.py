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
    nsubs=("Number of substitutions", "option", 1, int),
    model_filepath=("Output Model file path.", "option", "mof", str),
    plat_filepath=("Filepath of a serialized ParentLattice object.", "option", "plfp", str),
    sc_shape=("3x3 integer matrix to specify supercell shape.", "positional", "scsh", list),
    energy_range=("Energy range for sampling, list of maximum and minimum", "option", "er", list),
)
def wang_landau(
    nsubs: int,
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
        energy_model=model, scell=scell, ensemble="canonical", nsubs=nsubs
    )
    cdos = wl.wang_landau_sampling(
        energy_range=energy_range,
        energy_bin_width=0.002,
        f_range=[math.exp(1), 2],
        update_method="square_root",
        flatness_conditions=[[0.1, math.exp(1e-1)]],
    )
    energy_bins, gs = cdos.get_cdos(ln=True, normalization=False)
