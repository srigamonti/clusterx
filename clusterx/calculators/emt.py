# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

"""EMT calculator with additional illustrative species parameters."""

from ase.calculators.emt import EMT, parameters as emt_parameters


emt_parameters.update(
    {
        # These illustrative values are not fitted physical parameters.
        #      E0     s0    V0     eta2    kappa   lambda  n0
        #      eV     bohr  eV     bohr^-1 bohr^-1 bohr^-1 bohr^-3
        "Fe": (-3.92, 2.92, 2.20, 1.55, 2.60, 1.82, 0.00781),
        "Si": (-3.62, 2.42, 1.85, 1.59, 2.68, 1.77, 0.00873),
        "Ge": (-3.14, 3.08, 2.04, 1.73, 2.91, 2.03, 0.00603),
        "Sn": (-4.02, 3.22, 2.88, 1.90, 3.21, 2.26, 0.00741),
        "Ba": (-2.71, 3.34, 2.57, 1.77, 2.96, 2.34, 0.00591),
        "He": (-1.12, 1.10, 0.09, 2.81, 3.01, 4.12, 0.00321),
        "Na": (-2.47, 2.16, 0.43, 1.81, 2.61, 2.06, 0.01181),
        "Zn": (-3.72, 2.75, 2.25, 1.73, 2.93, 2.07, 0.00982),
        "B": (-4.80, 1.83, 0.232, 1.652, 2.790, 1.892, 0.01252),
        "X": (-3.38, 1.72, 0.41, 1.83, 2.94, 2.11, 0.01405),
        "Cs": (-2.05, 3.50, 2.90, 1.48, 2.48, 1.65, 0.00421),
        "I": (-4.25, 2.45, 2.15, 1.95, 3.32, 2.42, 0.00836),
        "Br": (-3.05, 2.68, 2.68, 1.72, 3.05, 2.18, 0.00672),
        "Pb": (-3.44, 3.18, 2.75, 1.88, 3.15, 2.25, 0.00792),
    }
)


class ToyEMText(EMT):
    """EMT calculator with the additional species parameters above."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


EMT2 = ToyEMText
