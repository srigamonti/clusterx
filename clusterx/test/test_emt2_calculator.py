# Copyright (c) 2015-2026, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.calculators import EMT2, ToyEMText
from ase import Atoms
from ase.calculators.emt import parameters as emt_parameters
import numpy as np


def test_emt2_parameters_are_unique():
    """Check that the added species have distinct parameter sets."""
    extra_symbols = ("Fe", "Si", "Ge", "Sn", "Ba", "He", "Na", "Zn", "B", "X", "Cs", "I", "Br", "Pb")
    builtin_symbols = ("Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt", "H", "C", "N", "O")
    extra_parameters = [emt_parameters[symbol] for symbol in extra_symbols]
    builtin_parameters = {emt_parameters[symbol] for symbol in builtin_symbols}

    assert len(set(extra_parameters)) == len(extra_parameters)
    assert all(parameters not in builtin_parameters for parameters in extra_parameters)


def test_emt2_is_compatibility_alias():
    """Check that the previous calculator name remains available."""
    assert EMT2 is ToyEMText


def test_toy_emt_ext_calculates_new_species():
    """Check that the newly added species can be evaluated by ToyEMText."""
    for symbol in ("Cs", "I", "Br", "Pb"):
        atoms = Atoms([symbol, symbol], positions=[[0, 0, 0], [2.5, 0, 0]], cell=[12, 12, 12], pbc=True)
        atoms.calc = ToyEMText()
        assert np.isfinite(atoms.get_potential_energy())


def test_emt2_calculator():
    """Test calcualtion of energies with EMT2 calculator.

    After successful execution of the test, the generated structures may be visualized with the command::

        ase gui test_emt2_calculator#.json

    """

    cell = [[3, 0, 0], [0, 1, 0], [0, 0, 5]]
    positions = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    pbc = [True, True, False]

    pri = Atoms(["H", "H", "H"], positions=positions, cell=cell, pbc=pbc)
    su1 = Atoms(["C", "H", "H"], positions=positions, cell=cell, pbc=pbc)
    su2 = Atoms(["H", "He", "H"], positions=positions, cell=cell, pbc=pbc)
    su3 = Atoms(["H", "N", "H"], positions=positions, cell=cell, pbc=pbc)

    plat = ParentLattice(pri, substitutions=[su1, su2, su3], pbc=pbc)

    scell = SuperCell(plat, np.array([(1, 0, 0), (0, 3, 0), (0, 0, 1)]))
    strset = StructuresSet(plat)
    strset.add_structure(
        Structure(scell, [1, 2, 1, 6, 7, 1, 1, 2, 1]), write_to_db=True
    )
    strset.add_structure(
        Structure(scell, [6, 1, 1, 1, 1, 1, 1, 1, 1]), write_to_db=True
    )
    strset.add_structure(
        Structure(scell, [1, 2, 1, 1, 7, 1, 6, 7, 1]), write_to_db=True
    )
    strset.add_structure(
        Structure(scell, [1, 1, 1, 6, 1, 1, 1, 7, 1]), write_to_db=True
    )
    strset.add_structure(
        Structure(scell, [6, 7, 1, 6, 7, 1, 6, 2, 1]), write_to_db=True
    )
    strset.add_structure(
        Structure(scell, [1, 7, 1, 6, 2, 1, 1, 1, 1]), write_to_db=True
    )
    strset.add_structure(
        Structure(scell, [6, 1, 1, 1, 1, 1, 1, 1, 1]), write_to_db=True
    )
    strset.add_structure(
        Structure(scell, [1, 1, 1, 1, 7, 1, 6, 7, 1]), write_to_db=True
    )
    strset.add_structure(
        Structure(scell, [1, 2, 1, 6, 2, 1, 6, 2, 1]), write_to_db=True
    )
    strset.add_structure(
        Structure(scell, [6, 1, 1, 6, 2, 1, 1, 1, 1]), write_to_db=True
    )
    strset.serialize(
        filepath="test_cluster_expansion_structures_set.json", overwrite=True
    )

    strset.set_calculator(EMT2())
    energies = strset.compute_property_values()

    # Generate output
    print("\n\n========Test writes========")
    print(test_emt2_calculator.__doc__)
    print("Energies:\n")
    print(np.array2string(np.array(energies), separator=",", max_line_width=1000))
    print("===========================\n")

    print("========Asserts========")

    np.testing.assert_allclose(
        [
            18.87160014,
            11.85277241,
            32.19105935,
            27.15487578,
            41.57652786,
            24.46212886,
            11.85277241,
            35.84806650,
            15.81521831,
            18.89778943,
        ],
        energies,
        atol=1e-5,
    )
    # print(comat)
