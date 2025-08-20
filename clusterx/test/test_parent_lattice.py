# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import pytest
import numpy as np
from ase.build import bulk
from ase.spacegroup import crystal

from clusterx.parent_lattice import ParentLattice
from clusterx.utils import dict_compare
from clusterx.super_cell import SuperCell


@pytest.fixture
def pristine():
    return bulk("Cu", "fcc", a=3.6)


def test_with_symbols(pristine):
    symbols = [["Cu", "Au"]]
    plat = ParentLattice(atoms=pristine, symbols=symbols)
    plat = ParentLattice(atoms=pristine, site_symbols=symbols)
    np.testing.assert_array_equal(plat.get_sublattice_types()[0], [29, 79])


def test_with_numbers(pristine):
    plat = ParentLattice(atoms=pristine, numbers=[[29, 79]])
    np.testing.assert_array_equal(plat.get_sublattice_types()[0], [29, 79])


def test_parent_lattice_creation():
    """Test creation of a parent lattice for a fictitious quaternary zincblende crystal and for a Ternary clathrate.

    After successful execution of the test, the generated parent lattices may be visualized with the command::

        ase gui test_parent_lattice_creation#.json
    """
    # Quaternary zincblende crystal
    cual = bulk("CuAl", "zincblende", a=6.1)
    agal = bulk("AgAl", "zincblende", a=6.1)
    sral = bulk("SrAl", "zincblende", a=6.1)
    cuc = bulk("CuC", "zincblende", a=6.1)

    parent_lattice0 = ParentLattice(atoms=cual, substitutions=[agal, sral, cuc])
    parent_lattice0.serialize(fname="test_parent_lattice_creation_0.json")
    # Clathrate
    a = 10.515
    x, y, z = 0.185, 0.304, 0.116
    wyckoff = [
        (0, y, z),  # 24k
        (x, x, x),  # 16i
        (1 / 4.0, 0, 1 / 2.0),  # 6c
        (1 / 4.0, 1 / 2.0, 0),  # 6d
        (0, 0, 0),  # 2a
    ]

    pri = crystal(
        ["Si", "Si", "Si", "Ba", "Ba"],
        wyckoff,
        spacegroup=223,
        cellpar=[a, a, a, 90, 90, 90],
    )
    sub1 = crystal(
        ["Al", "Al", "Al", "Ba", "Ba"],
        wyckoff,
        spacegroup=223,
        cellpar=[a, a, a, 90, 90, 90],
    )
    sub2 = crystal(
        ["X", "X", "X", "Ba", "Ba"],
        wyckoff,
        spacegroup=223,
        cellpar=[a, a, a, 90, 90, 90],
    )
    sub3 = crystal(
        ["Si", "Si", "Si", "Sr", "Ba"],
        wyckoff,
        spacegroup=223,
        cellpar=[a, a, a, 90, 90, 90],
    )

    parent_lattice1 = ParentLattice(atoms=pri, substitutions=[sub1, sub2, sub3])
    parent_lattice1.print_sublattice_types()
    parent_lattice1.serialize(fname="test_parent_lattice_creation_1.json")

    parent_lattice2 = ParentLattice(filepath="test_parent_lattice_creation_1.json")

    print("\n\n========Test writes========")
    print(test_parent_lattice_creation.__doc__)
    print("===========================\n")

    print("========Asserts========")
    assert check_result(parent_lattice0, 0)
    assert check_result(parent_lattice1, 1)
    assert parent_lattice2 == parent_lattice1
    assert parent_lattice2 != parent_lattice0


def check_result(parent_lattice, case):
    atol = 1e-8
    isok = True

    if case == 0:
        positions = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
        sites = {0: np.array([29, 47, 38]), 1: np.array([13, 6])}
        idx_subs = {0: np.array([13, 6]), 1: np.array([29, 47, 38])}
        tags = [1, 0]

    if case == 1:
        positions = np.array(
            [
                [0.0, 0.304, 0.116],
                [0.0, 0.696, 0.116],
                [0.0, 0.304, 0.884],
                [0.0, 0.696, 0.884],
                [0.116, 0.0, 0.304],
                [0.116, 0.0, 0.696],
                [0.884, 0.0, 0.304],
                [0.884, 0.0, 0.696],
                [0.304, 0.116, 0.0],
                [0.696, 0.116, 0.0],
                [0.304, 0.884, 0.0],
                [0.696, 0.884, 0.0],
                [0.804, 0.5, 0.384],
                [0.196, 0.5, 0.384],
                [0.804, 0.5, 0.616],
                [0.196, 0.5, 0.616],
                [0.5, 0.616, 0.196],
                [0.5, 0.616, 0.804],
                [0.5, 0.384, 0.196],
                [0.5, 0.384, 0.804],
                [0.616, 0.804, 0.5],
                [0.616, 0.196, 0.5],
                [0.384, 0.804, 0.5],
                [0.384, 0.196, 0.5],
                [0.185, 0.185, 0.185],
                [0.815, 0.815, 0.185],
                [0.815, 0.185, 0.815],
                [0.185, 0.815, 0.815],
                [0.685, 0.685, 0.315],
                [0.315, 0.315, 0.315],
                [0.685, 0.315, 0.685],
                [0.315, 0.685, 0.685],
                [0.815, 0.815, 0.815],
                [0.185, 0.185, 0.815],
                [0.185, 0.815, 0.185],
                [0.815, 0.185, 0.185],
                [0.315, 0.315, 0.685],
                [0.685, 0.685, 0.685],
                [0.315, 0.685, 0.315],
                [0.685, 0.315, 0.315],
                [0.25, 0.0, 0.5],
                [0.75, 0.0, 0.5],
                [0.5, 0.25, 0.0],
                [0.5, 0.75, 0.0],
                [0.0, 0.5, 0.25],
                [0.0, 0.5, 0.75],
                [0.25, 0.5, 0.0],
                [0.75, 0.5, 0.0],
                [0.0, 0.25, 0.5],
                [0.0, 0.75, 0.5],
                [0.5, 0.0, 0.25],
                [0.5, 0.0, 0.75],
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
            ]
        )
        sites = {
            0: np.array([14, 13, 0]),
            1: np.array([14, 13, 0]),
            2: np.array([14, 13, 0]),
            3: np.array([14, 13, 0]),
            4: np.array([14, 13, 0]),
            5: np.array([14, 13, 0]),
            6: np.array([14, 13, 0]),
            7: np.array([14, 13, 0]),
            8: np.array([14, 13, 0]),
            9: np.array([14, 13, 0]),
            10: np.array([14, 13, 0]),
            11: np.array([14, 13, 0]),
            12: np.array([14, 13, 0]),
            13: np.array([14, 13, 0]),
            14: np.array([14, 13, 0]),
            15: np.array([14, 13, 0]),
            16: np.array([14, 13, 0]),
            17: np.array([14, 13, 0]),
            18: np.array([14, 13, 0]),
            19: np.array([14, 13, 0]),
            20: np.array([14, 13, 0]),
            21: np.array([14, 13, 0]),
            22: np.array([14, 13, 0]),
            23: np.array([14, 13, 0]),
            24: np.array([14, 13, 0]),
            25: np.array([14, 13, 0]),
            26: np.array([14, 13, 0]),
            27: np.array([14, 13, 0]),
            28: np.array([14, 13, 0]),
            29: np.array([14, 13, 0]),
            30: np.array([14, 13, 0]),
            31: np.array([14, 13, 0]),
            32: np.array([14, 13, 0]),
            33: np.array([14, 13, 0]),
            34: np.array([14, 13, 0]),
            35: np.array([14, 13, 0]),
            36: np.array([14, 13, 0]),
            37: np.array([14, 13, 0]),
            38: np.array([14, 13, 0]),
            39: np.array([14, 13, 0]),
            40: np.array([14, 13, 0]),
            41: np.array([14, 13, 0]),
            42: np.array([14, 13, 0]),
            43: np.array([14, 13, 0]),
            44: np.array([14, 13, 0]),
            45: np.array([14, 13, 0]),
            46: np.array([56, 38]),
            47: np.array([56, 38]),
            48: np.array([56, 38]),
            49: np.array([56, 38]),
            50: np.array([56, 38]),
            51: np.array([56, 38]),
            52: np.array([56]),
            53: np.array([56]),
        }

        idx_subs = {0: np.array([14, 13, 0]), 1: np.array([56, 38]), 2: np.array([56])}

        tags = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
        ]

    if not np.allclose(parent_lattice.get_scaled_positions(), positions, atol=atol):
        return False

    if not dict_compare(sites, parent_lattice.get_sites()):
        return False

    if not dict_compare(idx_subs, parent_lattice.get_idx_subs()):
        return False

    pltags = parent_lattice.get_tags()
    if not (len(tags) == len(pltags)):
        return False
    else:
        for v1, v2 in zip(tags, pltags):
            if v1 != v2:
                return False

    return isok


def test_is_nary():
    """Test ParentLattice.is_nary() method"""
    cu = bulk("Cu", "fcc")
    au = bulk("Au", "fcc")
    ag = bulk("Ag", "fcc")

    plat = ParentLattice(atoms=cu, substitutions=[au, ag])

    scell = SuperCell(plat, [[2, 2, -2], [2, -2, 2], [-2, 2, 2]])

    s = scell.gen_random_structure(nsubs={0: [10, 3]})

    isok1 = (s.is_nary(3) is True) and (s.is_nary(4) is False)

    plat2 = ParentLattice(atoms=cu, site_symbols=[["Cu", "Au", "Ag", "X"]])

    isok2 = (plat2.is_nary(2) is False) and (plat2.is_nary(4) is True)

    cuau = bulk("CuAu", "zincblende", a=3.0)
    cuag = bulk("CuAg", "zincblende", a=3.0)

    plat3 = ParentLattice(atoms=cuau, substitutions=[cuag])
    isok3 = (plat3.is_nary(2) is True) and (plat3.is_nary(4) is False)

    cuau = bulk("CuAu", "zincblende", a=3.0)
    cuag = bulk("CuAg", "zincblende", a=3.0)
    siau = bulk("SiAu", "zincblende", a=3.0)

    plat4 = ParentLattice(atoms=cuau, substitutions=[cuag, siau])
    isok4 = (
        (plat4.is_nary(2) is False)
        and (plat4.is_nary(3) is False)
        and (plat4.is_nary(4) is False)
    )

    assert isok1 and isok2 and isok3 and isok4
