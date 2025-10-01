# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit
# https://www.apache.org/licenses/LICENSE-2.0.txt.

import pytest
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers as an
from ase.io import write
from ase.build import bulk, fcc111, add_adsorbate

from clusterx.parent_lattice import ParentLattice
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.super_cell import SuperCell
from clusterx.test.defaults import get_clathrate_plat
from clusterx.cli.build_cpool import build_cpool


@pytest.fixture
def plat():
    cell = [[3, 0, 0], [0, 1, 0], [0, 0, 5]]
    positions = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    pbc = [True, True, False]
    pri = Atoms(["H", "H", "H"], positions=positions, cell=cell, pbc=pbc)
    su1 = Atoms(["C", "H", "H"], positions=positions, cell=cell, pbc=pbc)
    su2 = Atoms(["H", "He", "H"], positions=positions, cell=cell, pbc=pbc)
    su3 = Atoms(["H", "N", "H"], positions=positions, cell=cell, pbc=pbc)
    return ParentLattice(pri, substitutions=[su1, su2, su3], pbc=pbc)


@pytest.fixture
def npoints():
    return [1, 2, 3]


@pytest.fixture
def radii():
    return [0, 2.1, 2.1]


@pytest.fixture
def cpool(plat, npoints, radii):
    return ClustersPool(plat, npoints=npoints, radii=radii)


@pytest.mark.parametrize("method", [0, 1])
def test_cli(plat, npoints, radii, method):
    filepath_plat = "test_clusters_pool_plat.json"
    plat.serialize(filepath_plat)
    build_cpool(
        npoints=npoints,  # List[int] or str
        radii=radii,  # List[float] or str
        sset_filepath=None,  # Optional[str]
        plat_filepath=filepath_plat,  # Optional[str]
        psc=2,  # Supercell definition
        method=method,  # int
        cpool_filepath="cpool_test_cli.json",  # str
        vacancy_atomic_number=0,  # int
    )


def test_serialize_load(cpool):
    cpool.serialize(filepath="cpool.json")
    _ = ClustersPool(filepath="cpool.json")


def test_0_point_raises(plat, npoints, radii):
    """Test that ClustersPool raises an error when npoints is set to 0."""
    with pytest.raises(ValueError):
        ClustersPool(plat, npoints=npoints + [0], radii=radii + [0])


@pytest.mark.xfail(raises=AssertionError, reason="Methods are known to currently produce different clusters.")
def test_methods(plat, npoints, radii):
    cp_list = []
    for method in [0, 1]:
        cp = ClustersPool(plat, npoints=npoints, radii=radii, method=method)
        cp_list.append(cp.get_cpool_list())
    for cp in cp_list:
        assert len(cp) > 0
        cp.sort()
    for cp0, cp1 in zip(cp_list[0], cp_list[1]):
        assert cp0 == cp1


def test_2D_radii(plat, npoints, radii):
    cp = ClustersPool(plat, npoints=npoints, radii=radii)
    cp.write_clusters_db(db_name="test_clusters_generation_1.json")

    mult = cp.get_multiplicities()
    npoints = cp.get_all_npoints()
    radii = cp.get_all_radii()

    mult_ref = np.array(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            1,
            2,
            2,
            1,
            1,
            2,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            2,
            2,
            2,
            1,
            2,
            1,
            2,
            1,
            1,
        ]
    )
    npoints_ref = np.array(
        [
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
        ]
    )
    radii_ref = np.array(
        [
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.41421356,
            1.41421356,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            1.41421356,
            1.41421356,
            1.41421356,
            1.41421356,
            1.41421356,
            1.41421356,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]
    )
    np.testing.assert_array_equal(mult, mult_ref)
    np.testing.assert_array_equal(npoints, npoints_ref)
    np.testing.assert_array_almost_equal(radii, radii_ref)


def test_2D_supercell():
    a = 3.0
    cell = np.array([[1, 0, 0], [0, 4, 0], [0, 0, 1]])
    positions = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]])
    sites = [[12, 13], [12, 13], [12, 13], [12, 13]]
    pri = Atoms(cell=cell * a, positions=positions * a)

    pl = ParentLattice(pri, sites=sites, pbc=(1, 0, 0))
    sc = SuperCell(pl, [[4, 0, 0], [0, 1, 0], [0, 0, 1]])
    # Here radii are not given, therefore all the clusters which can fit in the
    # supercell are generated.
    cp = ClustersPool(pl, npoints=[1, 2], super_cell=sc)
    cp.write_clusters_db(db_name="test_clusters_generation_2.json")

    mult = cp.get_multiplicities()
    npoints = cp.get_all_npoints()
    radii = cp.get_all_radii()

    mult_ref = np.array([2, 2, 2, 1, 2, 2, 4, 2, 2, 2, 2, 4, 4, 2, 4, 1, 2, 2])
    npoints_ref = np.array([1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    radii_ref = np.array(
        [
            0.0,
            0.0,
            3.0,
            3.0,
            3.0,
            3.0,
            4.242641,
            4.242641,
            6.0,
            6.0,
            6.0,
            6.708204,
            6.708204,
            6.708204,
            8.485281,
            9.0,
            9.486833,
            10.816654,
        ]
    )

    np.testing.assert_array_equal(mult, mult_ref)
    np.testing.assert_array_equal(npoints, npoints_ref)
    np.testing.assert_array_almost_equal(radii, radii_ref)


def test_fcc_radii():
    a = 4.1
    pris = bulk("Cu", crystalstructure="fcc", a=a)
    sites = [[an["Cu"], an["Au"]]]
    pl = ParentLattice(pris, sites=sites)

    cp = ClustersPool(pl, npoints=[1, 2, 3], radii=[0, 5.0, 5.0])
    cp.write_clusters_db(db_name="test_clusters_generation_3.json")

    mult = cp.get_multiplicities()
    npoints = cp.get_all_npoints()
    radii = cp.get_all_radii()

    mult_ref = np.array([1, 6, 12, 8, 48])
    npoints_ref = np.array([1, 2, 2, 3, 3])
    radii_ref = np.array([0.0, 2.8991378, 4.1, 2.8991378, 4.1])

    np.testing.assert_array_equal(mult, mult_ref)
    np.testing.assert_array_equal(npoints, npoints_ref)
    np.testing.assert_array_almost_equal(radii, radii_ref)


def test_2D_acute_angle():
    a = 4.1
    cell = np.array([[5, 0, 0], [4, 1, 0], [0, 0, 1]])
    positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
    pbc = (1, 1, 0)
    sites = [[an["Cu"], an["Au"]], [an["Cu"], an["Au"]], [12], [12], [12]]
    pris = Atoms(cell=cell * a, positions=positions * a, pbc=pbc)
    pl = ParentLattice(pris, sites=sites, pbc=pbc)

    cp = ClustersPool(pl, npoints=[2], radii=[2.0 * a])
    atoms = cp.get_cpool_scell().get_pristine_structure().get_atoms()
    write(
        filename="test_clusters_generation_scell_part4.json",
        images=atoms,
        format="json",
    )
    cp.write_clusters_db(db_name="test_clusters_generation_4.json")
    cp.get_multiplicities()

    atom_idxs, atom_nrs = cp.get_cpool_arrays()
    a = an["Au"]
    np.testing.assert_array_equal(atom_idxs, np.array([[0, 1], [5, 20]]))
    np.testing.assert_array_equal(atom_nrs, np.array([[a, a], [a, a]]))


def test_pt_111():
    cell = np.array(
        [
            [2.785200119, 0.0, 0.0],
            [-1.3926000595, 2.4120540577, 0.0],
            [0.0, 0.0, 7.2740998268],
        ]
    )
    positions = np.array(
        [
            [0.0, 0.0, 2.274101],
            [1.392600, 0.804018, 0.0],
            [0.0, 0.0, 2.916914],
            [1.392600, 0.804018, 2.924188],
            [0.0, 1.608036, 2.931462],
        ]
    )
    pbc = (1, 1, 0)
    sites = [[78], [78], [0, 8], [0, 8], [0, 8]]
    pris = Atoms(cell=cell, positions=positions, pbc=pbc)
    pl = ParentLattice(pris, sites=sites, pbc=pbc)

    cp = ClustersPool(pl, npoints=[1, 2, 3], radii=[0, 3.3, 3.0])
    cp.write_clusters_db(db_name="test_clusters_generation_5.json")

    mult = cp.get_multiplicities()
    radii = cp.get_all_radii()
    npoints = cp.get_all_npoints()

    mult_ref = np.array(
        [1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 6, 6, 3, 1, 1, 6, 1, 6, 6, 6]
    )
    npoints_ref = np.array(
        [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    )
    radii_ref = np.array(
        [
            0.0,
            0.0,
            0.0,
            1.60805243,
            1.60805243,
            1.60810181,
            2.78520012,
            2.78520012,
            2.78520012,
            3.21608028,
            3.21608028,
            3.21610496,
            1.60810181,
            2.78520012,
            2.78520012,
            2.78520012,
            2.78520012,
            2.78520012,
            2.78520012,
            2.78520012,
            2.78520012,
            2.78520012,
            2.78520012,
            2.78520012,
            2.78520012,
        ]
    )

    np.testing.assert_array_equal(mult, mult_ref)
    np.testing.assert_array_equal(npoints, npoints_ref)
    np.testing.assert_array_almost_equal(radii, radii_ref)


def test_clathrate():
    plat = get_clathrate_plat()

    # cp = ClustersPool(plat,npoints=[1,2],radii=[0,5.7],super_cell=SuperCell(plat,np.diag([2,1,1])))
    cp = ClustersPool(plat, npoints=[1, 2], radii=[0, 5.0])
    cp.write_clusters_db(db_name="test_clusters_generation_6.json")

    mult = cp.get_multiplicities()
    radii = cp.get_all_radii()
    npoints = cp.get_all_npoints()

    mult_ref = np.array([24, 16, 6, 8, 48, 12, 24, 24, 48, 48, 48, 48, 48, 12, 24])
    npoints_ref = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    radii_ref = np.array(
        [
            0.0,
            0.0,
            0.0,
            2.367582,
            2.424043,
            2.439434,
            2.496507,
            3.890476,
            3.901473,
            3.919454,
            3.920018,
            3.951327,
            4.054079,
            4.121802,
            4.365504,
        ]
    )

    np.testing.assert_array_equal(mult, mult_ref)
    np.testing.assert_array_equal(npoints, npoints_ref)
    np.testing.assert_array_almost_equal(radii, radii_ref)


def test_negative_radii():
    plat = ParentLattice(
        Atoms(cell=np.diag([2, 2, 5]), positions=[[0, 0, 0]]),
        site_symbols=[["Cu", "Al"]],
        pbc=(1, 1, 0),
    )

    scell = SuperCell(plat, np.array([(6, 0, 0), (0, 6, 0), (0, 0, 1)]))
    cp = ClustersPool(
        plat, npoints=[1, 2, 3, 4], radii=[0, -1, 4.1, 2.9], super_cell=scell
    )
    cp.write_clusters_db(db_name="test_clusters_generation_7.json")

    mult = cp.get_multiplicities()
    radii = cp.get_all_radii()
    npoints = cp.get_all_npoints()

    mult_ref = np.array([1, 2, 2, 2, 4, 2, 2, 4, 4, 2, 4, 2, 4, 1])
    npoints_ref = np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4])
    radii_ref = np.array(
        [
            0.0,
            2.0,
            2.82842712,
            4.0,
            4.47213595,
            5.65685425,
            6.0,
            6.32455532,
            7.21110255,
            8.48528137,
            2.82842712,
            4.0,
            4.0,
            2.82842712,
        ]
    )

    np.testing.assert_array_equal(mult, mult_ref)
    np.testing.assert_array_equal(npoints, npoints_ref)
    np.testing.assert_array_almost_equal(radii, radii_ref)


def test_fcc_111():
    """FCC(111) with alloying and Adsorption in hollow sites"""
    pristine = fcc111("Re", size=(1, 1, 3), a=3.2)  # 3-atomic-layer Al slab
    add_adsorbate(pristine, "X", 1.7, position="fcc")  # Hollow FCC vacancy site
    pristine.center(vacuum=10.0, axis=2)  # add vacuum along z-axis

    symbols = [["Co"], ["Co"], ["Co", "Ni"], ["X", "Al"]]
    plat = ParentLattice(pristine, symbols=symbols)

    scell = SuperCell(plat, [[5, 0], [0, 2]])
    npoints = [2]
    radii = [-1]
    cp = ClustersPool(plat, npoints=npoints, radii=radii, super_cell=scell, method=0)
    cp.write_clusters_db(db_name="test_clusters_generation_8.json")

    cp.get_multiplicities()
    cp.get_all_radii()
    cp.get_all_npoints()
