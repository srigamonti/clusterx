# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import time
from functools import partial

import pytest
from ase import Atoms
import numpy as np

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator, site_basis_function
from clusterx.utils import PolynomialBasis
from clusterx.cli.build_ccalc import build_ccalc


def scalar_product_basis_set(
    function1, function2, alpha1, alpha2, M=3, symmetric=False, scaled=True
):
    """
    Function to test basis single site basis functions regarding their orthogonality.
    Expects that sigma in {0,1,...,M-1}, where sigma is an ising type discrete spin varaible and M is the number of species in an alloy.
    """
    scaling = 1 / M if scaled else 1
    if symmetric:
        sigmas = [x for x in range(-int(M / 2), int(M / 2) + 1)]
        if M % 2 == 0:
            sigmas.remove(0)
    else:
        sigmas = [x for x in range(M)]
    scalar_product = 0
    for sigma in sigmas:
        scalar_product += function1(alpha1, sigma, M) * function2(alpha2, sigma, M)
    scalar_product = scalar_product * scaling
    return scalar_product


def print_orthonormality(function, m=3, symmetric=True):
    print("\northonormality:\nj k <theta_j | theta_k>")
    for j in range(m):
        for k in range(m):
            print(
                j,
                k,
                round(
                    scalar_product_basis_set(
                        function, function, j, k, M=m, symmetric=symmetric
                    ),
                    10,
                ),
            )


def test_all_verbose():
    """Test calculation of cluster correlations.

    After successful execution of the test, the generated structure and clusters pool may be visualized with the command::

        ase gui test_cluster_correlations_structure_#.json
        ase gui test_cluster_correlations_cpool.json

    """
    cell = [[3, 0, 0], [0, 1, 0], [0, 0, 5]]

    positions = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]

    pbc = [True, True, False]

    pri = Atoms(["H", "H", "H"], positions=positions, cell=cell, pbc=pbc)
    su1 = Atoms(["C", "H", "H"], positions=positions, cell=cell, pbc=pbc)
    su2 = Atoms(["H", "He", "H"], positions=positions, cell=cell, pbc=pbc)
    su3 = Atoms(["H", "N", "H"], positions=positions, cell=cell, pbc=pbc)

    plat = ParentLattice(pri, substitutions=[su1, su2, su3], pbc=pbc)
    cpool = ClustersPool(plat, npoints=[1, 2], radii=[0, 1.2])
    corrcal_tri = CorrelationsCalculator("trigonometric", plat, cpool)
    corrcal_poly = CorrelationsCalculator("polynomial", plat, cpool)

    scell1 = SuperCell(plat, np.array([(1, 0, 0), (0, 3, 0), (0, 0, 1)]))
    # structure 1
    # x:-, y:|
    # H  H   H     1 1 1
    # C  N   H     6 7 1
    # H  He  H     1 2 1
    structure1 = Structure(scell1, [1, 1, 1, 6, 7, 1, 1, 2, 1])
    corrs1_tri = corrcal_tri.get_cluster_correlations(structure1)
    assert np.allclose(
        [
            -0.33333333,
            0.0,
            -0.0,
            0.33333333,
            0.57735027,
            -0.33333333,
            -0.25,
            -0.0,
            -0.25,
        ],
        corrs1_tri,
        atol=1e-5,
    )
    corrs1_poly = corrcal_poly.get_cluster_correlations(structure1)

    # Doubling of structure1. Correlations should not change.
    scell2 = SuperCell(plat, np.array([(1, 0, 0), (0, 6, 0), (0, 0, 1)]))
    structure2 = Structure(
        scell2, [1, 1, 1, 6, 7, 1, 1, 2, 1, 1, 1, 1, 6, 7, 1, 1, 2, 1]
    )
    corrs2_tri = corrcal_tri.get_cluster_correlations(structure2)
    assert np.allclose(
        [
            -0.33333333,
            0.0,
            -0.0,
            0.33333333,
            0.57735027,
            -0.33333333,
            -0.25,
            -0.0,
            -0.25,
        ],
        corrs2_tri,
        atol=1e-5,
    )
    corrs2_poly = corrcal_poly.get_cluster_correlations(structure2)
    assert np.allclose(corrs1_poly, corrs2_poly, atol=1e-5)
    assert np.allclose(
        [-0.33333333, 0.0, 0.0, 0.81649658, 0.47140452, -0.33333333, -0.5, -0.0, -0.5],
        corrs2_poly,
        atol=1e-5,
    )

    t = time.time()
    # TODO: should this have 0.5 values?
    fun = partial(
        site_basis_function,
        basis_name=corrcal_tri.basis_name,
        basis_set=PolynomialBasis(symmetric=False),
    )
    print_orthonormality(fun, symmetric=False)
    print("Time for trigonometric basis", time.time() - t)
    t = time.time()
    fun = partial(
        site_basis_function,
        basis_name=corrcal_poly.basis_name,
        basis_set=PolynomialBasis(symmetric=True),
    )
    print_orthonormality(fun, symmetric=True)
    print("Time for polynomial basis", time.time() - t)

    print("\nPolynomial basis functions (m=3):")
    PolynomialBasis(symmetric=True).print_basis_functions(3)

    print("\n\n========Test writes========")
    print(test_all_verbose.__doc__)
    scell = cpool.get_cpool_scell()
    cpool.write_clusters_db(
        cpool.get_cpool(), scell, "test_cluster_correlations_cpool.json"
    )

    structure1.serialize(
        fmt="json", filepath="test_cluster_correlations_structure_1.json"
    )
    structure2.serialize(
        fmt="json", filepath="test_cluster_correlations_structure_2.json"
    )


@pytest.fixture
def cell():
    return [[3, 0, 0], [0, 1, 0], [0, 0, 5]]


@pytest.fixture
def pbc():
    return [True] * 3


@pytest.fixture
def positions():
    return [[0, 0, 0], [1, 0, 0], [2, 0, 0]]


@pytest.fixture
def primitive_lattice(positions, cell, pbc):
    return Atoms(["H", "H", "H"], positions=positions, cell=cell, pbc=pbc)


@pytest.fixture
def sub(positions, cell, pbc):
    return Atoms(["C", "H", "H"], positions=positions, cell=cell, pbc=pbc)


@pytest.fixture
def plat_quaternary(primitive_lattice, positions, cell, pbc):
    su1 = Atoms(["C", "H", "H"], positions=positions, cell=cell, pbc=pbc)
    su2 = Atoms(["H", "He", "H"], positions=positions, cell=cell, pbc=pbc)
    su3 = Atoms(["H", "N", "H"], positions=positions, cell=cell, pbc=pbc)
    return ParentLattice(primitive_lattice, substitutions=[su1, su2, su3], pbc=pbc)


@pytest.fixture
def cpool(plat_quaternary):
    return ClustersPool(plat_quaternary, npoints=[1, 2], radii=[0, 1.2])


@pytest.fixture
def scell131(plat_quaternary):
    return SuperCell(plat_quaternary, np.array([(1, 0, 0), (0, 3, 0), (0, 0, 1)]))


@pytest.fixture
def ccalc(plat_quaternary, cpool):
    return CorrelationsCalculator("trigonometric", plat_quaternary, cpool)


@pytest.fixture
def structure131(scell131):
    return Structure(scell131, [1, 1, 1, 6, 7, 1, 1, 2, 1])


def test_cli(plat_quaternary, cpool, structure131):
    """Test CLI build_ccalc command"""
    plat_quaternary.serialize(filepath="ccalc_test_plat.json")
    cpool.serialize(filepath="ccalc_test_cpool.json")
    build_ccalc(
        basis_name="trigonometric",
        plat_filepath="ccalc_test_plat.json",
        cpool_filepath="ccalc_test_cpool.json",
        ccalc_filepath="ccalc_test_ccalc.pickle",
    )
    ccalc = CorrelationsCalculator("ccalc_test_ccalc.pickle")
    corrs = ccalc.get_cluster_correlations(structure131)


def test_serialize_load(ccalc):
    ccalc.serialize(filepath="CCALC.pickle", fmt="pickle")
    ccalc_loaded = CorrelationsCalculator("CCALC.pickle")


def test_binary_linear_basis(primitive_lattice, sub, scell131):
    bin_plat = ParentLattice(primitive_lattice, substitutions=[sub], pbc=[True] * 3)
    bin_cpool = ClustersPool(bin_plat, npoints=[1, 2], radii=[0, 1.2])
    coorrcal_bin_lin = CorrelationsCalculator("binary-linear", bin_plat, bin_cpool)

    scell_bin = SuperCell(bin_plat, np.array([(1, 0, 0), (0, 3, 0), (0, 0, 1)]))
    structure_bin = Structure(scell131, [1, 1, 1, 6, 1, 1, 6, 1, 1])
    corrs_bin = coorrcal_bin_lin.get_cluster_correlations(structure_bin)
    np.testing.assert_allclose(corrs_bin, [0.6666666, 0.33333333], atol=1e-5)

    bin_cpool.write_clusters_db(
        bin_cpool.get_cpool(), scell_bin, "test_cluster_correlations_cpool_bin.json"
    )
    structure_bin.serialize(
        fmt="json", filepath="test_cluster_correlations_structure_bin.json"
    )


@pytest.mark.parametrize(
    "basis_name", ["binary-linear", "trigonometric", "polynomial", "chebyshev"]
)
def test_cell_doubling(basis_name, plat_quaternary, cpool):
    corrcal = CorrelationsCalculator(basis_name, plat_quaternary, cpool)

    scell = SuperCell(plat_quaternary, np.array([(1, 0, 0), (0, 3, 0), (0, 0, 1)]))
    structure = Structure(scell, [1, 1, 1, 6, 7, 1, 1, 2, 1])
    corrs1 = corrcal.get_cluster_correlations(structure)

    scell = SuperCell(plat_quaternary, np.array([(1, 0, 0), (0, 6, 0), (0, 0, 1)]))
    structure = Structure(scell, [1, 1, 1, 6, 7, 1, 1, 2, 1, 1, 1, 1, 6, 7, 1, 1, 2, 1])
    corrs2 = corrcal.get_cluster_correlations(structure)
    np.testing.assert_allclose(corrs1, corrs2, atol=1e-5)


@pytest.mark.parametrize(
    "basis_name,expected",
    [
        (
            "trigonometric",
            [-0.33333333, 0, 0, 0.33333333, 0.57735027, -0.33333333, -0.25, 0, -0.25],
        ),
        (
            "polynomial",
            [-0.33333333, 0, 0, 0.81649658, 0.47140452, -0.33333333, -0.5, 0, -0.5],
        ),
    ],
)
def test_basis(plat_quaternary, cpool, scell131, basis_name, expected):
    corrcal = CorrelationsCalculator(basis_name, plat_quaternary, cpool)
    structure = Structure(scell131, [1, 1, 1, 6, 7, 1, 1, 2, 1])
    corrs = corrcal.get_cluster_correlations(structure)
    np.testing.assert_allclose(corrs, expected, atol=1e-5)
