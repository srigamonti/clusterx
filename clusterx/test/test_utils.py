"""Test utility functions for CELL."""

import pytest
import numpy as np
from ase.build import bulk

from clusterx.parent_lattice import ParentLattice
from clusterx.utils import (
    Exponential,
    PolynomialFunction,
    PolynomialBasis,
    lattice_wrap_index,
    grid_mapping,
    normalize_shape_input,
    parent_lattice_to_atat,
    atat_to_cell,
)


@pytest.fixture
def parent_lattice():
    pri = bulk("H", crystalstructure="sc", a=1.0)
    sub = bulk("He", crystalstructure="sc", a=1.0)
    return ParentLattice(atoms=pri, substitutions=[sub], pbc=(1, 1, 1))


@pytest.mark.xfail(raises=ValueError, reason="Perhaps wrong usage of atat")
def test_atat(parent_lattice, tmp_path):
    parent_lattice_to_atat(parent_lattice, out_fname=tmp_path / "lat.in")
    parent_lattice_loaded = atat_to_cell(tmp_path / "lat.in")
    assert parent_lattice == parent_lattice_loaded


@pytest.mark.parametrize(
    "grid_shape,i_grid,p_reduced",
    [
        ((10, 10, 10, 2), np.array([0, 0, 0, 0]), (3, 3, 3)),
        ((10, 10, 10, 2), np.array([3, 3, 3, 0]), (4, 4, 4)),
        ((10, 10, 10, 3), np.array([9, 9, 9, 2]), (3, 3, 3)),
        ((10, 10, 1), np.array([9, 9, 0]), (3, 3)),
    ],
)
def test_grid_mapping(grid_shape, i_grid, p_reduced):
    grid = np.arange(np.prod(grid_shape)).reshape(grid_shape)
    grid_reduced, i_new = grid_mapping(grid, i_grid, p_reduced)
    np.testing.assert_array_equal(
        grid_reduced.shape, list(p_reduced) + [grid_shape[-1]]
    )
    assert np.all(np.array(p_reduced) - i_new[:-1] > 0)
    assert np.all(i_new[-1] >= 0)
    val = grid[(*i_grid,)]
    val_reduced = grid_reduced[(*i_new,)]
    assert int(val) == int(val_reduced)


@pytest.mark.parametrize(
    "shape,output",
    [
        (2, np.diag([2, 2, 2])),
        ([2, 3, 4], np.diag([2, 3, 4])),
        (
            [[1, -1, 0], [0, 1, -1], [0, 0, 2]],
            [[1, -1, 0], [0, 1, -1], [0, 0, 2]],
        ),
        (
            np.array([[1, -1, 0], [0, 1, -1], [0, 0, 2]]),
            np.array([[1, -1, 0], [0, 1, -1], [0, 0, 2]]),
        ),
        (
            (2, 3, 4),
            np.diag([2, 3, 4]),
        ),
    ],
)
def test_normalize_shape_input(shape, output):
    np.testing.assert_array_equal(normalize_shape_input(shape), output)


def test_lattice_wrap_index():
    """See https://stackoverflow.com/questions/38066785/np-ndarray-with-periodic-boundary-conditions"""
    arr = np.array(
        [
            [11.0, 12.0, 13.0, 14.0],
            [21.0, 22.0, 23.0, 24.0],
            [31.0, 32.0, 33.0, 34.0],
            [41.0, 42.0, 43.0, 44.0],
        ]
    )
    test_vals = [
        [(1, 1), 22.0],  # no wrapping
        [(3, 3), 44.0],  # no wrapping, last element
        [(4, 4), 11.0],  # single wrapping on diagonal
        [(3, 4), 41.0],  # single wrapping off diagonal
        [(4, 3), 14.0],
        [[4, 3], 14.0],  # indexing with list instead of tuple
        [(10, 10), 33.0],  # double wrapping
        [[slice(0, 10), 1], False],  # no slice indexing
        [1, False],  # no integer indexing
        [(1, 2, 3), False],  # wrong shape of index
    ]  # [index, expected value]
    for idx, value in test_vals:
        if isinstance(value, float):
            assert arr[lattice_wrap_index(idx, (4, 4))] == value
        else:
            with pytest.raises(ValueError):
                arr[lattice_wrap_index(idx, (4, 4))]


def test_exponential():
    coefficient = 2.0
    exponent = 3.0
    exp = Exponential(exponent, coefficient)
    x = np.array([1.0, 2.0, 3.0])
    exp.divide_scalar(2.0)
    exp.multiply_scalar(2.0)

    expected = coefficient * (x**exponent)
    np.testing.assert_allclose(exp.evaluate(x), expected)


def test_polynomial_function():
    poly = PolynomialFunction()
    poly.add_exponential(order=2, coefficient=1.0)
    poly.add_exponential(order=3, coefficient=2e-16)
    assert len(poly.exponentials) == 2
    poly.clear_exponentials()
    assert len(poly.exponentials) == 1
    poly.add_exponential(order=1, coefficient=3.0)
    assert poly.evaluate(2.0) == 2.0**2 + 3.0 * 2.0
    poly.multiply_scalar(2.0)
    assert poly.evaluate(2.0) == (2.0**2 + 3.0 * 2.0) * 2.0
    poly.print_polynomial()


@pytest.mark.parametrize("max_order", [2, 10])
@pytest.mark.parametrize("symmetric", [False, True])
def test_polynomial_basis(max_order, symmetric):
    basis = PolynomialBasis(max_order=max_order, symmetric=symmetric)
    basis.construct(m=max_order + 2)
    basis.evaluate(alpha=1, sigma=1, m=2)
    basis.print_basis_functions(m=2)
