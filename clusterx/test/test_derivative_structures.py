"""Test the derivative structures module."""

import pytest
from ase import Atoms
import numpy as np
import pandas as pd

from clusterx.derivative_structures import DSGenerator
from clusterx.parent_lattice import ParentLattice
from clusterx.calculators.emt import EMT2


@pytest.fixture
def parent_lattice():
    a = 3.62/np.sqrt(2.0)
    positions = [(0,0,0), (a/2,a/2,a/2)]
    cell = [(a,0,0),(0,a,0),(0,0,a)]
    pbc = (True,True,True)
    pri = Atoms('Cu2', positions=positions, cell= cell, pbc= pbc)
    sub = Atoms('Al2', positions=positions, cell= cell, pbc= pbc)

    plat = ParentLattice(pri, substitutions=[sub], pbc=pbc)
    return plat


@pytest.fixture
def ds_generator_empty(parent_lattice):
    return DSGenerator(parent_lattice)


@pytest.fixture
def ds_generator_full(ds_generator_empty):
    ds_generator_empty.generate(
        supercell_sizes=[1, 2],
        num_subs_list=[[1], [2]],
        sc_shape=None,
        sc_shapes=None,
        n_random=None,
        random_state=42
    )
    return ds_generator_empty


@pytest.mark.parametrize("n_random", [0, 1, 4, None])
@pytest.mark.parametrize(
    "supercell_sizes,num_subs_list,sc_shape,sc_shapes",
    [
        ([1, 2], [[1], [2]], None, None),
        (None, [[1]], np.diag([2, 2, 2]), None),
        (None, [[1], [2]], None, [np.diag([2, 2, 2]), np.diag([3, 3, 3])]),
    ]
)
def test_generate(
    supercell_sizes,
    num_subs_list,
    sc_shape,
    sc_shapes,
    n_random,
    ds_generator_empty
):
    ds_generator_empty.generate(
        supercell_sizes=supercell_sizes,
        num_subs_list=num_subs_list,
        sc_shape=sc_shape,
        sc_shapes=sc_shapes,
        n_random=1,
        random_state=42
    )


def test_compute_emt(ds_generator_full):
    calc = EMT2()
    ds_generator_full.compute_properties(
        property_name="energy_emt",
        ase_calculator=calc,
        cemodel=None,
        property_solver=None,
        property_solver_kwargs={},
        linear_reference=None,
        per_formula_unit=False)
    assert "energy_emt" in ds_generator_full.configurations.columns


@pytest.fixture
def solver_single_properties_dummy():
    def _solver(*args, **kwargs):
        return 1.
    return _solver


@pytest.fixture
def solver_single_properties_rand():
    def _solver(*args, **kwargs):
        return np.random.rand()
    return _solver


@pytest.mark.parametrize(
    "lin_ref",
    [
        None,
        'least-squares',
        'concentration-endpoints',
        [[0., 1.], [1., 2.]],
    ]
)
def test_compute_properties_single(
    ds_generator_full, solver_single_properties_rand, lin_ref):
    ds_generator_full.compute_properties(
        property_name="dummy",
        ase_calculator=None,
        cemodel=None,
        property_solver=solver_single_properties_rand,
        property_solver_kwargs={},
        linear_reference=lin_ref,
        per_formula_unit=False)
    assert "dummy" in ds_generator_full.configurations.columns


@pytest.fixture
def solver_multi_properties_dummy():
    def _solver(*args, **kwargs):
        return 1., 2. # Return multiple dummy properties
    return _solver


@pytest.fixture
def solver_multi_properties_rand():
    def _solver(*args, **kwargs):
        return np.random.rand(), np.random.rand() # Return multiple dummy properties
    return _solver


@pytest.mark.parametrize(
    "lin_ref",
    [
        None,
        'least-squares',
        'concentration-endpoints',
        [[0., 1.], [1., 2.]],
    ]
)
def test_compute_properties_multi(
    ds_generator_full, solver_multi_properties_rand, lin_ref):
    ds_generator_full.compute_properties(
        property_names=["dummy1", "dummy2"],
        ase_calculator=None,
        cemodel=None,
        property_solver=solver_multi_properties_rand,
        property_solver_kwargs={},
        linear_reference=lin_ref,
        per_formula_unit=False)
    assert "dummy1" in ds_generator_full.configurations.columns
    assert "dummy2" in ds_generator_full.configurations.columns


@pytest.mark.xfail(reason="Exception not yet implemented")
def test_compute_properties_raises(ds_generator_full):
    with pytest.raises(ValueError):
        ds_generator_full.compute_properties(
            property_name="dummy",
            ase_calculator=None,
            cemodel=None,
            property_solver=None,
            property_solver_kwargs=None,
            linear_reference=None,
            per_formula_unit=False)


def test_compute_properties_conflict(ds_generator_full, solver_single_properties_dummy):
    with pytest.raises(ValueError):
        ds_generator_full.compute_properties(
            property_name="dummy",
            property_names=["dummy1", "dummy2"],
            ase_calculator=None,
            cemodel=None,
            property_solver=solver_single_properties_dummy,
            property_solver_kwargs={},
            linear_reference=None,
            per_formula_unit=False)


def test_serialize_load(ds_generator_full, tmp_path):
    filepath = tmp_path / "test_ds_generator.json"
    ds_generator_full.serialize(filepath)
    ds_generator_loaded = DSGenerator.from_file(filepath)
    assert ds_generator_full.plat == ds_generator_loaded.plat
    pd.testing.assert_frame_equal(
        ds_generator_full.scell_shapes, ds_generator_loaded.scell_shapes)
    pd.testing.assert_frame_equal(
        ds_generator_full.configurations, ds_generator_loaded.configurations)


def test_concentration(ds_generator_full):
    assert "frconc_binary" not in ds_generator_full.configurations.columns
    ds_generator_full.add_fractional_concentration_binary()
    assert "frconc_binary" in ds_generator_full.configurations.columns
