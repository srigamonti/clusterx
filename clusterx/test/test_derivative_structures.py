"""Test the derivative structures module."""

import pytest
from ase import Atoms
from ase.data import atomic_numbers as an
import numpy as np
import pandas as pd

from clusterx.derivative_structures import (
    DSGenerator, get_HNFs, get_unique_supercells)
from clusterx.parent_lattice import ParentLattice
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
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

def test_get_all_hnf():
    nrs = []
    for n in range(1,7):
        hnfs = get_HNFs(n)
        nrs.append(len(hnfs))

    assert (np.array(nrs) == np.array([1,7,13,35,31,91])).all()

    hnfs = get_HNFs(4,pbc=(1,1,0))

    hnfs_ref = [
                [[1, 0, 0],
                 [0, 4, 0],
                 [0, 0, 1]],
                [[1, 1, 0],
                 [0, 4, 0],
                 [0, 0, 1]],
                [[1, 2, 0],
                 [0, 4, 0],
                 [0, 0, 1]],
                [[1, 3, 0],
                 [0, 4, 0],
                 [0, 0, 1]],
                [[2, 0, 0],
                 [0, 2, 0],
                 [0, 0, 1]],
                [[2, 1, 0],
                 [0, 2, 0],
                 [0, 0, 1]],
                [[4, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]]
                ]

    isok = True

    # Now check that hnfs_ref is identical to hnfs
    for m1,m2 in zip(hnfs,hnfs_ref):
        for r1,r2 in zip(m1,m2):
            for i1,i2 in zip(r1,r2):
                if i1 != i2:
                    isok = False
                    break
            if not isok:
                break
        if not isok:
            break

    assert isok


def test_get_unique_supercells():
    """Test generation of unique supercells.

    Three cases are tested: Square 2D lattice of index 4 (the case of
    Fig.1 of [1]); FCC lattice of index 4 (cf. the seven non-decorated
    structures of Fig. 2 and Table IV in [2]); and the simple cubic lattice
    (Fig.11 and Table IV of [2])

    [1] Computational Materials Science 59 (2012) 101–107
    [2] Phys. Rev. B 77, 224115 2008
    """
    for case in range(3):

        if case == 0: #Square (2D, i.e. pbc = (1,1,0))
            a=3.1
            index = 4
            cell = np.array([[1,0,0],[0,1,0],[0,0,1]])
            positions = np.array([[0,0,0]])
            sites = [[12,13]]
            pris = Atoms(cell=cell*a, positions=positions*a)

            pl = ParentLattice(pris, sites=sites, pbc=(1,1,0))

            unique_scs, unique_trafos = get_unique_supercells(index,pl)

            sset = StructuresSet(pl)
            for t in unique_trafos:
                scell = SuperCell(pl,t)
                sset.add_structure(Structure(scell,scell.get_atomic_numbers()),write_to_db = True)

            sset.serialize(filepath="test_get_unique_supercells-square_lattice.json", overwrite=True)
            print("\nFound ",len(unique_scs), " unique HNFs for a 2D square lattice of index ",index)
            #print("SCS: ", unique_scs)
            #print("TRA: ", unique_trafos)
            isok0 = len(unique_scs) == 4 and unique_scs[1][1][1] == 12.4

        if case == 1: #FCC
            a=3
            index = 4
            cell = np.array([[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]])
            positions = np.array([[0,0,0]])
            sites = [[an["Cu"],an["Au"]]]
            pris_fcc = Atoms(cell=cell*a,positions=positions*a,pbc=(1,1,1))

            pl = ParentLattice(pris_fcc,sites=sites)

            unique_scs, unique_trafos = get_unique_supercells(index,pl)

            sset = StructuresSet(pl)
            for t in unique_trafos:
                scell = SuperCell(pl,t)
                sset.add_structure(Structure(scell,scell.get_atomic_numbers()),write_to_db = True)

            sset.serialize(filepath="test_get_unique_supercells-fcc.json")
            print("Found ",len(unique_scs), " unique HNFs for a FCC lattice of index ",index)
            #print("SCS: ", unique_scs)
            #print("TRA: ", unique_trafos)
            isok1 = len(unique_scs) == 7 and unique_trafos[4][2][2] == 4

        if case == 2: #Simple cubic
            a=3.1
            index = 4
            cell = np.array([[1,0,0],[0,1,0],[0,0,1]])
            positions = np.array([[0,0,0]])
            sites = [[12,13]]
            pris = Atoms(cell=cell*a, positions=positions*a)

            pl = ParentLattice(pris, sites=sites, pbc=(1,1,1))

            unique_scs, unique_trafos = get_unique_supercells(index,pl)

            sset = StructuresSet(pl)
            for t in unique_trafos:
                scell = SuperCell(pl,t)
                sset.add_structure(Structure(scell,scell.get_atomic_numbers()),write_to_db = True)

            sset.serialize(filepath="test_get_unique_supercells-sc.json")
            print("Found ",len(unique_scs), " unique HNFs for a simple cubic lattice of index ",index)
            #print("SCS: ", unique_scs)
            #print("TRA: ", unique_trafos)
            isok2 = len(unique_scs) == 9 and unique_scs[4][2][2] == 12.4

    assert isok0 and isok1 and isok2