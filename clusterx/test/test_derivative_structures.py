"""Test the derivative structures module."""

import math
import pickle

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.data import atomic_numbers as an

from clusterx.calculators.emt import EMT2
from clusterx.cli.compute_comat import compute_comat
from clusterx.cli.compute_multiplicities import compute_multiplicities
from clusterx.cli.convert_dss_to_sset import convert_dss_to_sset
from clusterx.cli.convert_sset_to_dss import convert_sset_to_dss
from clusterx.derivative_structures import DSGenerator, DerivativeStructures, get_HNFs, get_unique_supercells
from clusterx.parent_lattice import ParentLattice
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
from clusterx.model import Model
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator


@pytest.fixture
def parent_lattice():
    a = 1.0
    positions = [(0, 0, 0), (a / 2, a / 2, a / 2)]
    cell = [(a, 0, 0), (0, a, 0), (0, 0, a)]
    pbc = (True, True, True)
    pri = Atoms("Cu2", positions=positions, cell=cell, pbc=pbc)
    sub = Atoms("Al2", positions=positions, cell=cell, pbc=pbc)

    plat = ParentLattice(pri, substitutions=[sub], pbc=pbc)
    return plat


@pytest.fixture
def plat_perov():
    """Perovskite multi-sublattice parent lattice"""
    cell = [6, 6, 6]
    positions = [
        [0, 0, 0],
        [3, 3, 3],
        [0, 3, 3],
        [3, 0, 3],
        [3, 3, 0],
    ]
    pbc = [True, True, True]
    pri = Atoms(["Cs", "Pb", "I", "I", "I"], positions=positions, cell=cell, pbc=pbc)
    sub_Sn = Atoms(["Cs", "Sn", "I", "I", "I"], positions=positions, cell=cell, pbc=pbc)
    sub_Br = Atoms(
        ["Cs", "Pb", "Br", "Br", "Br"], positions=positions, cell=cell, pbc=pbc
    )
    plat = ParentLattice(pri, substitutions=[sub_Sn, sub_Br], pbc=pbc)
    return plat


@pytest.fixture
def cpool(parent_lattice):
    # one-point clusters and nearest neighbor two-point clusters
    return ClustersPool(parent_lattice, npoints=[1, 2], radii=[0, 0.9])


@pytest.fixture
def corrc(parent_lattice, cpool):
    return CorrelationsCalculator("trigonometric", parent_lattice, cpool)


@pytest.fixture
def model_single(parent_lattice, cpool, corrc):
    # Ising-like model
    ecisE = [0.0, -1.0]
    multT = cpool.get_multiplicities()
    cemodel = Model(corrc, "energy", ecis=np.multiply(ecisE, multT))
    return cemodel


@pytest.fixture
def model_multi(parent_lattice, cpool, corrc):
    # Model for two properties in parallel
    ecis = np.array([[-1.0, -2.0], [-2.0, -4.0]])
    cemodel = Model(corrc, ["energy", "energy_double"], ecis=ecis)
    return cemodel


@pytest.fixture
def ds_generator_empty(parent_lattice):
    return DerivativeStructures(parent_lattice)


@pytest.fixture
def ds_generator_full(ds_generator_empty):
    ds_generator_empty.generate(
        supercell_sizes=[1, 2],
        num_subs_list=[[1], [2]],
        sc_shape=None,
        sc_shapes=None,
        n_random=None,
        random_state=42,
    )
    return ds_generator_empty


@pytest.mark.parametrize("model_", ["model_single", "model_multi"])
def test_cemodel_calc(ds_generator_full, model_, request):
    # prime example of using multiple fixtures with parametrize
    model = request.getfixturevalue(model_)
    dss = ds_generator_full
    prop = model.property_name
    if isinstance(prop, str):
        dss.compute_properties(property_name=prop, cemodel=model)
        assert prop in dss.configurations.columns
    else:
        dss.compute_properties(property_names=prop, cemodel=model)
        for prop_single in prop:
            assert prop_single in dss.configurations.columns


def test_generate_perov(plat_perov):
    ds_generator = DerivativeStructures(plat_perov)
    plat_perov.print_sublattice_types()
    p = [1, 1, 1]
    max_Br = np.prod(p) * 3
    max_Sn = np.prod(p)
    ds_generator.generate(
        num_subs_list=[{0: max_Br // 2, 2: max_Sn // 2}],
        sc_shape=np.diag(p),
        recursive=False,
    )
    assert ds_generator.configurations["degeneracy"].sum() > 0
    assert len(ds_generator.configurations) > 0


@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.parametrize("n_random", [2, None])
@pytest.mark.parametrize(
    "supercell_sizes,num_subs_list,sc_shape,sc_shapes,shapes_nearest_orthogonal",
    [
        ([1, 2], [[1], [2]], None, None, False),
        ([1, 2], [[1], [2]], None, None, True),
        (None, [[1]], np.diag([2, 2, 2]), None, False),
        (None, [[1], [2]], None, [np.diag([2, 2, 2]), np.diag([3, 3, 3])], False),
    ],
)
def test_generate(
    supercell_sizes,
    shapes_nearest_orthogonal,
    num_subs_list,
    sc_shape,
    sc_shapes,
    n_random,
    recursive,
    ds_generator_empty,
):
    ds_generator_empty.generate(
        supercell_sizes=supercell_sizes,
        shapes_nearest_orthogonal=shapes_nearest_orthogonal,
        num_subs_list=num_subs_list,
        sc_shape=sc_shape,
        sc_shapes=sc_shapes,
        n_random=1,
        random_state=42,
        recursive=recursive,
    )


@pytest.mark.parametrize("nsubs", [0, 1, 2, 3])
def test_degeneracy(ds_generator_empty, nsubs):
    dss = ds_generator_empty
    p = np.diag([2, 2, 2])
    scell = SuperCell(dss.plat, p)
    nsites = len(scell)
    sum_expected = math.comb(nsites, nsubs)
    dss.generate(
        supercell_sizes=None,
        num_subs_list=[[nsubs]],
        sc_shape=p,
        sc_shapes=None,
        n_random=None,
        random_state=42,
        recursive=False,
    )
    assert sum_expected == dss.configurations["degeneracy"].sum()


@pytest.mark.parametrize("recursive", [False, True])
def test_generate_pure(ds_generator_empty, recursive):
    p = np.diag([2, 2, 2])
    ds_generator_empty.generate(
        supercell_sizes=None,
        num_subs_list=[[0]],
        sc_shape=p,
        sc_shapes=None,
        n_random=None,
        random_state=42,
        recursive=recursive,
    )
    for _, row in ds_generator_empty.configurations.iterrows():
        assert np.all(np.array(row["sigma"]) == 0)
    assert len(ds_generator_empty.configurations) == 1


def test_compute_emt(ds_generator_full):
    calc = EMT2()
    ds_generator_full.compute_properties(
        property_name="energy_emt",
        ase_calculator=calc,
        cemodel=None,
        property_solver=None,
        property_solver_kwargs={},
        linear_reference=None,
        per_formula_unit=False,
    )
    assert "energy_emt" in ds_generator_full.configurations.columns


@pytest.fixture
def solver_single_properties_dummy():
    def _solver(*args, **kwargs):
        return 1.0

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
        "least-squares",
        "concentration-endpoints",
        [[0.0, 1.0], [1.0, 2.0]],
    ],
)
def test_compute_properties_single(
    ds_generator_full, solver_single_properties_rand, lin_ref
):
    ds_generator_full.compute_properties(
        property_name="dummy",
        ase_calculator=None,
        cemodel=None,
        property_solver=solver_single_properties_rand,
        property_solver_kwargs={},
        linear_reference=lin_ref,
        per_formula_unit=False,
    )
    assert "dummy" in ds_generator_full.configurations.columns


@pytest.fixture
def solver_multi_properties_dummy():
    def _solver(*args, **kwargs):
        return 1.0, 2.0  # Return multiple dummy properties

    return _solver


@pytest.fixture
def solver_multi_properties_rand():
    def _solver(*args, **kwargs):
        return np.random.rand(), np.random.rand()  # Return multiple dummy properties

    return _solver


@pytest.mark.parametrize(
    "lin_ref",
    [
        None,
        "least-squares",
        "concentration-endpoints",
        [[0.0, 1.0], [1.0, 2.0]],
    ],
)
def test_compute_properties_multi(
    ds_generator_full, solver_multi_properties_rand, lin_ref
):
    ds_generator_full.compute_properties(
        property_names=["dummy1", "dummy2"],
        ase_calculator=None,
        cemodel=None,
        property_solver=solver_multi_properties_rand,
        property_solver_kwargs={},
        linear_reference=lin_ref,
        per_formula_unit=False,
    )
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
            per_formula_unit=False,
        )


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
            per_formula_unit=False,
        )


def test_legacy_class_alias(parent_lattice):
    assert DSGenerator is DerivativeStructures
    assert DerivativeStructures.__name__ == "DerivativeStructures"
    assert type(DSGenerator(parent_lattice)) is DerivativeStructures
    assert isinstance(DerivativeStructures(parent_lattice), DSGenerator)


@pytest.mark.parametrize("protocol", [0, pickle.HIGHEST_PROTOCOL])
def test_load_legacy_pickle(ds_generator_full, tmp_path, monkeypatch, protocol):
    filepath = tmp_path / "legacy_dsgenerator.pickle"
    ds_generator_full.add_mask("selected", [0])
    ds_generator_full.metadata["source"] = "legacy"
    ds_generator_full.configurations["energy"] = np.arange(len(ds_generator_full.configurations), dtype=float)
    ds_generator_full.property_names.add("energy")

    # Reproduce the historical module/class path and instance state without
    # changing the canonical class's name or its methods.
    legacy_class = type("DSGenerator", (), {"__module__": DerivativeStructures.__module__})
    legacy_instance = legacy_class()
    legacy_instance.__dict__.update(ds_generator_full.__dict__)
    with monkeypatch.context() as legacy_name:
        legacy_name.setattr("clusterx.derivative_structures.DSGenerator", legacy_class)
        payload = pickle.dumps(legacy_instance, protocol=protocol)

    assert b"DSGenerator" in payload
    filepath.write_bytes(payload)

    for loaded in (pickle.loads(payload), DerivativeStructures.from_file(filepath), DSGenerator.from_file(filepath)):
        assert type(loaded) is DerivativeStructures
        assert loaded.plat == ds_generator_full.plat
        pd.testing.assert_frame_equal(loaded.scell_shapes, ds_generator_full.scell_shapes)
        pd.testing.assert_frame_equal(loaded.configurations, ds_generator_full.configurations)
        assert loaded.masks == ds_generator_full.masks
        assert loaded.property_names == ds_generator_full.property_names
        assert loaded.metadata == ds_generator_full.metadata


def test_serialize_load(ds_generator_full, tmp_path):
    filepath = tmp_path / "test_ds_generator.json"
    ds_generator_full.serialize(filepath)
    ds_generator_loaded = DerivativeStructures.from_file(filepath)
    assert type(ds_generator_loaded) is DerivativeStructures
    assert ds_generator_full.plat == ds_generator_loaded.plat
    pd.testing.assert_frame_equal(
        ds_generator_full.scell_shapes, ds_generator_loaded.scell_shapes
    )
    pd.testing.assert_frame_equal(
        ds_generator_full.configurations, ds_generator_loaded.configurations
    )


def test_compute_multiplicities_cli_persists(ds_generator_full, tmp_path):
    filepath = tmp_path / "dss.pickle"
    ds_generator_full.serialize(filepath)

    compute_multiplicities(str(filepath), quiet=True)

    ds_generator_loaded = DerivativeStructures.from_file(filepath)
    assert "multiplicity" in ds_generator_loaded.configurations.columns
    assert "multiplicity" in ds_generator_loaded.get_property_names()
    assert len(ds_generator_loaded.configurations["multiplicity"]) == len(
        ds_generator_full.configurations
    )
    assert all(ds_generator_loaded.configurations["multiplicity"] > 0)


def test_from_structures_set_file_loads_structures_set_db(ds_generator_full, tmp_path):
    data = ds_generator_full.configurations.head(4).copy()
    energies = [float(i) for i in range(len(data))]
    masks = ["selected", "other", "selected", None]
    sset = StructuresSet(ds_generator_full.plat)
    scell_cache = {}

    for _, row in data.iterrows():
        shape_id = row["shape_id"]
        if shape_id not in scell_cache:
            scell_cache[shape_id] = SuperCell(
                ds_generator_full.plat,
                ds_generator_full.get_scell_shape(shape_id),
            )

        sset.add_structure(Structure(scell_cache[shape_id], sigmas=row["sigma"]))

    sset.set_property_values(
        property_name="energy", property_vals=energies, update_db=False
    )
    sset.set_property_values(property_name="mask", property_vals=masks, update_db=False)

    filepath = tmp_path / "sset.db"
    sset.serialize(filepath=str(filepath), ase_db_type="db", overwrite=True, rm_vac=False)

    with pytest.raises(Exception):
        DerivativeStructures.from_file(filepath)

    dss = DerivativeStructures.from_structures_set_file(filepath)

    assert dss.plat == ds_generator_full.plat
    assert dss.property_names == {"energy", "mask"}
    assert dss.masks == {"selected": [0, 2], "other": [1]}
    assert dss.configurations["config_id"].tolist() == list(range(len(data)))
    assert dss.configurations["sigma"].tolist() == [
        tuple(row["sigma"]) for _, row in data.iterrows()
    ]
    np.testing.assert_array_equal(dss.configurations["energy"], energies)
    loaded_masks = dss.configurations["mask"].tolist()
    assert loaded_masks[:3] == masks[:3]
    assert pd.isna(loaded_masks[3])

    for loaded_shape_id, original_shape_id in zip(
        dss.configurations["shape_id"], data["shape_id"]
    ):
        np.testing.assert_array_equal(
            dss.get_scell_shape(loaded_shape_id),
            ds_generator_full.get_scell_shape(original_shape_id),
        )

    sset_selected = dss.to_structures_set(mask_name="selected")
    assert len(sset_selected) == 2
    assert sset_selected.get_property_values("mask").tolist() == [
        "selected",
        "selected",
    ]
    np.testing.assert_array_equal(
        sset_selected.get_property_values("energy"),
        [0.0, 2.0],
    )


def test_convert_sset_dss_cli_roundtrip(ds_generator_full, tmp_path):
    data = ds_generator_full.configurations.head(3).copy()
    sset = StructuresSet(ds_generator_full.plat)
    scell_cache = {}

    for _, row in data.iterrows():
        shape_id = row["shape_id"]
        if shape_id not in scell_cache:
            scell_cache[shape_id] = SuperCell(
                ds_generator_full.plat,
                ds_generator_full.get_scell_shape(shape_id),
            )

        sset.add_structure(Structure(scell_cache[shape_id], sigmas=row["sigma"]))

    sset.set_property_values(
        property_name="energy",
        property_vals=[1.0, 2.0, 3.0],
        update_db=False,
    )

    sset_filepath = tmp_path / "sset.db"
    dss_filepath = tmp_path / "dss.pickle"
    sset_roundtrip_filepath = tmp_path / "sset_roundtrip.db"
    sset.serialize(
        filepath=str(sset_filepath),
        ase_db_type="db",
        overwrite=True,
        rm_vac=False,
    )

    convert_sset_to_dss(str(sset_filepath), str(dss_filepath))
    dss = DerivativeStructures.from_file(dss_filepath)
    assert dss.configurations["sigma"].tolist() == [
        tuple(row["sigma"]) for _, row in data.iterrows()
    ]
    np.testing.assert_array_equal(dss.configurations["energy"], [1.0, 2.0, 3.0])

    convert_dss_to_sset(
        str(dss_filepath),
        str(sset_roundtrip_filepath),
        ase_db_type="db",
    )
    sset_roundtrip = StructuresSet(filepath=str(sset_roundtrip_filepath))
    assert len(sset_roundtrip) == 3
    np.testing.assert_array_equal(
        sset_roundtrip.get_property_values("energy"),
        [1.0, 2.0, 3.0],
    )


def _add_energy_property(dss):
    energies = np.arange(len(dss.configurations), dtype=float)
    dss.configurations["energy"] = energies
    dss.property_names.add("energy")
    return energies


def test_add_property_column_registers_property(ds_generator_full):
    ds_generator_full.add_property_column("energy", default_value=0.0)

    assert ds_generator_full.get_property_names() == {"energy"}
    assert ds_generator_full.configurations["energy"].tolist() == [0.0] * len(
        ds_generator_full.configurations
    )


def _write_ccalc(corrc, tmp_path):
    ccalc_filepath = tmp_path / "ccalc.pickle"
    corrc.serialize(filepath=str(ccalc_filepath), fmt="pickle")
    return ccalc_filepath


def test_compute_comat_combines_sset_filepaths(ds_generator_full, corrc, tmp_path):
    energies = _add_energy_property(ds_generator_full)
    sset = ds_generator_full.to_structures_set()
    ccalc_filepath = _write_ccalc(corrc, tmp_path)
    sset_filepath_a = tmp_path / "sset_a.db"
    sset_filepath_b = tmp_path / "sset_b.db"
    comat_filepath = tmp_path / "multi_sset_comat"

    for sset_filepath in [sset_filepath_a, sset_filepath_b]:
        sset.serialize(
            filepath=str(sset_filepath),
            ase_db_type="db",
            overwrite=True,
            rm_vac=False,
        )

    compute_comat(
        ccalc_filepath=str(ccalc_filepath),
        sset_filepaths=[str(sset_filepath_a), str(sset_filepath_b)],
        comat_filepath=str(comat_filepath),
        property_name="energy",
    )

    with np.load(f"{comat_filepath}.npz") as arrays:
        comat = arrays["comat"]
        pvals = arrays["property_energy"]

    assert comat.shape[0] == 2 * len(sset)
    np.testing.assert_array_equal(pvals, np.concatenate([energies, energies]))
    np.testing.assert_allclose(np.loadtxt(f"{comat_filepath}.txt"), comat)


def test_compute_comat_combines_dss_filepaths(ds_generator_full, corrc, tmp_path):
    energies = _add_energy_property(ds_generator_full)
    ccalc_filepath = _write_ccalc(corrc, tmp_path)
    dss_filepath_a = tmp_path / "dss_a.pickle"
    dss_filepath_b = tmp_path / "dss_b.pickle"
    comat_filepath = tmp_path / "multi_dss_comat"
    ds_generator_full.serialize(str(dss_filepath_a))
    ds_generator_full.serialize(str(dss_filepath_b))

    compute_comat(
        ccalc_filepath=str(ccalc_filepath),
        dss_filepaths=[str(dss_filepath_a), str(dss_filepath_b)],
        comat_filepath=str(comat_filepath),
        property_name="energy",
    )

    with np.load(f"{comat_filepath}.npz") as arrays:
        comat = arrays["comat"]
        pvals = arrays["property_energy"]

    assert comat.shape[0] == 2 * len(ds_generator_full.configurations)
    np.testing.assert_array_equal(pvals, np.concatenate([energies, energies]))


def test_compute_comat_rejects_multiple_sources():
    with pytest.raises(ValueError, match="Exactly one"):
        compute_comat(sset_filepath="sset.db", dss_filepath="dss.pickle")


def test_concentration(ds_generator_full):
    assert "frconc_binary" not in ds_generator_full.configurations.columns
    ds_generator_full.add_fractional_concentration_binary()
    assert "frconc_binary" in ds_generator_full.configurations.columns


def test_get_all_hnf():
    nrs = []
    for n in range(1, 7):
        hnfs = get_HNFs(n)
        nrs.append(len(hnfs))

    assert (np.array(nrs) == np.array([1, 7, 13, 35, 31, 91])).all()

    hnfs = get_HNFs(4, pbc=(1, 1, 0))

    hnfs_ref = [
        [[1, 0, 0], [0, 4, 0], [0, 0, 1]],
        [[1, 1, 0], [0, 4, 0], [0, 0, 1]],
        [[1, 2, 0], [0, 4, 0], [0, 0, 1]],
        [[1, 3, 0], [0, 4, 0], [0, 0, 1]],
        [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        [[2, 1, 0], [0, 2, 0], [0, 0, 1]],
        [[4, 0, 0], [0, 1, 0], [0, 0, 1]],
    ]

    isok = True

    # Now check that hnfs_ref is identical to hnfs
    for m1, m2 in zip(hnfs, hnfs_ref):
        for r1, r2 in zip(m1, m2):
            for i1, i2 in zip(r1, r2):
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
        if case == 0:  # Square (2D, i.e. pbc = (1,1,0))
            a = 3.1
            index = 4
            cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            positions = np.array([[0, 0, 0]])
            sites = [[12, 13]]
            pris = Atoms(cell=cell * a, positions=positions * a)

            pl = ParentLattice(pris, sites=sites, pbc=(1, 1, 0))

            unique_scs, unique_trafos = get_unique_supercells(index, pl)

            sset = StructuresSet(pl)
            for t in unique_trafos:
                scell = SuperCell(pl, t)
                sset.add_structure(
                    Structure(scell, scell.get_atomic_numbers()), write_to_db=True
                )

            sset.serialize(
                filepath="test_get_unique_supercells-square_lattice.json",
                overwrite=True,
            )
            print(
                "\nFound ",
                len(unique_scs),
                " unique HNFs for a 2D square lattice of index ",
                index,
            )
            # print("SCS: ", unique_scs)
            # print("TRA: ", unique_trafos)
            isok0 = len(unique_scs) == 4 and unique_scs[1][1][1] == 12.4

        if case == 1:  # FCC
            a = 3
            index = 4
            cell = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
            positions = np.array([[0, 0, 0]])
            sites = [[an["Cu"], an["Au"]]]
            pris_fcc = Atoms(cell=cell * a, positions=positions * a, pbc=(1, 1, 1))

            pl = ParentLattice(pris_fcc, sites=sites)

            unique_scs, unique_trafos = get_unique_supercells(index, pl)

            sset = StructuresSet(pl)
            for t in unique_trafos:
                scell = SuperCell(pl, t)
                sset.add_structure(
                    Structure(scell, scell.get_atomic_numbers()), write_to_db=True
                )

            sset.serialize(
                filepath="test_get_unique_supercells-fcc.json", overwrite=True
            )
            print(
                "Found ",
                len(unique_scs),
                " unique HNFs for a FCC lattice of index ",
                index,
            )
            # print("SCS: ", unique_scs)
            # print("TRA: ", unique_trafos)
            isok1 = len(unique_scs) == 7 and unique_trafos[4][2][2] == 4

        if case == 2:  # Simple cubic
            a = 3.1
            index = 4
            cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            positions = np.array([[0, 0, 0]])
            sites = [[12, 13]]
            pris = Atoms(cell=cell * a, positions=positions * a)

            pl = ParentLattice(pris, sites=sites, pbc=(1, 1, 1))

            unique_scs, unique_trafos = get_unique_supercells(index, pl)

            sset = StructuresSet(pl)
            for t in unique_trafos:
                scell = SuperCell(pl, t)
                sset.add_structure(
                    Structure(scell, scell.get_atomic_numbers()), write_to_db=True
                )

            sset.serialize(
                filepath="test_get_unique_supercells-sc.json", overwrite=True
            )
            print(
                "Found ",
                len(unique_scs),
                " unique HNFs for a simple cubic lattice of index ",
                index,
            )
            # print("SCS: ", unique_scs)
            # print("TRA: ", unique_trafos)
            isok2 = len(unique_scs) == 9 and unique_scs[4][2][2] == 12.4

    assert isok0 and isok1 and isok2
