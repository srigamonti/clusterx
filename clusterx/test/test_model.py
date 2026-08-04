# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

"""Tests for clusterx.model module."""

import os

import pytest
from ase.build import bulk
import numpy as np
from sklearn.linear_model import LinearRegression

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.model import Model, ModelBuilder
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.derivative_structures import DSGenerator
from clusterx.structures_set import StructuresSet
from clusterx.calculators.emt import EMT2
from clusterx.cli.build_model import build_model
from clusterx.cli.compute_predictions import compute_predictions


@pytest.fixture
def plat():
    pri = bulk("H", crystalstructure="sc", a=1.0)
    sub = bulk("He", crystalstructure="sc", a=1.0)
    return ParentLattice(atoms=pri, substitutions=[sub], pbc=(1, 1, 1))


@pytest.fixture
def cpool(plat):
    return ClustersPool(plat, npoints=[1, 2], radii=[0, 1.1])


@pytest.fixture
def corrc(plat, cpool):
    return CorrelationsCalculator("trigonometric", plat, cpool)


@pytest.fixture
def scell(plat):
    p = [3, 3, 3]
    return SuperCell(plat, p)


@pytest.fixture
def structure(plat, scell):
    rng = np.random.default_rng(42)
    return scell.gen_random_structure(nsubs=int(np.prod(scell.get_index()) / 2), rng=rng)


@pytest.fixture
def structures_set_empty(plat):
    return StructuresSet(parent_lattice=plat)


@pytest.fixture
def model(cpool, corrc):
    """Init with ECIS, default."""
    mult = cpool.get_multiplicities()
    model = Model(corrc, "energy", ecis=mult)
    return model


@pytest.fixture
def dummy_ce_11(cpool, corrc):
    model = Model(corrc, "energy", ecis=[1, 1])
    return model


@pytest.fixture
def dummy_ce_12(cpool, corrc):
    model = Model(corrc, "energy", ecis=[1, 2])
    return model


@pytest.fixture
def structures_set(structures_set_empty, scell, dummy_ce_11, dummy_ce_12):
    sset = structures_set_empty
    rng = np.random.default_rng(42)  # failed seeds: 42, success: 43
    n_structures = 10
    for i in range(n_structures):
        sset.add_structure(scell.gen_random_structure(rng=rng))
    sset.set_calculator(EMT2())
    sset.compute_property_values(property_name="energy_emt2")

    def a_prop(i, structure, **kwargs):
        at = structure.get_atoms()
        at.calc = EMT2()
        return at.get_potential_energy()

    sset.compute_property_values(property_name="energy_shift", property_calc=a_prop)
    ce_preds_11 = sset.get_predictions(dummy_ce_11)
    sset.set_property_values("dummy_ce_11", ce_preds_11)
    ce_preds_12 = sset.get_predictions(dummy_ce_12)
    sset.set_property_values("dummy_ce_12", ce_preds_12)
    return sset


@pytest.fixture
def model_estimator(cpool, corrc):
    """Init with estimator."""
    mult = cpool.get_multiplicities()
    estimator = LinearRegression()
    estimator.coef_ = mult
    estimator.intercept_ = 1.0
    model = Model(corrc, "energy", estimator=estimator)
    return model


def test_estimator_direct(corrc, model_estimator, structures_set, scell):
    model = model_estimator
    for structure in structures_set:
        correlations = corrc.get_cluster_correlations(structure)
        pred = model.estimator.predict(correlations.reshape(1, -1))
        pred_direct = model.predict(structure)
        structure.init_reduced_structure(scell)
        pred_reduced = model.predict(structure, reduce_structure=True)
        np.testing.assert_allclose(pred, pred_direct)
        np.testing.assert_allclose(pred, pred_reduced)


def test_raise_init(cpool, corrc):
    with pytest.raises(ValueError):
        Model(corrc, "energy", ecis=None, estimator=None)


def test_serialize_load(model, structure):
    filepath = "model_test.pickle"
    model.serialize(filepath=filepath)
    model_loaded = Model(filepath=filepath)
    prediction = model.predict(structure)
    prediction_loaded = model_loaded.predict(structure)
    np.testing.assert_allclose(prediction_loaded, prediction)


def test_compute_predictions_cli_structures_set(dummy_ce_11, plat, scell, tmp_path, capsys):
    sset_filepath = tmp_path / "sset.db"
    model_filepath = tmp_path / "model.pickle"
    plat_filepath = tmp_path / "plat.json"
    property_name = "predicted_energy"
    sigmas0 = np.zeros(scell.get_natoms(), dtype=int)
    sigmas1 = sigmas0.copy()
    sigmas1[0] = 1
    structures_set = StructuresSet(plat)
    structures_set.add_structure(scell.gen_structure(sigmas=sigmas0))
    structures_set.add_structure(scell.gen_structure(sigmas=sigmas1))
    expected = [dummy_ce_11.predict(structure) for structure in structures_set]

    structures_set.serialize(
        filepath=str(sset_filepath),
        ase_db_type="db",
        overwrite=True,
        rm_vac=False,
    )
    dummy_ce_11.serialize(filepath=str(model_filepath))
    plat.serialize(filepath=str(plat_filepath))

    compute_predictions(
        str(sset_filepath),
        property_name,
        str(model_filepath),
        plat_filepath=str(plat_filepath),
    )

    out = capsys.readouterr().out
    assert "CE model property_name is 'energy'" in out
    sset_loaded = StructuresSet(filepath=str(sset_filepath))
    np.testing.assert_allclose(
        sset_loaded.get_property_values(property_name),
        expected,
    )


def test_compute_predictions_cli_dsgenerator(dummy_ce_11, plat, tmp_path):
    dss_filepath = tmp_path / "dss.pickle"
    model_filepath = tmp_path / "model.pickle"
    property_name = "predicted_energy"
    shape = np.diag([2, 1, 1])
    scell = SuperCell(plat, shape)
    sigmas = [(1, 0), (0, 0)]
    expected = [dummy_ce_11.predict(scell.gen_structure(sigmas=sigma)) for sigma in sigmas]

    dss = DSGenerator(plat)
    shape_id = dss.add_scell_shape(shape)
    for sigma in sigmas:
        dss.add_configuration(sigma=sigma, shape_id=shape_id)
    dss.serialize(filepath=str(dss_filepath))
    dummy_ce_11.serialize(filepath=str(model_filepath))

    compute_predictions(str(dss_filepath), property_name, str(model_filepath))

    dss_loaded = DSGenerator.from_file(dss_filepath)
    assert property_name in dss_loaded.get_property_names()
    np.testing.assert_allclose(
        dss_loaded.configurations[property_name].to_numpy(),
        expected,
    )


def test_ecis_estimator_init(model, model_estimator, structure):
    prediction = model.predict(structure) + model_estimator.estimator.intercept_
    prediction_estimator = model_estimator.predict(structure)
    np.testing.assert_allclose(prediction_estimator, prediction)


@pytest.mark.parametrize("prop,shape", [("dummy_ce_11", ()), (["dummy_ce_11", "dummy_ce_12"], (2,))])
@pytest.mark.parametrize(
    "selector_type,selector_opts",
    [("identity", {}), ("lasso_cv", {}), ("skl_lasso_cv", {})],
)
def test_model_builder(structures_set, cpool, corrc, prop, shape, selector_type, selector_opts):
    mb = ModelBuilder(
        basis="trigonometric",
        selector_type=selector_type,
        selector_opts=selector_opts,
        estimator_type="skl_LinearRegression",
        estimator_opts={},
        standardize=False,
    )
    property_names = prop
    prop_list = [prop] if isinstance(prop, str) else prop
    model = mb.build(
        sset=structures_set,
        cpool=cpool,
        prop=property_names,
        corrc=corrc,
        verbose=False,
    )
    structure = structures_set[0]
    prediction = model.predict(structure)
    np.testing.assert_array_equal(prediction.shape, shape)
    prediction = model.predict_flip(
        structure,
        index=0,
        old_sigma=0,
        new_sigma=1,
    )
    np.testing.assert_array_equal(prediction.shape, shape)
    for property_name in prop_list:
        errors = model.get_errors(structures_set, property_name)
        np.testing.assert_allclose(errors["RMSE"], 0.0, atol=1e-10)
        np.testing.assert_allclose(errors["MAE"], 0.0, atol=1e-10)
        np.testing.assert_allclose(errors["MaxAE"], 0.0, atol=1e-10)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_cli(structures_set, cpool, corrc, mode):
    sset_filepath = "test_model_sset.json"
    structures_set.serialize(sset_filepath, overwrite=True)
    cpool_filepath = "test_model_cpool.json"
    cpool.serialize(cpool_filepath)
    ccalc_filepath = "test_model_ccalc.pickle"
    corrc.serialize(ccalc_filepath)
    model_filepath = "test_model_path.pickle"

    if mode == "single":
        prop = "dummy_ce_11"
    elif mode == "multi":
        prop = ["dummy_ce_11", "dummy_ce_12"]

    build_model(
        property_name=prop,
        ccalc_filepath=ccalc_filepath,
        sset_filepath=sset_filepath,
        cpool_filepath=cpool_filepath,
        model_filepath=model_filepath,
        selector_type="identity",
        selector_opts={"fit_intercept": True},
        estimator_type="skl_LinearRegression",
        estimator_opts={"fit_intercept": True},
        plot_optimization_vs_sparsity=None,
        weights_filepath=None,
    )
    assert os.path.exists(model_filepath)


@pytest.mark.parametrize("props", ["single", "multi"])
def test_cli_lasso(structures_set, cpool, corrc, props):
    sset_filepath = "test_model_sset.json"
    structures_set.serialize(sset_filepath, overwrite=True)
    cpool_filepath = "test_model_cpool.json"
    cpool.serialize(cpool_filepath)
    ccalc_filepath = "test_model_ccalc.pickle"
    corrc.serialize(ccalc_filepath)
    model_filepath = "test_model_path.pickle"
    property_names = structures_set.get_property_names()

    if props == "single":
        prop = property_names[0]
    elif props == "multi":
        prop = property_names

    selector_type = "lasso_cv"
    selector_opts = {
        "sparsity_max": 1,
        "sparsity_min": 1e-3,
        "sparsity_step": 0.0,
        "max_iter": 1_000_000,
        "cv_type": "10-fold",
        "cv_shuffle": True,
    }

    build_model(
        property_name=prop,
        ccalc_filepath=ccalc_filepath,
        sset_filepath=sset_filepath,
        cpool_filepath=cpool_filepath,
        model_filepath=model_filepath,
        selector_type=selector_type,
        selector_opts=selector_opts,
        estimator_type="skl_LinearRegression",
        estimator_opts={"fit_intercept": True},
        plot_optimization_vs_sparsity=None,
        weights_filepath=None,
    )
    assert os.path.exists(model_filepath)
