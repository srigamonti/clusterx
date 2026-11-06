"""Affine prediction agrees with fitted pipelines and preserves MC sampling."""

from copy import deepcopy
import pickle

import numpy as np
import pytest
from ase import Atoms
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.estimators.column_expander import ColumnExpander
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.structure import Structure
from clusterx.super_cell import SuperCell
from clusterx.thermodynamics.monte_carlo import MonteCarlo


def training_data(multi_output=False):
    x = np.random.default_rng(17).normal(size=(60, 3)) + [3, -2, 5]
    y = 13 + x @ [1.2, -0.4, 0.7]
    if multi_output:
        y = np.column_stack([y, -2 * y + 1])
    return x, y


def fitted_pipeline(bias=True, mean=True, std=True, fit_intercept=False, multi_output=False):
    x, y = training_data(multi_output)
    # Deliberately permute inputs and insert unused zero columns.
    pipeline = Pipeline([
        ("expand", ColumnExpander(6, [4, 0, 2])),
        ("poly", PolynomialFeatures(degree=1, include_bias=bias)),
        ("scale", StandardScaler(with_mean=mean, with_std=std)),
        ("select", SelectFromModel(LinearRegression(fit_intercept=False), threshold=1e-8)),
        ("regressor", Ridge(alpha=0.01, fit_intercept=fit_intercept)),
    ])
    return pipeline.fit(x, y)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mean,std", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("multi_output", [False, True])
def test_affine_pipeline_predictions_and_differences(bias, mean, std, fit_intercept, multi_output):
    pipeline = fitted_pipeline(bias, mean, std, fit_intercept, multi_output)
    model = Model(estimator=pipeline)
    x = np.random.default_rng(29).normal(size=(8, 3))
    expected = pipeline.predict(x)
    actual = np.asarray([model.predict_on_correlations(row) for row in x])
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=1e-12)
    assert actual.shape == expected.shape
    assert model._get_affine_parameters(3) is not None
    before, after = x[:2]
    model.predict_correlations_diff = lambda *args, **kwargs: after - before
    delta = model.predict_flip(None, 0, 0, 1)
    np.testing.assert_allclose(delta, expected[1] - expected[0], atol=2e-13, rtol=1e-12)


@pytest.mark.parametrize("estimator", [
    LinearRegression(), Ridge(), RidgeCV(), Lasso(alpha=0.001), ElasticNet(alpha=0.001),
    LassoCV(alphas=[0.001, 0.01], cv=3), ElasticNetCV(alphas=[0.001, 0.01], cv=3),
])
def test_supported_regressors(estimator):
    x, y = training_data()
    estimator.fit(x, y)
    model = Model(estimator=estimator)
    np.testing.assert_allclose([model.predict_on_correlations(row) for row in x], estimator.predict(x), atol=1e-13)
    assert model._get_affine_parameters(3) is not None


@pytest.mark.parametrize("degree", [0, (0, 0)])
def test_constant_polynomial_bias(degree):
    x, _ = training_data()
    pipeline = Pipeline([("bias", PolynomialFeatures(degree=degree)),
                         ("regressor", Ridge(alpha=0.01, fit_intercept=False))]).fit(x, np.full(len(x), 5.0))
    model = Model(estimator=pipeline)
    np.testing.assert_allclose(model.predict_on_correlations(x[0]), pipeline.predict(x[:1])[0])
    model.predict_correlations_diff = lambda *args, **kwargs: x[1] - x[0]
    assert model.predict_flip(None, 0, 0, 1) == 0


@pytest.mark.parametrize("degree", [(0, 1), (1, 1)])
def test_tuple_polynomial_degree_and_fortran_order(degree):
    x, y = training_data()
    pipeline = Pipeline([("poly", PolynomialFeatures(degree=degree, order="F")),
                         ("regressor", Ridge(fit_intercept=False))]).fit(x, y)
    model = Model(estimator=pipeline)
    np.testing.assert_allclose([model.predict_on_correlations(row) for row in x], pipeline.predict(x), atol=1e-13)
    assert model._get_affine_parameters(3) is not None


class CurvedRidge(Ridge):
    def predict(self, x):
        return super().predict(x) + (x ** 2).sum(axis=1)


@pytest.mark.parametrize("transformer,regressor", [
    (PolynomialFeatures(degree=2), Ridge()),
    # Cubing agrees with identity on zero and unit vectors, but is nonlinear.
    (FunctionTransformer(lambda x: x ** 3), Ridge()),
    (PolynomialFeatures(degree=1), CurvedRidge()),
])
def test_unsupported_pipeline_uses_reference_prediction(transformer, regressor, monkeypatch):
    x, y = training_data()
    pipeline = Pipeline([("transformer", transformer), ("regressor", regressor)]).fit(x, y)
    model = Model(estimator=pipeline)
    expected = pipeline.predict(x[:2])
    calls = []
    predict = pipeline.predict

    def reference(values):
        calls.append(1)
        return predict(values)

    monkeypatch.setattr(pipeline, "predict", reference)
    np.testing.assert_allclose([model.predict_on_correlations(row) for row in x[:2]], expected)
    assert len(calls) == 2
    assert model._get_affine_parameters(3) is None


def test_cache_reuse_refit_replacement_and_explicit_invalidation(monkeypatch):
    pipeline = fitted_pipeline()
    model = Model(estimator=pipeline)
    x, y = training_data()
    model.predict_on_correlations(x[0])
    cached = model._affine_parameters
    predict = pipeline.predict
    calls = []

    def reference(values):
        calls.append(1)
        return predict(values)

    monkeypatch.setattr(pipeline, "predict", reference)
    for row in x[:3]:
        model.predict_on_correlations(row)
    assert model._affine_parameters is cached
    assert calls == []
    pipeline.fit(x, -3 * y + 2)
    np.testing.assert_allclose(model.predict_on_correlations(x[0]), pipeline.predict(x[:1])[0])
    assert model._affine_parameters is not cached
    model.estimator = fitted_pipeline(mean=False)
    np.testing.assert_allclose(model.predict_on_correlations(x[0]), model.estimator.predict(x[:1])[0])
    model.estimator[-1].coef_[:] *= 2
    model.estimator.set_params(scale__with_mean=True)
    model.invalidate_prediction_cache()
    np.testing.assert_allclose(model.predict_on_correlations(x[0]), model.estimator.predict(x[:1])[0])


def test_replacing_a_transformer_refreshes_coefficients():
    x, y = training_data()
    pipeline = Pipeline([("scale", StandardScaler()), ("regressor", Ridge())]).fit(x, y)
    model = Model(estimator=pipeline)
    model.predict_on_correlations(x[0])
    pipeline.steps[0] = ("scale", StandardScaler().fit(2 * x + 1))
    np.testing.assert_allclose(model.predict_on_correlations(x[0]), pipeline.predict(x[:1])[0])


def test_explicit_ecis_remain_unscaled_and_mutable():
    model = Model(ecis=np.array([2.0, -3.0, 4.0]))
    x = np.array([0.25, 0.5, 0.75])
    assert model.predict_on_correlations(x) == pytest.approx(2.0)
    model.ecis[:] *= 2
    assert model.predict_on_correlations(x) == pytest.approx(4.0)


@pytest.fixture
def calculator():
    plat = ParentLattice(Atoms("H", cell=[1, 1, 4], pbc=[True, True, False]), numbers=[[1, 7]])
    pool = ClustersPool(plat, npoints=[1, 2], radii=[0, 1.5])
    assert len(pool) == 3
    return CorrelationsCalculator("trigonometric", plat, pool)


def test_serialization_old_state_and_mc_reset(calculator, tmp_path):
    model = Model(calculator, "prop", estimator=fitted_pipeline())
    x, _ = training_data()
    expected = model.estimator.predict(x[:1])[0]
    np.testing.assert_allclose(model.predict_on_correlations(x[0]), expected)
    filepath = str(tmp_path / "model.pickle")
    model.serialize(filepath)
    for restored in (Model(filepath=filepath), pickle.loads(pickle.dumps(model))):
        assert "_affine_sources" not in restored.__dict__
        np.testing.assert_allclose(restored.predict_on_correlations(x[0]), expected)
        assert restored._get_affine_parameters(3) is not None
    # Older pickle states have the original estimator and no derived cache fields.
    old_state = model.__getstate__()
    assert not any(key.startswith("_affine_") for key in old_state)
    restored = Model.__new__(Model)
    restored.__setstate__(old_state)
    np.testing.assert_allclose(restored.predict_on_correlations(x[0]), expected)
    restored.estimator[-1].coef_[:] *= 2
    restored.reset_mc()
    np.testing.assert_allclose(restored.predict_on_correlations(x[0]), restored.estimator.predict(x[:1])[0])


def test_affine_reduced_total_prediction(calculator):
    model = Model(calculator, "prop", estimator=fitted_pipeline())
    scell = SuperCell(model.get_parent_lattice(), [7, 7, 1])
    structure = scell.gen_random_structure(nsubs={0: [24]}, rng=np.random.default_rng(31))
    expected = model.predict(structure)
    model.init_reduced_model(structure)
    structure.init_reduced_structure(model.scell_reduced)
    np.testing.assert_allclose(model.predict(structure, reduce_structure=True), expected, atol=2e-12)


def test_fixed_seed_mc_matches_pipeline_sampling(calculator, tmp_path):
    original = Model(calculator, "prop", estimator=fitted_pipeline())
    scell = SuperCell(original.get_parent_lattice(), [7, 7, 1])
    initial = scell.gen_random_structure(nsubs={0: [24]}, rng=np.random.default_rng(43))
    results = []
    for fast in (False, True):
        model = deepcopy(original)
        if not fast:
            pipeline = model.estimator
            offset = pipeline.predict(np.zeros((1, 3)))[0]
            model._get_affine_parameters = lambda width: None

            def reference_flip(structure, index, old_sigma, new_sigma, reduce_structure=False):
                delta = model.predict_correlations_diff(structure, index, old_sigma, new_sigma, reduce_structure)
                return pipeline.predict(delta.reshape(1, -1))[0] - offset

            model.predict_flip = reference_flip
        mc = MonteCarlo(model, scell, nsubs={0: [24]}, predict_swap=True, reduce_structure=True, seed=91)
        trajectory = mc.metropolis(
            no_of_sampling_steps=80, temperature=0.3, initial_decoration=initial.get_decoration(),
            serialize=True, filename=str(tmp_path / f"trajectory-{fast}.json"),
        )
        np.testing.assert_array_equal(np.bincount(mc.structure.sigmas, minlength=2), [25, 24])
        sigmas = initial.sigmas.copy()
        energy0 = original.estimator.predict(calculator.get_cluster_correlations(initial).reshape(1, -1))[0]
        for nid in range(len(trajectory.get_energies())):
            entry = trajectory.get_sampling_step_entry(nid)
            sigmas[entry["indices"]] = entry["new_sigmas"]
            structure = Structure(scell, sigmas=sigmas.copy())
            correlations = calculator.get_cluster_correlations(structure)
            expected = original.estimator.predict(correlations.reshape(1, -1))[0] - energy0
            np.testing.assert_allclose(entry["energy"], expected, atol=2e-12)
        results.append((mc.structure.sigmas.copy(), trajectory.get_sampling_step_nos(), trajectory.get_energies(),
                        deepcopy(mc._rng_np.bit_generator.state), mc._rng_std.getstate()))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])
    np.testing.assert_allclose(results[0][2], results[1][2], atol=2e-12)
    np.testing.assert_equal(results[0][3], results[1][3])
    assert results[0][4] == results[1][4]
