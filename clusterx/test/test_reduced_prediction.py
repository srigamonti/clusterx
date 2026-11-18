"""Full predictions with reduced geometry and local linear Monte Carlo."""

import numpy as np
import pytest
from ase import Atoms
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import clusterx.model as model_module
from clusterx.cli.build_model_from_sset import build_model_from_sset
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.estimators.column_expander import ColumnExpander
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.structure import Structure
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
from clusterx.thermodynamics.monte_carlo import MonteCarlo


def calculator(pbc=(True, True, False)):
    atoms = Atoms("HHe", positions=[[0, 0, 0], [0.3, 0.4, 0.2]],
                  cell=[[1, 0, 0], [0.2, 1.1, 0], [0.1, 0.3, 1.4]], pbc=pbc)
    plat = ParentLattice(atoms, numbers=[[1, 7], [2]])
    pool = ClustersPool(plat, npoints=[1, 2, 3], radii=[0, 1.25, 1.55])
    return CorrelationsCalculator("trigonometric", plat, pool)


def fitted_model(degree):
    corrc = calculator()
    width = len(corrc.get_cpool())
    x = np.random.default_rng(103).uniform(-1, 1, size=(140, width))
    y = 1.3 + x @ np.linspace(0.15, 0.4, width) + 0.7 * x[:, 1] ** degree + 0.3 * x[:, 1] * x[:, 2]
    pipeline = Pipeline([
        ("expand", ColumnExpander(width + 2, list(reversed(range(width))))),
        ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
        ("scale", StandardScaler()),
        ("select", SelectFromModel(LinearRegression(), threshold=1e-8)),
        ("regressor", Ridge(alpha=0.001)),
    ]).fit(x, y)
    return Model(corrc, "energy", estimator=pipeline)


def initialize_reduced(model, structure):
    model.init_reduced_model(structure)
    structure.init_reduced_structure(model.scell_reduced)


@pytest.mark.parametrize("packed", [False, True])
def test_streamed_correlations_analytic_chain(packed, monkeypatch):
    atoms = Atoms("HHe", positions=[[0, 0, 0], [0.3, 0.2, 0.1]],
                  cell=[1, 3, 4], pbc=[True, False, False])
    plat = ParentLattice(atoms, numbers=[[1, 7], [2]])
    pool = ClustersPool(plat, npoints=[1, 2, 3], radii=[0, 1.1, 2.1])
    model = Model(CorrelationsCalculator("trigonometric", plat, pool), "energy", ecis=np.ones(len(pool)))
    scell = SuperCell(plat, (8, 1, 1))
    sigmas = np.zeros(len(scell), dtype=np.uint8)
    sigmas[scell.get_substitutional_sites()] = np.arange(8) % 2
    structure = Structure(scell, sigmas=sigmas)
    if not packed:
        monkeypatch.setattr(model_module, "_pack_cluster_interactions", lambda *args: None)
    initialize_reduced(model, structure)
    expected = np.array([-1.0 if n == 2 else 0.0 for n in pool.get_all_npoints()])
    np.testing.assert_allclose(model.get_correlations(structure, reduce_structure=True), expected, atol=1e-12)


@pytest.mark.parametrize("pbc,shape", [
    ((True, False, False), (9, 1, 1)),
    ((True, True, False), (8, 8, 1)),
    ((True, True, True), (5, 5, 5)),
])
def test_streamed_correlations_match_full_geometry(pbc, shape):
    corrc = calculator(pbc)
    model = Model(corrc, "energy", ecis=np.ones(len(corrc.get_cpool())))
    scell = SuperCell(model.get_parent_lattice(), shape)
    structure = scell.gen_random_structure(rng=np.random.default_rng(107))
    expected = corrc.get_cluster_correlations(structure)
    initialize_reduced(model, structure)
    np.testing.assert_allclose(model.get_correlations(structure, reduce_structure=True), expected, atol=2e-12)


def test_streamed_ternary_correlations_match_full_geometry():
    atoms = Atoms("HHe", positions=[[0, 0, 0], [0.3, 0.4, 0.2]],
                  cell=[[1, 0, 0], [0.2, 1.1, 0], [0, 0, 4]], pbc=[True, True, False])
    plat = ParentLattice(atoms, numbers=[[1, 6, 7], [2]])
    pool = ClustersPool(plat, npoints=[1, 2, 3], radii=[0, 1.55, 1.55])
    corrc = CorrelationsCalculator("trigonometric", plat, pool)
    model = Model(corrc, "energy", ecis=np.ones(len(pool)))
    structure = SuperCell(plat, (6, 5, 1)).gen_random_structure(rng=np.random.default_rng(227))
    expected = corrc.get_cluster_correlations(structure)
    initialize_reduced(model, structure)
    np.testing.assert_allclose(model.get_correlations(structure, reduce_structure=True), expected, atol=2e-12)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_reduced_prediction_evaluates_global_correlations(degree):
    model = fitted_model(degree)
    scell = SuperCell(model.get_parent_lattice(), (8, 8, 1))
    structure = scell.gen_random_structure(rng=np.random.default_rng(109))
    expected = model.predict(structure)
    initialize_reduced(model, structure)
    np.testing.assert_allclose(model.predict(structure, reduce_structure=True), expected, atol=2e-11)


def test_streaming_requires_a_matching_reduced_grid():
    model = fitted_model(2)
    plat = model.get_parent_lattice()
    structure = SuperCell(plat, (8, 8, 1)).get_pristine_structure()
    structure.init_reduced_structure(SuperCell(plat, (2, 2, 1)))
    with pytest.raises(ValueError, match="odd repeat lengths"):
        model.get_correlations(structure, reduce_structure=True)


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("standardize", [False, True])
def test_pruned_builder_nonlinear_model_roundtrip(degree, standardize, tmp_path):
    atoms = Atoms("HHe", positions=[[0, 0, 0], [0.3, 0.2, 0.1]],
                  cell=[1, 3, 4], pbc=[True, False, False])
    plat = ParentLattice(atoms, numbers=[[1, 7], [2]])
    pool = ClustersPool(plat, npoints=[1, 2], radii=[0, 2.1])
    corrc = CorrelationsCalculator("trigonometric", plat, pool)
    training = StructuresSet(plat)
    rng = np.random.default_rng(137)
    for i in range(100):
        cell = SuperCell(plat, (9 + 2 * (i % 3), 1, 1))
        training.add_structure(cell.gen_random_structure(rng=rng))
    x = corrc.get_correlation_matrix(training)
    training.set_property_values("prop", 1.2 + 0.3 * x[:, 0] + 0.4 * x[:, 1] + 0.7 * x[:, 1] ** degree)
    source = str(tmp_path / "training.xyz")
    training.serialize(source)
    prefix = str(tmp_path / "fit")
    regression = {"module": "sklearn.linear_model", "class": "LinearRegression", "args": {}}
    build_model_from_sset(
        "prop", source, npoints=[1, 2], radii=[0, 2.1], output_prefix=prefix,
        regression_model=regression, selection_model=regression, selection_threshold=1e-7,
        nonlinear_transformation={"args": {"degree": degree, "include_bias": False}}, standardize=standardize,
    )
    full = Model(filepath=f"{prefix}_1_full_model.pickle")
    pruned = Model(filepath=f"{prefix}_2_pruned_model.pickle")
    assert 0 < len(pruned.corrc.get_cpool()) < len(full.corrc.get_cpool())
    assert pruned._get_affine_parameters(len(pruned.corrc.get_cpool())) is None
    scell = SuperCell(plat, (16, 1, 1))
    tag = int(scell.get_substitutional_tags()[0])
    initial = scell.gen_random_structure(nsubs={tag: [7]}, rng=np.random.default_rng(139))
    np.testing.assert_allclose(pruned.predict(initial), full.predict(initial), atol=2e-10)


def test_affine_mc_does_not_initialize_global_correlations(monkeypatch):
    model = fitted_model(1)
    scell = SuperCell(model.get_parent_lattice(), (8, 8, 1))
    tag = int(scell.get_substitutional_tags()[0])

    def unexpected_initialization(*args, **kwargs):
        pytest.fail("Affine MC must retain the direct local energy-difference path")

    monkeypatch.setattr(model, "get_correlations", unexpected_initialization)
    mc = MonteCarlo(model, scell, nsubs={tag: [32]}, predict_swap=True, reduce_structure=True, seed=151)
    mc.metropolis(no_of_sampling_steps=24, temperature=0.05,
        energy_factor=1.0,
        boltzmann_constant=1.0,
    )


def test_local_mc_rejects_cluster_images_on_the_same_atom():
    plat = ParentLattice(Atoms("H", cell=[1, 3, 4], pbc=[True, False, False]), numbers=[[1, 7]])
    pool = ClustersPool(plat, npoints=[1, 2, 3], radii=[0, 1.1, 2.1])
    corrc = CorrelationsCalculator("trigonometric", plat, pool)
    x = np.random.default_rng(157).normal(size=(80, len(pool)))
    pipeline = Pipeline([("poly", PolynomialFeatures(1)), ("regressor", Ridge())]).fit(x, x[:, 2])
    model = Model(corrc, "energy", estimator=pipeline)
    scell = SuperCell(plat, (2, 1, 1))
    mc = MonteCarlo(model, scell, nsubs={0: [1]}, predict_swap=True, reduce_structure=True, seed=163)
    with pytest.raises(ValueError, match="wraps onto the same atom"):
        mc.metropolis(no_of_sampling_steps=1,
            energy_factor=1.0,
            boltzmann_constant=1.0,
            temperature=1.0,
        )


def test_small_periodic_cell_without_self_aliasing_is_supported():
    plat = ParentLattice(Atoms("H", cell=[1, 3, 4], pbc=[True, False, False]), numbers=[[1, 7]])
    pool = ClustersPool(plat, npoints=[1, 2], radii=[0, 1.1])
    corrc = CorrelationsCalculator("trigonometric", plat, pool)
    x = np.random.default_rng(181).normal(size=(50, len(pool)))
    pipeline = Pipeline([("poly", PolynomialFeatures(1)), ("regressor", Ridge())]).fit(x, x[:, 1])
    model = Model(corrc, "energy", estimator=pipeline)
    scell = SuperCell(plat, (2, 1, 1))
    mc = MonteCarlo(model, scell, nsubs={0: [1]}, predict_swap=True, reduce_structure=True, seed=191)
    mc.metropolis(no_of_sampling_steps=8,
        energy_factor=1.0,
        boltzmann_constant=1.0,
        temperature=1.0,
    )
