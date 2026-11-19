"""Tests for the build_model_from_sset command and its helpers."""

import numpy as np
import pytest
from ase.build import bulk
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from clusterx.cli.build_model_from_sset import _output_paths, build_model_from_sset
from clusterx.cli.build_nonlinear_model import _build_pipeline, used_input_columns
from clusterx.cli.report_utils import MarkdownReport
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.estimators.column_expander import ColumnExpander
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell

LINEAR = {"module": "sklearn.linear_model", "class": "LinearRegression", "args": {}}
POLY2 = {
    "module": "sklearn.preprocessing",
    "class": "PolynomialFeatures",
    "args": {"degree": 2, "include_bias": False},
}
LASSO = {
    "module": "sklearn.linear_model",
    "class": "LassoCV",
    "args": {"cv": 3, "max_iter": 10000},
}


# ----------------------------------------------------------------------
# ColumnExpander
# ----------------------------------------------------------------------
def test_column_expander_scatters_and_pads():
    exp = ColumnExpander(5, [1, 3])
    out = exp.fit_transform(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert out.shape == (2, 5)
    np.testing.assert_array_equal(out[:, [1, 3]], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(out[:, [0, 2, 4]], 0)


def test_column_expander_wrong_width():
    with pytest.raises(ValueError):
        ColumnExpander(5, [1, 3]).transform(np.zeros((2, 3)))


# ----------------------------------------------------------------------
# used_input_columns
# ----------------------------------------------------------------------
@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(120, 6))
    y = 3 * x[:, 0] * x[:, 1] + 2 * x[:, 3] + 0.5
    return x, y


def test_used_columns_polynomial_with_lasso_selection(data):
    x, y = data
    pipe = _build_pipeline(LINEAR, POLY2, LASSO, selection_threshold=1e-3)
    pipe.fit(x, y)
    assert used_input_columns(pipe, 6).tolist() == [0, 1, 3]


def test_used_columns_must_include_is_kept(data):
    x, y = data
    pipe = _build_pipeline(
        LINEAR, POLY2, LASSO, selection_threshold=1e-3, selection_must_include=[4]
    )
    pipe.fit(x, y)
    assert 4 in used_input_columns(pipe, 6)


def test_used_columns_no_transformer_no_selector(data):
    x, y = data
    pipe = _build_pipeline(LINEAR)
    pipe.fit(x, y)
    assert used_input_columns(pipe, 6).tolist() == [0, 1, 2, 3, 4, 5]


def test_used_columns_zero_regressor_coefficients_are_dropped(data):
    x, y = data
    pipe = _build_pipeline(LINEAR, None, LASSO, selection_threshold=0.0)
    pipe.fit(x, y)
    pipe.named_steps["regressor"].coef_[2] = 0.0
    assert 2 not in used_input_columns(pipe, 6)


def test_pruned_pipeline_matches_full(data):
    """Zero-padding the reduced input is exact for the dropped columns."""
    x, y = data
    pipe = _build_pipeline(LINEAR, POLY2, LASSO, selection_threshold=1e-3)
    pipe.fit(x, y)
    kept = used_input_columns(pipe, 6)
    assert len(kept) < 6
    pruned = Pipeline([("expander", ColumnExpander(6, kept)), *pipe.steps])
    np.testing.assert_allclose(pruned.predict(x[:, kept]), pipe.predict(x), atol=1e-12)


# ----------------------------------------------------------------------
# Report helper
# ----------------------------------------------------------------------
def test_markdown_report(tmp_path, capsys):
    rep = MarkdownReport()
    rep.title("T")
    rep.table(["a", "b"], [[1, 2]])
    rep.capture(lambda: print("captured"))
    rep.save(str(tmp_path / "r.md"))
    text = (tmp_path / "r.md").read_text()
    assert "# T" in text and "| a | b |" in text and "captured" in text
    assert "captured" in capsys.readouterr().out


# ----------------------------------------------------------------------
# File naming
# ----------------------------------------------------------------------
def test_output_paths_sort_before_and_after_pruning():
    paths = _output_paths("prop", True)
    files = [paths["log"], *paths["full"].values(), *paths["pruned"].values()]
    stages = [f.split("_")[1] for f in sorted(files)]
    assert stages == sorted(stages)
    assert "pruned" not in _output_paths("prop", False)


# ----------------------------------------------------------------------
# End-to-end
# ----------------------------------------------------------------------
@pytest.fixture
def sset_filepath(tmp_path):
    pri = bulk("H", crystalstructure="sc", a=1.0)
    sub = bulk("He", crystalstructure="sc", a=1.0)
    plat = ParentLattice(atoms=pri, substitutions=[sub], pbc=(1, 1, 1))
    scell = SuperCell(plat, [3, 3, 3])
    rng = np.random.default_rng(7)

    sset = StructuresSet(parent_lattice=plat)
    for _ in range(40):
        sset.add_structure(scell.gen_random_structure(rng=rng))

    # Property depends only on the first two clusters (empty and point cluster)
    cpool = ClustersPool(plat, npoints=[1, 2], radii=[0, 2.1])
    ccalc = CorrelationsCalculator("trigonometric", plat, cpool)
    comat = ccalc.get_correlation_matrix(sset)
    sset.set_property_values("prop", 2.0 * comat[:, 0] + 1.5 * comat[:, 1] ** 2)

    path = str(tmp_path / "sset.xyz")
    sset.serialize(path)
    return path


def _build(sset_filepath, prefix, **kwargs):
    build_model_from_sset(
        "prop",
        sset_filepath,
        npoints=[1, 2],
        radii=[0, 2.1],
        output_prefix=str(prefix),
        **kwargs,
    )


def test_build_model_from_sset_pruned(sset_filepath, tmp_path):
    prefix = tmp_path / "run" / "prop"
    _build(
        sset_filepath,
        prefix,
        regression_model={"class": "LinearRegression"},
        selection_model={"args": {"cv": 3, "max_iter": 10000}},
        selection_threshold=1e-2,
        nonlinear_transformation=POLY2,
    )

    names = sorted(p.name for p in prefix.parent.iterdir() if "CCALC" not in p.name)
    assert names[0] == "prop_0_log.md"
    assert all("_1_full_" in n for n in names[1:5])
    assert all("_2_pruned_" in n for n in names[5:])
    assert len(names) == 9

    full = Model(filepath=f"{prefix}_1_full_model.pickle")
    pruned = Model(filepath=f"{prefix}_2_pruned_model.pickle")
    n_full = len(full.get_correlations_calculator().get_cpool())
    n_pruned = len(pruned.get_correlations_calculator().get_cpool())
    assert n_pruned < n_full

    # pruned model must use the pruned ccalc as input and reproduce the full model
    sset = StructuresSet.from_file(sset_filepath)
    for s in sset.get_structures()[:5]:
        assert pruned.predict(s) == pytest.approx(full.predict(s), abs=1e-9)
    assert isinstance(pruned.estimator, Pipeline)
    assert isinstance(pruned.estimator.steps[0][1], ColumnExpander)

    log = (prefix.parent / "prop_0_log.md").read_text()
    assert "Consistency of pruned and full models" in log
    assert "WARNING" not in log


def test_build_model_from_sset_no_prune(sset_filepath, tmp_path):
    prefix = tmp_path / "prop"
    _build(sset_filepath, prefix, prune=False)
    names = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("prop"))
    assert not any("pruned" in n for n in names)
    assert (tmp_path / "prop_0_log.md").exists()
    assert (tmp_path / "prop_1_full_model.pickle").exists()


def test_build_model_from_sset_default_threshold_keeps_all(sset_filepath, tmp_path):
    prefix = tmp_path / "prop"
    _build(
        sset_filepath,
        prefix,
        regression_model={"class": "Ridge", "args": {"alpha": 1e-8}},
        selection_model={"args": {"cv": 3, "max_iter": 10000}},
        prune=False,
    )
    full = Model(filepath=f"{prefix}_1_full_model.pickle")
    sel = full.estimator.named_steps["selector"]
    assert sel.get_support().all()


def test_selection_report_explains_counts(data):
    from clusterx.cli.build_nonlinear_model import fit_nonlinear_pipeline

    x, y = data
    rep = MarkdownReport(echo=False)
    fit = fit_nonlinear_pipeline(
        x,
        y,
        regression_model={"class": "Ridge", "args": {"alpha": 1e-8}},
        selection_model={"args": {"cv": 3, "max_iter": 10000}},
        selection_threshold=1e-2,
        selection_must_include=[4],
        nonlinear_transformation=POLY2,
        report=rep,
    )
    text = "\n".join(rep._lines)
    n_sel = int(fit["pipeline"].named_steps["selector"].get_support().sum())
    assert "Feature selection (selector -> regressor)" in text
    assert f"| selected features entering the regressor | {n_sel} |" in text
    assert "index before selection" in text
    assert "selection_must_include [4]" in text
