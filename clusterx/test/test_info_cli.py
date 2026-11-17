"""Artifact introspection, CLI discovery, and backwards-compatible command names."""

import hashlib
import pickle
import re
import subprocess

import numpy as np
import pytest
import toml
from ase import Atoms
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from clusterx.cli.print_info_ccalc import print_info_ccalc
from clusterx.cli.print_info_comat import print_info_comat
from clusterx.cli.print_info_derivative_structures import print_info_derivative_structures
from clusterx.cli.print_info_model import print_info_model
from clusterx.cli.print_info_plat import print_info_plat
from clusterx.cli.print_info_sset import print_info_sset
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.derivative_structures import DerivativeStructures
from clusterx.estimators.column_expander import ColumnExpander
from clusterx.model import Model
from clusterx.parent_lattice import ParentLattice
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell


@pytest.fixture
def artifacts(tmp_path):
    plat = ParentLattice(Atoms("Cu", cell=[1, 1, 1], pbc=True), numbers=[[29, 79]])
    cpool = ClustersPool(plat, npoints=[1, 2], radii=[0, 1.1])
    ccalc = CorrelationsCalculator("trigonometric", plat, cpool)
    sset = StructuresSet(plat)
    scell = SuperCell(plat, [2, 1, 1])
    for sigmas in [(0, 0), (0, 1)]:
        sset.add_structure(scell.gen_structure(sigmas=sigmas))
    sset.set_property_values("reference_energy", [1.0, 3.0], update_db=False)
    sset.set_property_values("mask", ["selected", "other"], update_db=False)
    comat = ccalc.get_correlation_matrix(sset)
    model = Model(ccalc, "energy", ecis=np.ones(len(cpool)))
    dss = DerivativeStructures.from_structures_set(sset)
    dss.metadata["source"] = "test dataset"
    plat.serialize(filepath=str(tmp_path / "plat.json"))
    cpool.serialize(filepath=str(tmp_path / "cpool.json"))
    ccalc.serialize(filepath=str(tmp_path / "ccalc.pickle"))
    model.serialize(filepath=str(tmp_path / "model.pickle"))
    for extension in ("xyz", "db", "json"):
        sset.serialize(filepath=str(tmp_path / f"sset.{extension}"))
    dss.serialize(tmp_path / "dss.pickle")
    np.savez(tmp_path / "comat.npz", comat=comat, property_energy=[np.nan, 2.0],
             property_gap=[0.0, 1.0], sample_ids=[3, 7])
    return tmp_path, model, sset


@pytest.mark.parametrize("command,filename,expected", [
    ("print_info_plat", "plat.json", "Allowed species (in sigma order)"),
    ("print_plat_info", "plat.json", "Allowed species (in sigma order)"),
    ("print_info_cpool", "cpool.json", "Number of clusters:"),
    ("print_cpool_info", "cpool.json", "Number of clusters:"),
    ("print_info_comat", "comat.npz", "Sample count matches"),
    ("print_info_model", "model.pickle", "Largest stored coefficients"),
    ("print_info_ccalc", "ccalc.pickle", "Cached orbit sets"),
    ("print_info_sset", "sset.xyz", "Number of structures"),
    ("print_info_derivative_structures", "dss.pickle", "Number of configurations: 2"),
    ("print_info_dss", "dss.pickle", "Number of configurations: 2"),
    ("summarize_derivative_structures", "dss.pickle", "Number of configurations: 2"),
    ("summarize_dsgenerator", "dss.pickle", "Number of configurations: 2"),
])
def test_info_commands(command, filename, expected, artifacts):
    directory, _, _ = artifacts
    paths = [path for path in directory.iterdir() if path.is_file()]
    before = {path: hashlib.sha256(path.read_bytes()).digest() for path in paths}
    result = subprocess.run(["cell", command, str(directory / filename)], cwd=directory,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout
    if command != "print_info_comat":
        assert "Supported species" in result.stdout
    assert all(hashlib.sha256(path.read_bytes()).digest() == digest for path, digest in before.items())


def test_info_comat_properties(artifacts, capsys):
    directory, _, _ = artifacts
    print_info_comat(str(directory / "comat.npz"))
    output = capsys.readouterr().out
    assert re.search(r"\| Samples\s*\| 2\s*\|", output)
    assert re.search(r"\| Features\s*\| 2\s*\|", output)
    assert "Number of properties: 2" in output
    assert "energy" in output and "gap" in output and "sample_ids" in output
    assert "NaN values" in output


@pytest.mark.parametrize("payload,message", [
    ({"other": np.zeros((2, 3))}, "must contain a 'comat' array"),
    ({"comat": np.zeros(3)}, "two-dimensional numeric matrix"),
])
def test_invalid_comat(tmp_path, payload, message):
    path = tmp_path / "invalid.npz"
    np.savez(path, **payload)
    with pytest.raises(ValueError, match=message):
        print_info_comat(str(path))


def test_empty_comat_and_object_property(tmp_path, capsys):
    path = tmp_path / "empty.npz"
    np.savez(path, comat=np.empty((0, 3)), property_notes=np.array([], dtype=object))
    print_info_comat(str(path))
    output = capsys.readouterr().out
    assert "Number of properties: 1" in output
    assert "notes" in output and "not loaded" in output


def test_comat_mismatched_property_count(tmp_path, capsys):
    path = tmp_path / "mismatch.npz"
    np.savez(path, comat=np.zeros((2, 3)), property_energy=[1.0])
    print_info_comat(str(path))
    output = capsys.readouterr().out
    assert "NO (expected 2)" in output


def test_supported_species_per_sublattice(tmp_path, capsys):
    atoms = Atoms(["Cu", "H", "He"], positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                  cell=[3, 2, 8], pbc=[True, True, False])
    plat = ParentLattice(atoms, numbers=[[29, 79], [1, 0, 3], [2]])
    path = tmp_path / "plat.json"
    plat.serialize(filepath=str(path))
    print_info_plat(str(path))
    output = capsys.readouterr().out
    assert re.search(r"\| Sublattices\s*\| 3\s*\|", output)
    assert re.search(r"\|\s*\d+\s*\| 1\s*\| 2\s*\| Cu, Au\s*\|", output)
    assert re.search(r"\|\s*\d+\s*\| 1\s*\| 3\s*\| H, X, Li\s*\|", output)
    assert re.search(r"\|\s*\d+\s*\| 1\s*\| 1\s*\| He\s*\|", output)
    assert "spectator" in output and "False" in output


@pytest.mark.parametrize("extension", ["db", "json", "xyz"])
def test_info_sset_formats(extension, artifacts, capsys):
    directory, _, _ = artifacts
    print_info_sset(str(directory / f"sset.{extension}"))
    output = capsys.readouterr().out
    assert re.search(r"\| Number of structures\s*\| 2\s*\|", output)
    assert "reference_energy" in output and "mask" in output
    assert "Number of supercell shapes: 1" in output


def test_info_dss_empty(artifacts, capsys):
    directory, model, _ = artifacts
    dss = DerivativeStructures(model.corrc.get_cpool().get_plat())
    dss.serialize(directory / "empty.pickle")
    print_info_derivative_structures(str(directory / "empty.pickle"))
    output = capsys.readouterr().out
    assert "Number of configurations: 0" in output
    assert "Number of masks: 0" in output
    assert "Number of supercell shapes: 0" in output


def test_info_does_not_generate_orbits(artifacts, monkeypatch, capsys):
    directory, _, _ = artifacts

    def unexpected(*args, **kwargs):
        raise AssertionError("Inspection must not generate cluster orbits")

    monkeypatch.setattr(CorrelationsCalculator, "get_cluster_orbits_for_scell", unexpected)
    print_info_ccalc(str(directory / "ccalc.pickle"))
    print_info_model(str(directory / "model.pickle"))
    assert "trigonometric" in capsys.readouterr().out


def test_info_pipeline_and_standalone_model(artifacts, capsys):
    directory, model, _ = artifacts
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.2], [0.3, 0.8]])
    y = x[:, 0] + 2 * x[:, 1] + x[:, 0] * x[:, 1]
    pipeline = Pipeline([
        ("expander", ColumnExpander(3, [0, 2])),
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_regression, k=2)),
        ("regressor", LinearRegression()),
    ]).fit(x, y)
    fitted = Model(model.corrc, "energy", estimator=pipeline)
    path = directory / "standalone.pickle"
    with path.open("wb") as handle:
        pickle.dump(fitted, handle)
    print_info_model(str(path))
    output = capsys.readouterr().out
    for text in ("embedded in model", "ColumnExpander", "PolynomialFeatures", "StandardScaler",
                 "SelectKBest", "LinearRegression", "kept_columns", "degree", "Nonzero"):
        assert text in output
    assert re.search(r"\| n_jobs\s*\| None\s*\|", output)
    assert "np.int64(" not in output


def test_info_bagging_model(artifacts, capsys):
    directory, model, _ = artifacts
    x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    ensemble = BaggingRegressor(n_estimators=3, random_state=0).fit(x, np.arange(4.))
    model = Model(model.corrc, "energy", estimator=ensemble)
    model.serialize(filepath=str(directory / "ensemble.pickle"))
    print_info_model(str(directory / "ensemble.pickle"))
    output = capsys.readouterr().out
    assert "BaggingRegressor" in output
    assert re.search(r"\| Fitted ensemble members\s*\| 3\s*\|", output)


def test_model_uses_companion_calculator(artifacts, capsys):
    directory, model, _ = artifacts
    model.corrc = None
    path = directory / "model.pickle"
    with path.open("wb") as handle:
        pickle.dump(model, handle)
    print_info_model(str(path))
    output = capsys.readouterr().out
    # the filename 'model_CCALC.pickle' might be truncated due to limit of 120
    # characters in clusterx.cli.info_utils.format_value
    #assert "model_CCALC.pickle" in output
    assert "Correlation calculator" in output
    assert "trigonometric" in output


def test_multioutput_model_coefficients(artifacts, capsys):
    directory, model, _ = artifacts
    model = Model(model.corrc, ["energy", "gap"], ecis=[[1.0, 0.0], [0.0, 2.0]])
    model.serialize(filepath=str(directory / "multi.pickle"))
    print_info_model(str(directory / "multi.pickle"))
    output = capsys.readouterr().out
    assert "energy" in output and "gap" in output
    assert re.search(r"\| Shape\s*\| \(2, 2\)\s*\|", output)
    assert re.search(r"\| Nonzero\s*\| 2\s*\|", output)


@pytest.mark.parametrize("command", [print_info_model, print_info_ccalc, print_info_derivative_structures])
def test_wrong_pickle_type(command, tmp_path):
    path = tmp_path / "wrong.pickle"
    path.write_bytes(pickle.dumps({"not": "a CELL object"}))
    with pytest.raises(TypeError, match="Expected"):
        command(str(path))


def test_info_toml_dispatch(artifacts):
    directory, _, _ = artifacts
    config = {"print_info_plat": {"plat_filepath": str(directory / "plat.json")},
              "print_info_comat": {"comat_filepath": str(directory / "comat.npz")},
              "print_info_model": {"model_filepath": str(directory / "model.pickle")},
              "print_info_ccalc": {"ccalc_filepath": str(directory / "ccalc.pickle")},
              "print_info_sset": {"sset_filepath": str(directory / "sset.xyz")},
              "print_info_derivative_structures": {"dss_filepath": str(directory / "dss.pickle")}}
    path = directory / "info.toml"
    path.write_text(toml.dumps(config))
    result = subprocess.run(["cell", "--input_file", str(path)], cwd=directory, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    for title in ("ParentLattice information", "Correlation matrix information", "CE model information",
                  "CorrelationsCalculator information", "StructuresSet information", "DerivativeStructures summary"):
        assert title in result.stdout
