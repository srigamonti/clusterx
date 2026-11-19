"""Test utility functions for CELL."""

import json
import importlib
import subprocess
import sys

import matplotlib.axes as axes_module
import matplotlib.figure as figure_module
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pytest
import toml
from ase.build import bulk

from clusterx.cli.plot_columns import plot_columns
from clusterx.cli.ssets_equivalence_check import (
    ssets_equivalence_check as cli_ssets_equivalence_check,
)
from clusterx.parent_lattice import ParentLattice
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
from clusterx.utils import (
    Exponential,
    ListDict,
    PolynomialBasis,
    PolynomialFunction,
    RunningAverage,
    atat_to_cell,
    decorate_supercell,
    lattice_wrap_index,
    list_integer_named_folders,
    normalize_shape_input,
    parent_lattice_to_atat,
    ssets_equivalence_check,
    take_tile,
    transformation_containing_sphere_diag,
)


def test_RunningAverage():
    rave = RunningAverage(n=10)
    rave.update(2)
    np.testing.assert_approx_equal(rave.get_average(), 2)
    np.testing.assert_approx_equal(rave.update_and_get_average(1), 1.5)
    for _ in range(10):
        rave.update(7)
    np.testing.assert_approx_equal(rave.get_average(), 7)


def test_ase_make_supercell_error():
    from ase.atoms import Atoms
    from ase.build import make_supercell

    prim = Atoms(
        symbols="Cu",
        pbc=True,
        cell=[[0.0, 1.805, 1.805], [1.805, 0.0, 1.805], [1.805, 1.805, 0.0]],
    )

    _ = make_supercell(
        prim,
        P=[[2, 2, -2], [2, -2, 2], [-2, 2, 2]],
        wrap=True,
        tol=1e-05,
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


def test_ListDict():
    list_dict = ListDict()
    n = 1000
    for i in range(n):
        list_dict.add(i)
    assert [i for i in list_dict]  # __iter__
    assert len(list_dict) == n  # __len__
    assert list_dict.choice() >= 0
    for i in range(n):
        assert i in list_dict  # __contains__
        list_dict.remove(i)


def test_decorate_supercell(parent_lattice):
    scell = SuperCell(parent_lattice, np.diag([2, 2, 2]))
    structure = scell.gen_random_structure()
    atoms = structure.get_atoms()
    # slightly offset the positions, to check tolerance of the decoration
    np.random.seed(42)
    dpos = np.random.rand(*atoms.get_positions().shape) * 1e-6
    atoms.positions += dpos
    decorated = decorate_supercell(scell, atoms)
    np.testing.assert_array_equal(decorated.get_cell(), scell.get_cell())
    np.testing.assert_array_equal(decorated.sigmas, structure.sigmas)


@pytest.mark.parametrize(
    "r,cell,full_sphere,p",
    [
        (1.0, np.eye(3), False, [1, 1, 1]),
        (2.0, np.eye(3), False, [2, 2, 2]),
        (2.0, np.diag([2, 1, 1]), False, [1, 2, 2]),
        (2.0, np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]]), False, [2, 2, 1]),
        (1.0, np.eye(3), True, [3, 3, 3]),
    ],
)
def test_containing_supercell(r, cell, full_sphere, p):
    p_result = transformation_containing_sphere_diag(r, cell, full_sphere)
    np.testing.assert_array_equal(p_result, np.diag(p))


@pytest.mark.parametrize(
    "grid_shape,i_grid,p_reduced",
    [
        ((10, 10, 10, 2), np.array([0, 0, 0, 0]), (3, 3, 3)),
        (
            (10, 10, 10, 2),
            np.array([3, 3, 3, 0]),
            (5, 5, 5),
        ),  # TODO: figure out what the result should be for even p
        ((10, 10, 10, 3), np.array([9, 9, 9, 2]), (3, 3, 3)),
        ((10, 10, 1), np.array([9, 9, 0]), (3, 3)),
    ],
)
def test_take_tile(grid_shape, i_grid, p_reduced):
    grid = np.arange(np.prod(grid_shape)).reshape(grid_shape)
    grid_reduced, i_new = take_tile(grid, i_grid, p_reduced)
    np.testing.assert_array_equal(grid_reduced.shape, list(p_reduced) + [grid_shape[-1]])
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


def test_list_integer_named_folders_preserves_zero_padded_names(tmp_path):
    for name in ["010", "001", "abc", "002", "0"]:
        (tmp_path / name).mkdir()

    assert list_integer_named_folders(tmp_path) == ["0", "001", "002", "010"]


def test_list_integer_named_folders_preserves_zero_padded_names_with_affixes(tmp_path):
    for name in ["run_010_done", "run_001_done", "run_old_done", "001_done"]:
        (tmp_path / name).mkdir()

    assert list_integer_named_folders(tmp_path, prefix="run_", suffix="_done") == ["run_001_done", "run_010_done"]


def _make_equivalence_sset(parent_lattice, sigmas_list):
    scell = SuperCell(parent_lattice, np.diag([2, 1, 1]))
    sset = StructuresSet(parent_lattice)
    for sigmas in sigmas_list:
        sset.add_structure(scell.gen_structure(sigmas=np.array(sigmas)))
    return sset


def test_ssets_equivalence_check_self(parent_lattice):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 1], [0, 0]])

    equivalent = ssets_equivalence_check(sset)

    assert {k: v.tolist() for k, v in equivalent.items()} == {
        0: [0, 1],
        2: [2],
    }


def test_ssets_equivalence_check_two_sets(parent_lattice):
    sset1 = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset2 = _make_equivalence_sset(parent_lattice, [[0, 1], [0, 0], [1, 1]])

    equivalent = ssets_equivalence_check(sset1, sset2)

    assert {k: v.tolist() for k, v in equivalent.items()} == {
        0: [0],
        1: [1],
    }


def test_cli_ssets_equivalence_check_two_sets(parent_lattice, tmp_path, capsys):
    sset1 = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset2 = _make_equivalence_sset(parent_lattice, [[0, 1], [0, 0], [1, 1]])
    sset1_filepath = tmp_path / "sset1.db"
    sset2_filepath = tmp_path / "sset2.db"
    sset1.serialize(filepath=str(sset1_filepath), ase_db_type="db", overwrite=True)
    sset2.serialize(filepath=str(sset2_filepath), ase_db_type="db", overwrite=True)

    cli_ssets_equivalence_check(str(sset1_filepath), str(sset2_filepath))

    assert "{0: [0], 1: [1]}" in capsys.readouterr().out


def test_plot_columns_accepts_serialized_structures_set(parent_lattice, tmp_path):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="x",
        property_vals=[0.0, 1.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="y",
        property_vals=[2.0, 3.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    figdata_filepath = tmp_path / "plot_data.json"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    plot_columns(
        source_filepath=str(sset_filepath),
        show_plot=False,
        figdata_fname=str(figdata_filepath),
        series=[{"xcolumn": "x", "ycolumn": "y"}],
    )

    with open(figdata_filepath, encoding="utf-8") as f:
        plot_data = json.load(f)
    assert plot_data["series"][0]["x"] == [0.0, 1.0]
    assert plot_data["series"][0]["y"] == [2.0, 3.0]


def test_plot_columns_uses_top_level_defaults_without_series(parent_lattice, tmp_path):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="x",
        property_vals=[0.0, 1.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="y",
        property_vals=[2.0, 3.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    figdata_filepath = tmp_path / "plot_data.json"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    plot_columns(
        source_filepath=str(sset_filepath),
        xcolumn="x",
        ycolumn="y",
        show_plot=False,
        figdata_fname=str(figdata_filepath),
    )

    with open(figdata_filepath, encoding="utf-8") as f:
        plot_data = json.load(f)
    assert len(plot_data["series"]) == 1
    assert plot_data["series"][0]["xcolumn"] == "x"
    assert plot_data["series"][0]["ycolumn"] == "y"
    assert plot_data["series"][0]["x"] == [0.0, 1.0]
    assert plot_data["series"][0]["y"] == [2.0, 3.0]


def test_plot_columns_applies_yfactor(parent_lattice, tmp_path):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="x",
        property_vals=[0.0, 1.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="y",
        property_vals=[2.0, 3.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    figdata_filepath = tmp_path / "plot_data.json"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    plot_columns(
        source_filepath=str(sset_filepath),
        yfactor=10.0,
        show_plot=False,
        figdata_fname=str(figdata_filepath),
        series=[
            {"xcolumn": "x", "ycolumn": "y"},
            {"xcolumn": "x", "ycolumn": "y", "yfactor": -1.0},
        ],
    )

    with open(figdata_filepath, encoding="utf-8") as f:
        plot_data = json.load(f)
    assert plot_data["yfactor"] == 10.0
    assert plot_data["series"][0]["yfactor"] == 10.0
    assert plot_data["series"][0]["y"] == [20.0, 30.0]
    assert plot_data["series"][0]["y_samples"] == [[20.0], [30.0]]
    assert plot_data["series"][1]["yfactor"] == -1.0
    assert plot_data["series"][1]["y"] == [-2.0, -3.0]


def test_plot_columns_boxplot_alpha_applies_to_whiskers_and_caps(parent_lattice, tmp_path, monkeypatch):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="x",
        property_vals=[0.0, 1.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="y",
        property_vals=[1.0, 2.0, 3.0, 4.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    captured = {}
    original_boxplot = axes_module.Axes.boxplot

    def capture_boxplot(self, *args, **kwargs):
        bp = original_boxplot(self, *args, **kwargs)
        captured["bp"] = bp
        return bp

    monkeypatch.setattr(axes_module.Axes, "boxplot", capture_boxplot)

    plot_columns(
        source_filepath=str(sset_filepath),
        show_plot=False,
        series=[{"xcolumn": "x", "ycolumn": "y", "ymode": "boxplot", "alpha": 0.25}],
    )

    bp = captured["bp"]
    assert bp["boxes"][0].get_alpha() == pytest.approx(0.25)
    assert bp["whiskers"][0].get_alpha() == pytest.approx(0.25)
    assert bp["caps"][0].get_alpha() == pytest.approx(0.25)
    assert bp["medians"][0].get_alpha() == pytest.approx(0.25)


def test_plot_columns_grid_lines(parent_lattice, tmp_path, monkeypatch):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="x",
        property_vals=[0.0, 1.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="y",
        property_vals=[2.0, 3.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    figdata_filepath = tmp_path / "plot_data.json"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    captured = {}
    real_close = plt.close

    def capture_close(fig=None):
        captured["fig"] = fig if fig is not None else plt.gcf()

    monkeypatch.setattr(plt, "close", capture_close)
    try:
        plot_columns(
            source_filepath=str(sset_filepath),
            grid=True,
            grid_axis="x",
            grid_style={"linestyle": ":", "linewidth": 0.7, "alpha": 0.4},
            show_plot=False,
            figdata_fname=str(figdata_filepath),
            series=[{"xcolumn": "x", "ycolumn": "y"}],
        )

        fig = captured["fig"]
        ax = fig.axes[0]
        x_gridlines = [line for line in ax.xaxis.get_gridlines() if line.get_visible()]
        y_gridlines = [line for line in ax.yaxis.get_gridlines() if line.get_visible()]
        assert len(x_gridlines) > 0
        assert len(y_gridlines) == 0
        assert x_gridlines[0].get_linestyle() == ":"
        assert x_gridlines[0].get_linewidth() == 0.7
        assert x_gridlines[0].get_alpha() == 0.4

        with open(figdata_filepath, encoding="utf-8") as f:
            plot_data = json.load(f)
        assert plot_data["grid"] is True
        assert plot_data["grid_axis"] == "x"
        assert plot_data["grid_which"] == "major"
        assert plot_data["grid_style"] == {
            "linestyle": ":",
            "linewidth": 0.7,
            "alpha": 0.4,
        }
    finally:
        if "fig" in captured:
            real_close(captured["fig"])


def test_plot_columns_target_vs_predictions_loglog_axes(parent_lattice, tmp_path, monkeypatch):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="target",
        property_vals=[1.0, 10.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="prediction",
        property_vals=[2.0, 8.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    figdata_filepath = tmp_path / "plot_data.json"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    captured = {}
    real_close = plt.close

    def capture_close(fig=None):
        captured["fig"] = fig if fig is not None else plt.gcf()

    monkeypatch.setattr(plt, "close", capture_close)
    try:
        plot_columns(
            source_filepath=str(sset_filepath),
            target_vs_predictions=True,
            loglog=True,
            show_plot=False,
            figdata_fname=str(figdata_filepath),
            series=[{"xcolumn": "target", "ycolumn": "prediction"}],
        )

        fig = captured["fig"]
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        np.testing.assert_allclose(ax.get_xlim(), ax.get_ylim())
        assert ax.get_box_aspect() == 1

        parity_lines = [line for line in ax.lines if line.get_label() == "y=x"]
        assert len(parity_lines) == 1
        np.testing.assert_allclose(
            parity_lines[0].get_xdata(),
            parity_lines[0].get_ydata(),
        )

        with open(figdata_filepath, encoding="utf-8") as f:
            plot_data = json.load(f)
        assert plot_data["target_vs_predictions"] is True
        assert plot_data["loglog"] is True
        assert plot_data["axis_limits"][0] < 1.0
        assert plot_data["axis_limits"][1] > 10.0
    finally:
        if "fig" in captured:
            real_close(captured["fig"])


def test_plot_columns_target_vs_predictions_uses_width_for_square_figure(parent_lattice, tmp_path, monkeypatch):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="target",
        property_vals=[1.0, 2.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="prediction",
        property_vals=[1.1, 1.9],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    captured = {}
    real_close = plt.close

    def capture_close(fig=None):
        captured["fig"] = fig if fig is not None else plt.gcf()

    monkeypatch.setattr(plt, "close", capture_close)
    try:
        plot_columns(
            source_filepath=str(sset_filepath),
            target_vs_predictions=True,
            width=7.25,
            height=3.0,
            show_plot=False,
            series=[{"xcolumn": "target", "ycolumn": "prediction"}],
        )

        fig = captured["fig"]
        np.testing.assert_allclose(fig.get_size_inches(), [7.25, 7.25])
    finally:
        if "fig" in captured:
            real_close(captured["fig"])


def test_plot_columns_axis_name_and_ticklabel_visibility_toggles(parent_lattice, tmp_path, monkeypatch):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="x",
        property_vals=[0.0, 1.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="y",
        property_vals=[1.0, 2.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    captured = {}
    real_close = plt.close

    def capture_close(fig=None):
        captured["fig"] = fig if fig is not None else plt.gcf()

    monkeypatch.setattr(plt, "close", capture_close)
    try:
        plot_columns(
            source_filepath=str(sset_filepath),
            xlabel="X axis",
            ylabel="Y axis",
            show_xlabel=False,
            show_ylabel=False,
            show_xticklabels=False,
            show_yticklabels=False,
            show_plot=False,
            series=[{"xcolumn": "x", "ycolumn": "y"}],
        )

        fig = captured["fig"]
        ax = fig.axes[0]
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
        assert all(not tick.get_visible() for tick in ax.get_xticklabels())
        assert all(not tick.get_visible() for tick in ax.get_yticklabels())
    finally:
        if "fig" in captured:
            real_close(captured["fig"])


def test_plot_columns_can_place_axes_on_top_and_right(parent_lattice, tmp_path, monkeypatch):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="x",
        property_vals=[0.0, 1.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="y",
        property_vals=[1.0, 2.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    figdata_filepath = tmp_path / "plot_data.json"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    captured = {}
    real_close = plt.close

    def capture_close(fig=None):
        captured["fig"] = fig if fig is not None else plt.gcf()

    monkeypatch.setattr(plt, "close", capture_close)
    try:
        plot_columns(
            source_filepath=str(sset_filepath),
            xlabel="X axis",
            ylabel="Y axis",
            xaxis_top=True,
            yaxis_right=True,
            show_plot=False,
            figdata_fname=str(figdata_filepath),
            series=[{"xcolumn": "x", "ycolumn": "y"}],
        )

        fig = captured["fig"]
        fig.canvas.draw()
        ax = fig.axes[0]

        assert ax.xaxis.get_label_position() == "top"
        assert ax.yaxis.get_label_position() == "right"

        x_tick = ax.xaxis.get_major_ticks()[0]
        y_tick = ax.yaxis.get_major_ticks()[0]
        assert x_tick.tick1line.get_visible() is False
        assert x_tick.tick2line.get_visible() is True
        assert x_tick.label1.get_visible() is False
        assert x_tick.label2.get_visible() is True
        assert y_tick.tick1line.get_visible() is False
        assert y_tick.tick2line.get_visible() is True
        assert y_tick.label1.get_visible() is False
        assert y_tick.label2.get_visible() is True

        with open(figdata_filepath, encoding="utf-8") as f:
            plot_data = json.load(f)
        assert plot_data["xaxis_top"] is True
        assert plot_data["yaxis_right"] is True
    finally:
        if "fig" in captured:
            real_close(captured["fig"])


def test_plot_columns_sets_custom_axis_ticks(parent_lattice, tmp_path, monkeypatch):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="x",
        property_vals=[0.0, 1.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="y",
        property_vals=[1.0, 2.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    figdata_filepath = tmp_path / "plot_data.json"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    captured = {}
    real_close = plt.close

    def capture_close(fig=None):
        captured["fig"] = fig if fig is not None else plt.gcf()

    monkeypatch.setattr(plt, "close", capture_close)
    try:
        plot_columns(
            source_filepath=str(sset_filepath),
            xticks=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            yticks=[-1.0, 0.0, 1.0, 2.0],
            show_plot=False,
            figdata_fname=str(figdata_filepath),
            series=[{"xcolumn": "x", "ycolumn": "y"}],
        )

        fig = captured["fig"]
        ax = fig.axes[0]
        np.testing.assert_allclose(ax.get_xticks(), [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        np.testing.assert_allclose(ax.get_yticks(), [-1.0, 0.0, 1.0, 2.0])

        with open(figdata_filepath, encoding="utf-8") as f:
            plot_data = json.load(f)
        assert plot_data["xticks"] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        assert plot_data["yticks"] == [-1.0, 0.0, 1.0, 2.0]
    finally:
        if "fig" in captured:
            real_close(captured["fig"])


def test_plot_columns_savefig_uses_configured_dpi(parent_lattice, tmp_path, monkeypatch):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="x",
        property_vals=[0.0, 1.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="y",
        property_vals=[1.0, 2.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    output_filepath = tmp_path / "plot.png"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    captured = {}
    real_savefig = figure_module.Figure.savefig

    def capture_savefig(self, fname, *args, **kwargs):
        captured["fname"] = fname
        captured["dpi"] = kwargs.get("dpi")
        return real_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(figure_module.Figure, "savefig", capture_savefig)

    plot_columns(
        source_filepath=str(sset_filepath),
        show_plot=False,
        savefig=str(output_filepath),
        dpi=180,
        series=[{"xcolumn": "x", "ycolumn": "y"}],
    )

    assert captured["fname"] == str(output_filepath)
    assert captured["dpi"] == 180.0
    assert output_filepath.exists()


@pytest.mark.parametrize("show_legend_frame", [True, False])
def test_plot_columns_controls_legend_frame_visibility(parent_lattice, tmp_path, monkeypatch, show_legend_frame):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="x",
        property_vals=[0.0, 1.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="y",
        property_vals=[1.0, 2.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    captured = {}
    real_close = plt.close

    def capture_close(fig=None):
        captured["fig"] = fig if fig is not None else plt.gcf()

    monkeypatch.setattr(plt, "close", capture_close)
    try:
        plot_columns(
            source_filepath=str(sset_filepath),
            legend=True,
            show_legend_frame=show_legend_frame,
            show_plot=False,
            series=[{"xcolumn": "x", "ycolumn": "y", "label": "fit"}],
        )

        fig = captured["fig"]
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_frame_on() is show_legend_frame
    finally:
        if "fig" in captured:
            real_close(captured["fig"])


@pytest.mark.parametrize(
    "legend_location,expected_loc",
    [
        ("upper left", "upper left"),
        ("upper center", "upper center"),
        ("upper right", "upper right"),
        ("lower_center", "lower center"),
    ],
)
def test_plot_columns_controls_legend_location(parent_lattice, tmp_path, monkeypatch, legend_location, expected_loc):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="x",
        property_vals=[0.0, 1.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="y",
        property_vals=[1.0, 2.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    captured = {}
    original_legend = axes_module.Axes.legend

    def capture_legend(self, *args, **kwargs):
        captured["loc"] = kwargs.get("loc")
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(axes_module.Axes, "legend", capture_legend)

    plot_columns(
        source_filepath=str(sset_filepath),
        legend=True,
        legend_location=legend_location,
        show_plot=False,
        series=[{"xcolumn": "x", "ycolumn": "y", "label": "fit"}],
    )

    assert captured["loc"] == expected_loc


def test_plot_columns_controls_legend_handle_spacing(parent_lattice, tmp_path, monkeypatch):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="x",
        property_vals=[0.0, 1.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="y",
        property_vals=[1.0, 2.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    captured = {}
    original_legend = axes_module.Axes.legend

    def capture_legend(self, *args, **kwargs):
        captured["handlelength"] = kwargs.get("handlelength")
        captured["handletextpad"] = kwargs.get("handletextpad")
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(axes_module.Axes, "legend", capture_legend)

    plot_columns(
        source_filepath=str(sset_filepath),
        legend=True,
        legend_handlelength=3.0,
        legend_handletextpad=0.7,
        show_plot=False,
        series=[{"xcolumn": "x", "ycolumn": "y", "label": "fit"}],
    )

    assert captured["handlelength"] == 3.0
    assert captured["handletextpad"] == 0.7


def test_plot_columns_rejects_negative_legend_handle_spacing(parent_lattice, tmp_path):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="x",
        property_vals=[0.0, 1.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="y",
        property_vals=[1.0, 2.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    with pytest.raises(ValueError, match="legend_handlelength must be non-negative"):
        plot_columns(
            source_filepath=str(sset_filepath),
            legend_handlelength=-0.1,
            show_plot=False,
            series=[{"xcolumn": "x", "ycolumn": "y", "label": "fit"}],
        )

    with pytest.raises(ValueError, match="legend_handletextpad must be non-negative"):
        plot_columns(
            source_filepath=str(sset_filepath),
            legend_handletextpad=-0.1,
            show_plot=False,
            series=[{"xcolumn": "x", "ycolumn": "y", "label": "fit"}],
        )


def test_plot_columns_rejects_invalid_legend_location(parent_lattice, tmp_path):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="x",
        property_vals=[0.0, 1.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="y",
        property_vals=[1.0, 2.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    with pytest.raises(ValueError, match="legend_location must be one of"):
        plot_columns(
            source_filepath=str(sset_filepath),
            legend=True,
            legend_location="top right",
            show_plot=False,
            series=[{"xcolumn": "x", "ycolumn": "y", "label": "fit"}],
        )


def test_plot_columns_target_vs_predictions_symlog_axes(parent_lattice, tmp_path, monkeypatch):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="target",
        property_vals=[-10.0, 0.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="prediction",
        property_vals=[-8.0, -0.5],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    figdata_filepath = tmp_path / "plot_data.json"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    captured = {}
    real_close = plt.close

    def capture_close(fig=None):
        captured["fig"] = fig if fig is not None else plt.gcf()

    monkeypatch.setattr(plt, "close", capture_close)
    try:
        plot_columns(
            source_filepath=str(sset_filepath),
            target_vs_predictions=True,
            symlog=True,
            symlog_linthresh=0.2,
            show_plot=False,
            figdata_fname=str(figdata_filepath),
            series=[{"xcolumn": "target", "ycolumn": "prediction"}],
        )

        fig = captured["fig"]
        ax = fig.axes[0]
        assert ax.get_xscale() == "symlog"
        assert ax.get_yscale() == "symlog"
        np.testing.assert_allclose(ax.get_xlim(), ax.get_ylim())
        assert ax.get_box_aspect() == 1

        parity_lines = [line for line in ax.lines if line.get_label() == "y=x"]
        assert len(parity_lines) == 1
        np.testing.assert_allclose(
            parity_lines[0].get_xdata(),
            parity_lines[0].get_ydata(),
        )

        with open(figdata_filepath, encoding="utf-8") as f:
            plot_data = json.load(f)
        assert plot_data["symlog"] is True
        assert plot_data["symlog_linthresh"] == 0.2
        assert plot_data["axis_limits"][0] < -10.0
        assert plot_data["axis_limits"][1] > 0.0
    finally:
        if "fig" in captured:
            real_close(captured["fig"])


def test_plot_columns_loglog_rejects_non_positive_values(parent_lattice, tmp_path):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="target",
        property_vals=[0.0, 1.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="prediction",
        property_vals=[1.0, 2.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    try:
        with pytest.raises(ValueError, match="non-positive"):
            plot_columns(
                source_filepath=str(sset_filepath),
                target_vs_predictions=True,
                loglog=True,
                show_plot=False,
                series=[{"xcolumn": "target", "ycolumn": "prediction"}],
            )
    finally:
        plt.close("all")


def test_plot_columns_rejects_loglog_and_symlog_together(parent_lattice, tmp_path):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(
        property_name="target",
        property_vals=[1.0, 2.0],
        update_db=False,
    )
    sset.set_property_values(
        property_name="prediction",
        property_vals=[1.0, 2.0],
        update_db=False,
    )
    sset_filepath = tmp_path / "sset.db"
    sset.serialize(filepath=str(sset_filepath), ase_db_type="db", overwrite=True)

    try:
        with pytest.raises(ValueError, match="Use only one"):
            plot_columns(
                source_filepath=str(sset_filepath),
                target_vs_predictions=True,
                loglog=True,
                symlog=True,
                show_plot=False,
                series=[{"xcolumn": "target", "ycolumn": "prediction"}],
            )
    finally:
        plt.close("all")


@pytest.fixture
def presentation_source(parent_lattice, tmp_path):
    sset = _make_equivalence_sset(parent_lattice, [[1, 0], [0, 0]])
    sset.set_property_values(property_name="x", property_vals=[0.001, 0.004], update_db=False)
    sset.set_property_values(property_name="y", property_vals=[0.003, 0.002], update_db=False)
    path = tmp_path / "presentation.db"
    sset.serialize(filepath=str(path), ase_db_type="db", overwrite=True)
    return path


@pytest.fixture
def presentation_figures(monkeypatch):
    figures = []
    original = plt.subplots

    def create(*args, **kwargs):
        fig, ax = original(*args, **kwargs)
        figures.append(fig)
        return fig, ax

    monkeypatch.setattr(plt, "subplots", create)
    yield figures
    for fig in figures:
        plt.close(fig)


@pytest.mark.parametrize("extension", ["json", "npz", "txt", "dat"])
def test_plot_columns_x_and_y_scaling_exports(
    presentation_source, presentation_figures, tmp_path, monkeypatch, extension,
):
    original = presentation_source.read_bytes()
    module = importlib.import_module("clusterx.cli.plot_columns")
    source = module._load_plot_source(str(presentation_source))
    properties = source.configurations.copy(deep=True)
    monkeypatch.setattr(module, "_load_plot_source", lambda _path: source)
    output = tmp_path / f"coordinates.{extension}"
    plot_columns(str(presentation_source), xcolumn="x", ycolumn="y", xfactor=1000, yfactor=2000,
                 show_plot=False, figdata_fname=str(output),
                 series=[{}, {"xfactor": -100, "yfactor": 300}])
    ax = presentation_figures[-1].axes[0]
    np.testing.assert_allclose(ax.collections[0].get_offsets(), [[1, 6], [4, 4]])
    np.testing.assert_allclose(ax.collections[1].get_offsets(), [[-0.1, 0.9], [-0.4, 0.6]])
    assert presentation_source.read_bytes() == original
    assert source.configurations.equals(properties)
    if extension == "json":
        data = json.loads(output.read_text())
    elif extension == "npz":
        with np.load(output, allow_pickle=True) as archive:
            np.testing.assert_allclose(archive["x_0"], [1, 4])
            np.testing.assert_allclose(archive["y_1"], [0.9, 0.6])
            assert archive["xfactor_1"] == -100
            assert archive["yfactor_0"] == 2000
            data = json.loads(archive["plot_metadata"].item())
    else:
        np.testing.assert_allclose(np.loadtxt(output)[:, 1:], [[1, 6], [4, 4], [-0.1, 0.9], [-0.4, 0.6]])
        factors = json.loads(output.read_text().splitlines()[1].split("=", 1)[1])
        assert factors == [{"xfactor": 1000, "yfactor": 2000}, {"xfactor": -100, "yfactor": 300}]
        return
    assert data["xfactor"] == 1000
    assert data["series"][1]["xfactor"] == -100
    np.testing.assert_allclose(data["series"][0]["x"], [1, 4])
    np.testing.assert_allclose(data["series"][0]["y_samples"], [[6], [4]])


def test_plot_columns_scaled_parity_and_displayed_ticks(presentation_source, presentation_figures, tmp_path):
    path = tmp_path / "parity.json"
    plot_columns(str(presentation_source), xcolumn="x", ycolumn="y", xfactor=1000, yfactor=2000,
                 target_vs_predictions=True, xticks=[0, 3, 6], yticks=[0, 2, 4, 6],
                 show_plot=False, figdata_fname=str(path))
    ax = presentation_figures[-1].axes[0]
    np.testing.assert_allclose(ax.get_xlim(), [0.75, 6.25])
    np.testing.assert_allclose(ax.get_ylim(), ax.get_xlim())
    np.testing.assert_allclose(ax.get_xticks(), [0, 3, 6])
    line = next(line for line in ax.lines if line.get_label() == "y=x")
    np.testing.assert_allclose(line.get_xdata(), ax.get_xlim())
    np.testing.assert_allclose(line.get_ydata(), ax.get_ylim())
    np.testing.assert_allclose(json.loads(path.read_text())["axis_limits"], [0.75, 6.25])


def test_plot_columns_binning_uses_displayed_units(presentation_source, presentation_figures, tmp_path):
    path = tmp_path / "bins.json"
    plot_columns(str(presentation_source), xcolumn="x", ycolumn="y", xfactor=1000, yfactor=2000,
                 xrange=[-1, 7], show_plot=False, figdata_fname=str(path),
                 series=[{"binned": True, "xtol": 3, "ymode": "marker"}])
    data = json.loads(path.read_text())["series"][0]
    np.testing.assert_allclose(data["x"], [0, 3])
    np.testing.assert_allclose(data["y"], [6, 4])
    np.testing.assert_allclose(presentation_figures[-1].axes[0].get_xlim(), [-1, 7])


def test_plot_columns_four_value_legend_anchor(presentation_source, presentation_figures):
    anchor = [0.1, 0.8, 0.6, 0.1]
    plot_columns(str(presentation_source), xcolumn="x", ycolumn="y", legend_bbox_to_anchor=anchor,
                 legend_location="lower left", show_plot=False)
    ax = presentation_figures[-1].axes[0]
    bounds = ax.get_legend().get_bbox_to_anchor().transformed(ax.transAxes.inverted()).bounds
    np.testing.assert_allclose(bounds, anchor)


@pytest.mark.parametrize("extension", ["png", "pdf", "svg"])
def test_plot_columns_outside_legend_is_inside_export(
    presentation_source, presentation_figures, tmp_path, monkeypatch, extension,
):
    saved = {}
    original = figure_module.Figure.savefig

    def save(fig, path, *args, **kwargs):
        assert kwargs["bbox_inches"] == "tight"
        original(fig, path, *args, **kwargs)
        fig.canvas.draw()
        ax = fig.axes[0]
        renderer = fig.canvas.get_renderer()
        legend = ax.get_legend()
        bounds = legend.get_window_extent(renderer)
        assert legend.get_in_layout()
        assert legend._ncols == 2
        assert bounds.y0 > ax.get_window_extent(renderer).y1
        labels = [text.get_window_extent(renderer) for text in legend.get_texts()]
        assert abs(labels[0].y0 - labels[1].y0) < 1e-6  # Same row, not a vertical legend.
        tight = fig.get_tightbbox(renderer).padded(plt.rcParams["savefig.pad_inches"])
        legend_inches = bounds.transformed(fig.dpi_scale_trans.inverted())
        assert tight.contains(legend_inches.x0, legend_inches.y0)
        assert tight.contains(legend_inches.x1, legend_inches.y1)
        saved.update(tight=tight, legend=legend_inches, dpi=kwargs["dpi"])

    monkeypatch.setattr(figure_module.Figure, "savefig", save)
    output = tmp_path / f"legend.{extension}"
    metadata = tmp_path / f"legend-{extension}.json"
    rc = {"font.size": 16, "axes.labelsize": 20, "legend.fontsize": 18}
    plot_columns(str(presentation_source), xcolumn="x", ycolumn="y", show_plot=False,
                 width=8, height=5, rc_params=rc, legend_ncols=2,
                 legend_location="lower center", legend_bbox_to_anchor=[0.5, 1.04],
                 savefig=str(output), savefig_bbox_inches="tight", dpi=100,
                 figdata_fname=str(metadata), series=[{"label": "TI"}, {"label": "CE"}])
    assert output.stat().st_size > 0
    data = json.loads(metadata.read_text())
    assert data["rc_params"] == rc
    assert data["legend_ncols"] == 2
    assert data["legend_bbox_to_anchor"] == [0.5, 1.04]
    assert data["savefig_bbox_inches"] == "tight"
    if extension == "png":
        # Compare actual exported pixel dimensions and legend position in that canvas.
        pixels = plt.imread(output)
        tight, bounds, dpi = saved["tight"], saved["legend"], saved["dpi"]
        assert abs(pixels.shape[1] - tight.width*dpi) <= 1
        assert abs(pixels.shape[0] - tight.height*dpi) <= 1
        assert (bounds.x0 - tight.x0)*dpi >= 0
        assert (bounds.x1 - tight.x0)*dpi <= pixels.shape[1] + 1
        assert (bounds.y0 - tight.y0)*dpi >= 0
        assert (bounds.y1 - tight.y0)*dpi <= pixels.shape[0] + 1


def test_plot_columns_annotations_and_rc_scope(presentation_source, presentation_figures, tmp_path):
    keys = ["font.size", "axes.labelsize", "xtick.labelsize", "axes.linewidth", "axes.spines.top"]
    original = {key: plt.rcParams[key] for key in keys}
    rc = dict(zip(keys, [19, 23, 17, 2, False]))
    annotations = [{"text": "300 K", "x": 0.06, "y": 0.94, "ha": "left", "va": "top", "fontweight": "bold"},
                   {"text": "point", "x": 2, "y": 4, "coordinates": "data", "fontsize": 12}]
    path = tmp_path / "annotated.json"
    plot_columns(str(presentation_source), xcolumn="x", ycolumn="y", xfactor=1000,
                 xrange=[-10, 20], rc_params=rc, annotations=annotations, show_plot=False, figdata_fname=str(path))
    ax = presentation_figures[-1].axes[0]
    text, point = ax.texts
    assert text.get_transform() is ax.transAxes
    assert text.get_position() == (0.06, 0.94)
    assert text.get_fontweight() == "bold"
    assert text.get_fontsize() == 19
    assert point.get_transform() is ax.transData
    assert point.get_position() == (2, 4)
    assert ax.xaxis.label.get_fontsize() == 23
    assert ax.get_xticklabels()[0].get_fontsize() == 17
    assert ax.spines["left"].get_linewidth() == 2
    assert not ax.spines["top"].get_visible()
    ax.set_xlim(-1000, 1000)
    np.testing.assert_allclose(
        text.get_transform().transform(text.get_position()), ax.transAxes.transform((0.06, 0.94)),
    )
    assert {key: plt.rcParams[key] for key in keys} == original
    assert "coordinates" not in annotations[0]  # Defaults must not mutate the caller's dictionaries.
    assert json.loads(path.read_text())["annotations"][0]["coordinates"] == "axes fraction"

    plot_columns(str(presentation_source), xcolumn="x", ycolumn="y", show_plot=False)
    default_ax = presentation_figures[-1].axes[0]
    assert default_ax.xaxis.label.get_fontsize() == FontProperties(size=original["axes.labelsize"]).get_size_in_points()
    assert default_ax.spines["top"].get_visible() == original["axes.spines.top"]


@pytest.mark.parametrize("failure", ["render", "save", "rc"])
def test_plot_columns_rc_restored_on_error(presentation_source, tmp_path, monkeypatch, failure):
    original = plt.rcParams["font.size"]
    options = dict(xcolumn="x", ycolumn="y", rc_params={"font.size": 31}, show_plot=False)
    if failure == "render":
        options["series"] = [{"color": "not-a-matplotlib-color"}]
    elif failure == "save":
        def fail(*args, **kwargs):
            assert plt.rcParams["font.size"] == 31
            raise ValueError("save failed")
        monkeypatch.setattr(figure_module.Figure, "savefig", fail)
        options["savefig"] = str(tmp_path / "fail.png")
    else:
        options["rc_params"]["invalid.rc.key"] = 5
    with pytest.raises((ValueError, KeyError)):
        plot_columns(str(presentation_source), **options)
    assert plt.rcParams["font.size"] == original


@pytest.mark.parametrize("options,message", [
    ({"legend_ncols": 0}, "legend_ncols"), ({"legend_ncols": True}, "legend_ncols"),
    ({"legend_ncols": 1.5}, "legend_ncols"), ({"legend_bbox_to_anchor": [0.5]}, "2 or 4"),
    ({"legend_bbox_to_anchor": [np.inf, 1]}, "finite"),
    ({"legend_bbox_to_anchor": ["0.5", 1]}, "finite"), ({"rc_params": []}, "rc_params"),
    ({"rc_params": {"backend": "Agg"}}, "backend"), ({"annotations": {}}, "annotations"),
    ({"annotations": [{"text": "x", "x": 0}]}, "missing required keys"),
    ({"annotations": [{"text": "x", "x": 0, "y": 0, "color": "red"}]}, "unknown keys"),
    ({"annotations": [{"text": "x", "x": 0, "y": 0, "coordinates": "figure"}]}, "coordinates"),
    ({"annotations": [{"text": "x", "x": np.nan, "y": 0}]}, "finite number"),
    ({"savefig_bbox_inches": "invalid"}, "savefig_bbox_inches"),
])
def test_plot_columns_presentation_validation(presentation_source, options, message):
    with pytest.raises(ValueError, match=message):
        plot_columns(str(presentation_source), show_plot=False, **options)


def test_plot_columns_presentation_defaults(presentation_source, presentation_figures, tmp_path):
    path = tmp_path / "defaults.json"
    output = tmp_path / "defaults.png"
    plot_columns(str(presentation_source), xcolumn="x", ycolumn="y", show_plot=False,
                 savefig=str(output), figdata_fname=str(path), dpi=100)
    fig = presentation_figures[-1]
    np.testing.assert_allclose(fig.get_size_inches(), [6, 4])
    assert plt.imread(output).shape[:2] == (400, 600)
    assert not fig.axes[0].texts
    assert fig.axes[0].get_legend()._ncols == 1
    data = json.loads(path.read_text())
    assert data["xfactor"] == data["yfactor"] == 1
    assert data["rc_params"] is None
    assert data["legend_bbox_to_anchor"] is None
    assert data["savefig_bbox_inches"] is None
    assert data["annotations"] == []


def test_plot_columns_direct_and_named_dispatch(presentation_source, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    source = str(presentation_source)
    options = dict(source_filepath=source, xcolumn="x", ycolumn="y", xfactor=1000, yfactor=2000,
                   legend_ncols=2, legend_bbox_to_anchor=[0.5, 1.04], legend_location="lower center",
                   savefig_bbox_inches="tight", rc_params={"axes.labelsize": 20, "axes.spines.top": False},
                   annotations=[{"text": "300 K", "x": 0.06, "y": 0.94, "fontweight": "bold"}],
                   dpi=100)
    config = {"do": ["presentation"], "presentation": dict(command="plot_columns", show_plot=False,
               savefig=str(tmp_path / "named.png"), figdata_fname=str(tmp_path / "named.json"), **options)}
    path = tmp_path / "plots.toml"
    path.write_text(toml.dumps(config))
    command = [sys.executable, "-m", "clusterx.cli.main"]
    named = subprocess.run(command + ["--input_file", str(path)], capture_output=True, text=True)
    assert named.returncode == 0, named.stdout + named.stderr
    flags = []
    aliases = {"source_filepath": None, "xfactor": "xf", "yfactor": "yf"}
    for name, value in options.items():
        if name == "source_filepath":
            flags.append(source)
        else:
            argument = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            flags.extend(["-" + aliases.get(name, name).replace("_", "-"), argument])
    direct = subprocess.run(command + ["plot_columns"] + flags + [
        "-sf", str(tmp_path / "direct.png"), "-fdn", str(tmp_path / "direct.json"),
    ], capture_output=True, text=True)
    assert direct.returncode == 0, direct.stdout + direct.stderr
    assert json.loads((tmp_path / "direct.json").read_text()) == json.loads((tmp_path / "named.json").read_text())


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
