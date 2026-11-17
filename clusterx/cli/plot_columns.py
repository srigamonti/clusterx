# Copyright (c) CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import json
import operator
import os
import warnings
from numbers import Real
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import plac
from scipy.stats import gaussian_kde

from clusterx.cli.config_utils import cmd_message
from clusterx.derivative_structures import DerivativeStructures

commands = ["plot_columns"]


def _to_1d_float_array(value: Any) -> np.ndarray:
    """Convert scalar/array-like value to 1D float array."""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).ravel()
    else:
        arr = np.asarray([value], dtype=float)
    return arr


def _aggregate(values: np.ndarray, method: str, weights: Optional[np.ndarray] = None) -> float:
    if weights is None:
        weights = np.ones(len(values), dtype=float)
    if method == "median":
        order = np.argsort(values)
        sorted_values = values[order]
        sorted_weights = weights[order]
        return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), np.sum(weights) / 2.0)])
    return float(np.average(values, weights=weights))


def _weighted_samples(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Repeat samples according to integer multiplicity weights."""
    return np.repeat(values, weights.astype(int))


def _weighted_violin(ax, values, weights, position, width, color, scatter_kwargs):
    """Draw one violin from values and weights without expanding samples."""
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = np.asarray(values[finite], dtype=float)
    weights = np.asarray(weights[finite], dtype=float)
    if len(values) == 0:
        return

    lower = float(np.min(values))
    upper = float(np.max(values))
    if np.isclose(lower, upper):
        # A singleton or zero-spread bin has no violin width to display.
        return

    grid = np.linspace(lower, upper, 100)
    if len(np.unique(values)) > 1:
        density = gaussian_kde(values, weights=weights)(grid)
        density /= np.max(density)
    else:
        density = np.ones_like(grid)

    half_width = 0.5 * width * density
    facecolor = scatter_kwargs.get("facecolors", color)
    edgecolor = scatter_kwargs.get("edgecolors", color)
    body = ax.fill_betweenx(
        grid,
        position - half_width,
        position + half_width,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=float(scatter_kwargs.get("linewidths", 0.8)),
        alpha=float(scatter_kwargs.get("alpha", 0.3)),
    )
    return body


def _validate_multiplicity_weights(values: Any, series_index: int) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0) or np.any(weights != weights.astype(int)):
        raise ValueError(f"series[{series_index}] multiplicities must be positive integers.")
    return weights


def _infer_box_violin_width(x: np.ndarray) -> float:
    ux = np.unique(np.sort(np.asarray(x, dtype=float)))
    if len(ux) > 1:
        return 0.8 * float(np.min(np.diff(ux)))
    return 0.05


def _validate_axis_range(axis_range: Optional[list], name: str) -> Optional[List[float]]:
    if axis_range is None:
        return None
    if len(axis_range) != 2:
        axis_label = name[0]
        raise ValueError(f"{name} must have exactly two values: [{axis_label}min, {axis_label}max].")
    return [float(axis_range[0]), float(axis_range[1])]


def _finite_values(arrays: List[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.asarray([], dtype=float)
    values = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    return values[np.isfinite(values)]


def _require_positive(values: np.ndarray, context: str) -> None:
    if len(values) == 0:
        raise ValueError(f"Cannot use loglog=True without finite {context}.")
    if np.any(values <= 0):
        raise ValueError(
            f"Cannot use loglog=True because {context} contains non-positive values. Use symlog=True for signed data."
        )


def _infer_symlog_linthresh(values: np.ndarray) -> float:
    nonzero_abs = np.abs(values[np.abs(values) > 0])
    if len(nonzero_abs) == 0:
        return 1.0
    return 0.1 * float(np.min(nonzero_abs))


def _normalize_legend_location(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("legend_location must be a string.")

    key = value.strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())

    canonical = {
        "best",
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    }
    if key in canonical:
        return key

    valid = sorted(canonical)
    raise ValueError("legend_location must be one of: " + ", ".join(valid))


def _legend_anchor(value):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) not in (2, 4):
        raise ValueError("legend_bbox_to_anchor must be a list of 2 or 4 finite numbers in axes fractions.")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in value):
        raise ValueError("legend_bbox_to_anchor must contain only finite numbers.")
    try:
        anchor = [float(v) for v in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("legend_bbox_to_anchor must contain only finite numbers.") from exc
    if not np.all(np.isfinite(anchor)):
        raise ValueError("legend_bbox_to_anchor must contain only finite numbers.")
    return anchor


def _plot_annotations(value):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("annotations must be a list of dictionaries.")
    required = {"text", "x", "y"}
    allowed = required | {"coordinates", "ha", "va", "fontsize", "fontweight"}
    annotations = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"annotations[{i}] must be a dictionary.")
        if item.keys() - allowed:
            unknown = ", ".join(sorted(str(key) for key in item.keys() - allowed))
            raise ValueError(f"annotations[{i}] has unknown keys: {unknown}.")
        if required - item.keys():
            raise ValueError(f"annotations[{i}] is missing required keys: {', '.join(sorted(required - item.keys()))}.")
        item = item.copy()
        if not isinstance(item["text"], str):
            raise ValueError(f"annotations[{i}].text must be a string.")
        for name in ("x", "y"):
            coordinate = item[name]
            if (isinstance(coordinate, (bool, np.bool_)) or not isinstance(coordinate, Real)
                    or not np.isfinite(coordinate)):
                raise ValueError(f"annotations[{i}].{name} must be a finite number.")
            item[name] = float(coordinate)
        coordinates = item.setdefault("coordinates", "axes fraction")
        if coordinates not in ("axes fraction", "data"):
            raise ValueError(f"annotations[{i}].coordinates must be 'axes fraction' or 'data'.")
        annotations.append(item)
    return annotations


def _shared_target_prediction_limits(
    x_values: np.ndarray,
    y_values: np.ndarray,
    xrange_values: Optional[List[float]],
    yrange_values: Optional[List[float]],
    loglog: bool,
) -> List[float]:
    values_for_limits = [x_values, y_values]
    explicit_range = xrange_values is not None or yrange_values is not None
    if xrange_values is not None:
        values_for_limits.append(np.asarray(xrange_values, dtype=float))
    if yrange_values is not None:
        values_for_limits.append(np.asarray(yrange_values, dtype=float))

    values = _finite_values(values_for_limits)
    if len(values) == 0:
        raise ValueError("Cannot set target_vs_predictions limits without finite x/y data.")
    if loglog:
        _require_positive(values, "axis limits")

    lower = float(np.min(values))
    upper = float(np.max(values))

    if loglog:
        if np.isclose(lower, upper):
            return [lower / np.sqrt(10.0), upper * np.sqrt(10.0)]
        if explicit_range:
            return [lower, upper]
        log_lower = np.log10(lower)
        log_upper = np.log10(upper)
        pad = 0.05 * (log_upper - log_lower)
        return [float(10 ** (log_lower - pad)), float(10 ** (log_upper + pad))]

    if np.isclose(lower, upper):
        pad = 0.5 if np.isclose(lower, 0.0) else 0.05 * abs(lower)
        return [lower - pad, upper + pad]
    if explicit_range:
        return [lower, upper]
    pad = 0.05 * (upper - lower)
    return [lower - pad, upper + pad]


@plac.annotations(
    source_filepath=(
        "Path to serialized DerivativeStructures or StructuresSet object.",
        "positional",
        None,
        str,
    ),
    xcolumn=(
        "Default x-axis column used by series that do not define xcolumn.",
        "option",
        None,
        str,
    ),
    ycolumn=(
        "Default y-axis column used by series that do not define ycolumn.",
        "option",
        None,
        str,
    ),
    title=("Figure title.", "option", "t", str),
    xlabel=("Label for the x-axis of the plot.", "option", "xl", str),
    ylabel=("Label for the y-axis of the plot.", "option", "yl", str),
    xfactor=("Multiplicative factor applied to x data before plotting/export.", "option", "xf", float),
    yfactor=(
        "Multiplicative factor applied to y data before plotting/export.",
        "option",
        "yf",
        float,
    ),
    yrange=("Range for the y-axis of the plot (min,max).", "option", None, list),
    xrange=("Range for the x-axis of the plot (min,max).", "option", None, list),
    colors=(
        "Optional list of default colors for series without explicit color.",
        "option",
        None,
        list,
    ),
    width=("Figure width in inches.", "option", None, float),
    height=("Figure height in inches.", "option", None, float),
    target_vs_predictions=(
        "Enable target-vs-predictions parity mode: square axes, shared limits, and y=x line.",
        "option",
        None,
        bool,
    ),
    loglog=("Use logarithmic scale on both x and y axes.", "option", None, bool),
    symlog=(
        "Use symmetric logarithmic scale on both axes; supports negative and zero values.",
        "option",
        None,
        bool,
    ),
    symlog_linthresh=(
        "Linear threshold around zero for symlog axes. Inferred from data when omitted.",
        "option",
        None,
        float,
    ),
    grid=("Show plot grid lines.", "option", None, bool),
    grid_which=(
        "Grid lines to draw: major, minor, or both.",
        "option",
        None,
        str,
    ),
    grid_axis=(
        "Axis for grid lines: both, x, or y.",
        "option",
        None,
        str,
    ),
    grid_style=(
        "Optional matplotlib grid style dictionary, e.g. {linewidth=0.5, alpha=0.4}.",
        "option",
        None,
        dict,
    ),
    show_yzero_axis=("Show horizontal reference line at y=0.", "option", None, bool),
    show_xlabel=("Show x-axis name label.", "option", None, bool),
    show_ylabel=("Show y-axis name label.", "option", None, bool),
    show_xticklabels=("Show x-axis tick labels (numbers).", "option", None, bool),
    show_yticklabels=("Show y-axis tick labels (numbers).", "option", None, bool),
    xaxis_top=("Place x-axis label, tick marks, and tick labels on the top side.", "option", None, bool),
    yaxis_right=("Place y-axis label, tick marks, and tick labels on the right side.", "option", None, bool),
    xticks=("Custom x-axis tick locations.", "option", None, list),
    yticks=("Custom y-axis tick locations.", "option", None, list),
    legend=("Show legend if True.", "option", None, bool),
    show_legend_frame=("Show frame box around legend.", "option", None, bool),
    legend_location=("Legend position using Matplotlib names, e.g. upper left/best.", "option", None, str),
    legend_handlelength=("Legend handle length in font-size units.", "option", None, float),
    legend_handletextpad=("Legend padding between handle and text in font-size units.", "option", None, float),
    legend_ncols=("Positive number of legend columns (default 1).", "option", None, int),
    legend_bbox_to_anchor=("JSON list of 2 or 4 legend anchor coordinates in axes fractions.",
                           "option", None, json.loads),
    rc_params=("Matplotlib rcParams as a JSON dictionary, scoped to this plot.", "option", None, json.loads),
    annotations=("JSON list of text/x/y dictionaries; coordinates: 'axes fraction' or 'data'.",
                 "option", None, json.loads),
    show_plot=("Display the plot window if True.", "option", None, bool),
    savefig=(
        "Filepath to save generated figure (e.g. plot.png).",
        "option",
        "sf",
        str,
    ),
    dpi=("Resolution in dots per inch used when saving figures.", "option", None, float),
    savefig_bbox_inches=("Use 'tight' to include outside legends and labels when saving.", "option", None, str),
    figdata_fname=(
        "Filepath to save plotted data. Supported extensions: .npz, .json, .txt, .dat.",
        "option",
        "fdn",
        str,
    ),
    series=(
        "List of series dictionaries from TOML. Per-series keys: xcolumn, ycolumn, "
        "source_filepath (optional), mask_name (optional), label (optional), "
        "filter_ycolumn/filter_value (optional), "
        "xfactor/yfactor (optional), color/marker/size/edgecolor/facecolor/linewidth/alpha (optional), "
        "or style={...} for grouped styling. Also supports ymode in "
        "{marker,cloud,violin,boxplot,auto} for array-valued y columns. "
        "Set binned=true to group scalar y values by unique x bin (with optional xtol snap). "
        "Set weight_by_multiplicity=true to weight y values by DerivativeStructures multiplicity.",
        "option",
        None,
        list,
    ),
)
def plot_columns(
    source_filepath: Optional[str] = None,
    xcolumn: Optional[str] = None,
    ycolumn: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    yfactor: float = 1.0,
    yrange: Optional[list] = None,
    xrange: Optional[list] = None,
    colors: Optional[list] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    target_vs_predictions: bool = False,
    loglog: bool = False,
    symlog: bool = False,
    symlog_linthresh: Optional[float] = None,
    grid: bool = False,
    grid_which: str = "major",
    grid_axis: str = "both",
    grid_style: Optional[Dict[str, Any]] = None,
    show_yzero_axis: bool = False,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_xticklabels: bool = True,
    show_yticklabels: bool = True,
    xaxis_top: bool = False,
    yaxis_right: bool = False,
    xticks: Optional[list] = None,
    yticks: Optional[list] = None,
    legend: bool = True,
    show_legend_frame: bool = True,
    legend_location: str = "best",
    legend_handlelength: Optional[float] = None,
    legend_handletextpad: Optional[float] = None,
    show_plot: bool = True,
    savefig: Optional[str] = None,
    dpi: float = 300.0,
    figdata_fname: Optional[str] = None,
    series: Optional[List[Dict[str, Any]]] = None,
    xfactor: float = 1.0,
    rc_params: Optional[Dict[str, Any]] = None,
    legend_ncols: int = 1,
    legend_bbox_to_anchor: Optional[list] = None,
    annotations: Optional[List[Dict[str, Any]]] = None,
    savefig_bbox_inches: Optional[str] = None,
):
    """Plot one or more columns from serialized DerivativeStructures or StructuresSet objects.

    This command can be driven from TOML using ``[plot_columns]`` and optional
    ``[[plot_columns.series]]`` blocks. If ``series`` is omitted or empty,
    one series is plotted using the top-level ``source_filepath``, ``xcolumn``,
    and ``ycolumn``. Options available only per series use their built-in
    defaults.

    Top-level options
    -----------------
    - ``source_filepath`` (str, optional): default serialized DerivativeStructures or
        StructuresSet file used by all series that do not explicitly set
        ``source_filepath``.
    - ``xcolumn`` / ``ycolumn`` (str, optional): default x/y columns used by
        series that do not explicitly set them.
    - ``title`` (str, optional): figure title.
    - ``xlabel`` / ``ylabel`` (str, optional): axis labels.
    - ``xfactor`` / ``yfactor`` (float): multiply each coordinate once before
        grouping, plotting, parity limits and export (both default to ``1.0``).
        Source properties are unchanged. Each series can override either factor.
    - ``xrange`` / ``yrange`` (list[float, float], optional): axis limits in
        displayed units, after coordinate scaling. Ticks, ``xtol``, distribution
        widths and x jitter also use displayed units and are not scaled again.
    - ``rc_params`` (dict, optional): Matplotlib rcParams applied inside an
        ``rc_context`` spanning figure creation, formatting and saving. Settings
        are restored even on errors. Set the backend before calling this command;
        ``backend`` is not a supported presentation setting here.
    - ``colors`` (list[str], optional): default color palette for series without
        explicit ``color``.
    - ``width`` / ``height`` (float, optional): figure size in inches. Defaults
        preserve existing behavior: ``6x4`` for regular plots and ``6x6`` for
        ``target_vs_predictions``.
    - For ``target_vs_predictions``, the figure is kept square. If ``width`` is
        provided, the size becomes ``(width, width)`` (``height`` is ignored).
    - ``target_vs_predictions`` (bool): enable parity-plot mode for target values
        on x versus predictions on y. This uses a square plot frame, shared x/y
        limits, and a ``y=x`` reference line.
    - ``loglog`` (bool): use logarithmic scale on both axes. All finite plotted
        x/y values must be positive.
    - ``symlog`` (bool): use symmetric logarithmic scale on both axes. This is
        the signed-log option for data containing negative or zero values.
    - ``symlog_linthresh`` (float, optional): half-width of the linear region
        around zero for ``symlog``. If omitted, it is inferred from the smallest
        nonzero plotted absolute value.
    - ``grid`` (bool): draw plot grid lines.
    - ``grid_which`` (str): ``"major"`` (default), ``"minor"``, or ``"both"``.
    - ``grid_axis`` (str): ``"both"`` (default), ``"x"``, or ``"y"``.
    - ``grid_style`` (dict, optional): matplotlib grid style passed to
        ``Axes.grid`` (e.g. ``linewidth``, ``linestyle``, ``alpha``, ``color``).
    - ``show_yzero_axis`` (bool): draw a horizontal line at ``y=0``.
    - ``show_xlabel`` / ``show_ylabel`` (bool): show/hide axis name labels.
    - ``show_xticklabels`` / ``show_yticklabels`` (bool): show/hide numeric
        tick labels on each axis.
    - ``xaxis_top`` (bool): if ``True``, place x-axis label, tick marks, and
        tick labels on the top side.
    - ``yaxis_right`` (bool): if ``True``, place y-axis label, tick marks, and
        tick labels on the right side.
    - ``xticks`` / ``yticks`` (list[float], optional): explicit tick locations
        to apply to the x/y axes via ``Axes.set_xticks`` and ``Axes.set_yticks``.
    - ``legend`` (bool): show legend.
    - ``show_legend_frame`` (bool): show/hide the legend frame box.
    - ``legend_location`` (str): legend position using Matplotlib ``loc``
        names (e.g. ``"best"``, ``"upper left"``, ``"lower center"``).
    - ``legend_ncols`` (int): positive number of legend columns (default ``1``).
    - ``legend_bbox_to_anchor`` (list, optional): 2 or 4 finite numbers in axes
        fractions, passed as Matplotlib's ``bbox_to_anchor``. For two entries,
        anchor at (x, y); four entries specify (x, y, width, height). For a
        horizontal legend above the plot, use ``legend_ncols=2``,
        ``legend_location="lower center"`` and anchor ``[0.5, 1.04]``.
    - ``legend_handlelength`` (float, optional): handle length passed to
        Matplotlib ``legend(handlelength=...)``.
    - ``legend_handletextpad`` (float, optional): spacing between legend
        handles and labels passed to ``legend(handletextpad=...)``.
    - ``show_plot`` (bool): show interactive window.
    - ``savefig`` (str, optional): output figure path (e.g. ``plot.png``).
    - ``savefig_bbox_inches`` (str, optional): ``"tight"`` includes outside
        legends and labels when saving PNG, PDF or SVG. Omitted by default to
        preserve existing figure dimensions. Legends participate in layout.
    - ``annotations`` (list[dict], optional): text labels with required keys
        ``text`` (str), ``x`` and ``y`` (finite numbers). ``coordinates`` defaults
        to ``"axes fraction"``: (0, 0) is the lower-left axes corner, (1, 1) the
        upper-right, independent of data limits. ``"data"`` uses displayed data
        coordinates. Optional keys are ``ha``, ``va``, ``fontsize``, ``fontweight``.
        Other keys are rejected. Annotation coordinates are not scaled again.
    - ``dpi`` (float): resolution used for ``savefig`` (default: ``300``).
    - ``figdata_fname`` (str, optional): data export path
        (``.json/.npz/.txt/.dat``).

    Per-series options (``[[plot_columns.series]]``)
    -------------------------------------------------
    Required (directly or via top-level defaults):

    - ``xcolumn`` (str): x-axis column from the loaded configurations table.
    - ``ycolumn`` (str): y-axis column from the loaded configurations table.

    Optional:

    - ``source_filepath`` (str): override top-level source for this series.
    - ``mask_name`` (str): mask name in the DerivativeStructures.
    - ``filter_ycolumn`` (str): optional column name used to filter rows before
        plotting.
    - ``filter_value``: only rows where ``filter_ycolumn == filter_value`` are
        kept.
    - ``label`` (str): legend label.
    - ``xfactor`` / ``yfactor`` (float): override the corresponding top-level
        scaling factor for this series.
    - ``color`` (str): matplotlib color spec (e.g. ``"k"``, ``"#1f77b4"``,
        ``"red"``).
    - ``marker`` (str): matplotlib marker (e.g. ``"o"``, ``"s"``, ``"^"``, ``"."``).
    - ``size`` (float): marker area passed as ``s`` to
        ``matplotlib.pyplot.scatter``.
    - ``edgecolor`` (str): marker edge color.
    - ``facecolor`` (str): marker face color (e.g. ``"none"`` for hollow markers).
    - ``linewidth`` (float): marker edge line width.
    - ``alpha`` (float): marker transparency.
    - ``style`` (dict): compact style dict with keys ``color``, ``marker``,
        ``size``/``s``, ``edgecolor``/``edgecolors``,
        ``facecolor``/``facecolors``, ``linewidth``/``linewidths``, ``alpha``.
    - ``ymode`` (str): ``"marker"``, ``"cloud"``, ``"violin"``,
        ``"boxplot"``, or ``"auto"``.
    - ``aggregation`` (str): for ``ymode="marker"`` with array-valued y,
        use ``"mean"`` (default) or ``"median"``.
    - ``xjitter`` (float): optional x-jitter amplitude for ``ymode="cloud"``.
    - ``width`` (float): width for ``violin``/``boxplot``.
    - ``binned`` (bool): if ``True``, group the scalar y values by unique x
        before plotting; ``ymode`` and ``aggregation`` then act on the
        per-bin y distribution (e.g. ``ymode="marker", aggregation="mean"``
        plots one mean per bin; ``ymode="violin"`` plots the full
        per-bin distribution). Ignored when y is already array-valued.
    - ``weight_by_multiplicity`` (bool): if ``True``, weight each structure by
        its derivative-structure multiplicity. Weighted means and medians are
        used for aggregation, and integer-weighted samples are used for
        cloud/violin/box distributions.
    - ``xtol`` (float): snap x values to a grid of this spacing before
        grouping (useful when concentrations differ only by floating-point
        noise, e.g. ``xtol=1e-6``).
    - ``mean_line`` (bool): if ``True``, overlay a line joining per-x means
        (x sorted in increasing order). Useful with ``binned=true`` or
        array-valued y columns.
    - ``mean_line_style`` (dict): optional matplotlib ``plot`` style for the
        mean line (e.g. ``linewidth``, ``linestyle``, ``alpha``, ``color``).
    - ``mean_line_label`` (str): optional legend label for the mean line.

    If both ``style`` and direct keys are provided, direct keys win.

    JSON exports record presentation settings and the transformed series
    ``x``, ``y`` and ``y_samples`` values, with their effective x/y factors.
    NPZ exports include ``xfactor_i``, ``yfactor_i`` and a JSON ``plot_metadata``
    string. Text exports retain their three numeric columns and add factors
    in the comment header.

    Presentation example with named TOML steps
    ------------------------------------------
    Run with ``cell --input_file plots.toml``. Both energy columns in this
    example are stored in eV/atom and displayed in meV/atom::

        do = ["parity"]

        [parity]
        command = "plot_columns"
        source_filepath = "training_predictions.xyz"
        xcolumn = "f_mix"
        ycolumn = "f_mix_pred"
        xfactor = 1000.0
        yfactor = 1000.0
        target_vs_predictions = true
        xlabel = '$F_v$, TI [meV / atom]'
        ylabel = '$F_v$, CE [meV / atom]'
        show_plot = false
        savefig = "parity.png"
        savefig_bbox_inches = "tight"
        figdata_fname = "parity.json"

        [parity.rc_params]
        "font.size" = 20.0
        "axes.labelsize" = 24.0
        "axes.spines.top" = false

        [[parity.annotations]]
        text = "300 K"
        x = 0.06
        y = 0.94
        ha = "left"
        va = "top"
        fontweight = "bold"

    Direct CLI structured options use JSON, for example
    ``-rc-params '{"axes.labelsize":24}'``,
    ``-legend-bbox-to-anchor '[0.5,1.04]'``, and
    ``-annotations '[{"text":"300 K","x":0.06,"y":0.94}]'``.

    Color precedence
    ----------------
    1) Per-series ``color``
    2) Top-level ``colors[i]``
    3) Internal default palette (tab10-like)

    Example
    -------
    .. code-block:: toml

            [plot_columns]
            source_filepath = "file1.pickle"
            title = "Energy of mixing"
            xlabel = "Concentration"
            ylabel = "Energy"
            yfactor = 1000.0
            colors = ["k", "r", "b"]
            width = 7.0
            height = 4.5
            legend = true
            show_plot = true
            savefig = "plot.png"
            dpi = 300
            # target_vs_predictions = true  # Square parity plot with y=x
            # width = 7.5                   # In target_vs_predictions: uses (width, width)
            # loglog = true                 # Logarithmic x/y axes
            # symlog = true                 # Signed-log x/y axes
            # grid = true
            # grid_which = "both"
            # grid_axis = "both"
            # show_xlabel = false
            # show_xticklabels = false
            # show_legend_frame = false
            # legend_location = "upper right"

            [[plot_columns.series]]
            xcolumn = "frconc_binary"
            ycolumn = "energy_of_mixing"
            marker = "o"
            label = "file1"

            [[plot_columns.series]]
            xcolumn = "frconc_binary"
            ycolumn = "energy_of_mixing"
            color = "#2ca02c"
            marker = "^"
            mask_name = "lowest"
            label = "file1 lowest"

            [[plot_columns.series]]
            xcolumn = "frconc_binary"
            ycolumn = "energy_of_mixing"
            label = "styled"
            [plot_columns.series.style]
            marker = "o"
            edgecolor = "red"
            facecolor = "none"
            size = 20
            linewidth = 0.85

            # Bagging/ensemble predictions stored as arrays per row
            [[plot_columns.series]]
            xcolumn = "frconc_binary"
            ycolumn = "energy_of_mixing_ensemble_predictions"
            ymode = "cloud"
            xjitter = 0.003
            color = "tab:blue"
            size = 8
            alpha = 0.25
            label = "bagging cloud"

            [[plot_columns.series]]
            xcolumn = "frconc_binary"
            ycolumn = "energy_of_mixing_ensemble_predictions"
            ymode = "violin"
            width = 0.02
            color = "tab:red"
            facecolor = "none"
            linewidth = 0.8
            label = "bagging violin"

            [[plot_columns.series]]
            xcolumn = "frconc_binary"
            ycolumn = "energy_of_mixing_ensemble_predictions"
            ymode = "boxplot"
            width = 0.02
            showfliers = false
            color = "black"
            label = "bagging box"

            # Group scalar y-values by concentration bins (after optional snapping)
            [[plot_columns.series]]
            xcolumn = "frconc_binary"
            ycolumn = "energy_of_mixing"
            binned = true
            xtol = 1e-6
            ymode = "marker"
            aggregation = "mean"
            marker = "o"
            color = "tab:purple"
            label = "binned mean"

            # Overlay a sorted mean line on top of per-bin distributions
            [[plot_columns.series]]
            xcolumn = "frconc_binary"
            ycolumn = "energy_of_mixing"
            binned = true
            xtol = 1e-6
            ymode = "violin"
            mean_line = true
            mean_line_label = "binned mean trend"
            [plot_columns.series.mean_line_style]
            linewidth = 1.4
            linestyle = "-"
            alpha = 0.95

            # Target-vs-predictions parity plot
            [[plot_columns.series]]
            xcolumn = "target_energy"
            ycolumn = "predicted_energy"
            marker = "o"
            label = "fit"
    """
    cmd_message("head")

    if series is None:
        series = []
    elif not isinstance(series, list):
        raise ValueError("series must be a list of dictionaries.")
    if not series:
        series = [{}]

    if source_filepath is None:
        missing_src = [
            i for i, item in enumerate(series) if not isinstance(item, dict) or item.get("source_filepath") is None
        ]
        if missing_src:
            raise ValueError(
                "source_filepath must be provided at top level or inside every "
                "series entry. Missing in series indices: " + ", ".join(str(i) for i in missing_src)
            )

    if colors is not None and not isinstance(colors, list):
        raise ValueError("colors must be a list when provided.")

    if width is not None and float(width) <= 0:
        raise ValueError("width must be positive when provided.")

    if height is not None and float(height) <= 0:
        raise ValueError("height must be positive when provided.")

    if float(dpi) <= 0:
        raise ValueError("dpi must be positive.")

    xfactor = float(xfactor)
    yfactor = float(yfactor)

    if rc_params is not None and not isinstance(rc_params, dict):
        raise ValueError("rc_params must be a dictionary when provided.")
    if rc_params is not None and "backend" in rc_params:
        raise ValueError("rc_params cannot set backend; select the Matplotlib backend before calling plot_columns.")
    try:
        if isinstance(legend_ncols, (bool, np.bool_)):
            raise TypeError
        legend_ncols = operator.index(legend_ncols)
    except TypeError as exc:
        raise ValueError("legend_ncols must be a positive integer.") from exc
    if legend_ncols < 1:
        raise ValueError("legend_ncols must be a positive integer.")
    legend_bbox_to_anchor = _legend_anchor(legend_bbox_to_anchor)
    annotations = _plot_annotations(annotations)
    if savefig_bbox_inches not in (None, "tight"):
        raise ValueError("savefig_bbox_inches must be None or 'tight'.")

    if loglog and symlog:
        raise ValueError("Use only one of loglog=True or symlog=True.")

    if symlog_linthresh is not None and symlog_linthresh <= 0:
        raise ValueError("symlog_linthresh must be positive when provided.")

    if grid_which not in ("major", "minor", "both"):
        raise ValueError("grid_which must be one of 'major', 'minor', or 'both'.")

    if grid_axis not in ("both", "x", "y"):
        raise ValueError("grid_axis must be one of 'both', 'x', or 'y'.")

    if grid_style is None:
        grid_style = {}
    elif not isinstance(grid_style, dict):
        raise ValueError("grid_style must be a dictionary when provided.")

    if xticks is not None and not isinstance(xticks, list):
        raise ValueError("xticks must be a list when provided.")
    if yticks is not None and not isinstance(yticks, list):
        raise ValueError("yticks must be a list when provided.")

    if legend_handlelength is not None:
        legend_handlelength = float(legend_handlelength)
        if legend_handlelength < 0:
            raise ValueError("legend_handlelength must be non-negative when provided.")

    if legend_handletextpad is not None:
        legend_handletextpad = float(legend_handletextpad)
        if legend_handletextpad < 0:
            raise ValueError("legend_handletextpad must be non-negative when provided.")

    legend_location_normalized = _normalize_legend_location(legend_location)

    # Cache DerivativeStructures objects by filepath to avoid repeated I/O
    dsg_cache: Dict[str, DerivativeStructures] = {}
    plot_payload: Dict[str, Any] = {
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "width": width,
        "height": height,
        "xfactor": xfactor,
        "yfactor": yfactor,
        "rc_params": rc_params,
        "legend_ncols": legend_ncols,
        "legend_bbox_to_anchor": legend_bbox_to_anchor,
        "annotations": annotations,
        "savefig_bbox_inches": savefig_bbox_inches,
        "target_vs_predictions": target_vs_predictions,
        "loglog": loglog,
        "symlog": symlog,
        "symlog_linthresh": symlog_linthresh,
        "grid": grid,
        "grid_which": grid_which,
        "grid_axis": grid_axis,
        "grid_style": grid_style,
        "show_xlabel": show_xlabel,
        "show_ylabel": show_ylabel,
        "show_xticklabels": show_xticklabels,
        "show_yticklabels": show_yticklabels,
        "xaxis_top": xaxis_top,
        "yaxis_right": yaxis_right,
        "xticks": xticks,
        "yticks": yticks,
        "show_legend_frame": show_legend_frame,
        "legend_location": legend_location_normalized,
        "legend_handlelength": legend_handlelength,
        "legend_handletextpad": legend_handletextpad,
        "dpi": float(dpi),
        "series": [],
    }

    if target_vs_predictions:
        fig_width = 6.0 if width is None else float(width)
        fig_height = fig_width
    else:
        fig_width = 6.0 if width is None else float(width)
        fig_height = 4.0 if height is None else float(height)

    with plt.rc_context(rc=rc_params):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        try:
            plotted_x_values: List[np.ndarray] = []
            plotted_y_values: List[np.ndarray] = []

            # Color priority: series[i].color > colors[i] > matplotlib tab10 cycle
            default_colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]
            palette = colors if (colors is not None and len(colors) > 0) else default_colors

            for i, item in enumerate(series):
                if not isinstance(item, dict):
                    raise ValueError(f"series[{i}] must be a dictionary.")

                src = item.get("source_filepath", source_filepath)
                xcolumn_s = item.get("xcolumn", xcolumn)
                ycolumn_s = item.get("ycolumn", ycolumn)
                xfactor_s = float(item.get("xfactor", xfactor))
                yfactor_s = float(item.get("yfactor", yfactor))
                mask_name = item.get("mask_name")
                filter_ycolumn = item.get("filter_ycolumn")
                has_filter_value = "filter_value" in item
                filter_value = item.get("filter_value")

                if xcolumn_s is None or ycolumn_s is None:
                    raise ValueError(
                        f"series[{i}] must define both 'xcolumn' and 'ycolumn', "
                        "either directly or via top-level defaults."
                    )

                if filter_ycolumn is not None and not has_filter_value:
                    raise ValueError(f"series[{i}] defines 'filter_ycolumn' but is missing 'filter_value'.")
                if has_filter_value and filter_ycolumn is None:
                    raise ValueError(f"series[{i}] defines 'filter_value' but is missing 'filter_ycolumn'.")

                if src not in dsg_cache:
                    dsg_cache[src] = _load_plot_source(src)

                dsg = dsg_cache[src]

                if xcolumn_s == "frconc_binary" and xcolumn_s not in dsg.configurations.columns:
                    dsg.add_fractional_concentration_binary()

                if filter_ycolumn == "frconc_binary" and filter_ycolumn not in dsg.configurations.columns:
                    dsg.add_fractional_concentration_binary()

                if xcolumn_s not in dsg.configurations.columns:
                    raise ValueError(f"Column '{xcolumn_s}' not found in configurations from '{src}'.")

                if ycolumn_s not in dsg.configurations.columns:
                    raise ValueError(f"Column '{ycolumn_s}' not found in configurations from '{src}'.")

                weight_by_multiplicity = bool(item.get("weight_by_multiplicity", False))
                if weight_by_multiplicity and "multiplicity" not in dsg.configurations.columns:
                    dsg.add_multiplicity()

                df = dsg.get_configurations(mask_name=mask_name)

                if filter_ycolumn is not None:
                    if filter_ycolumn not in df.columns:
                        raise ValueError(f"Column '{filter_ycolumn}' not found in configurations from '{src}'.")
                    df = df[df[filter_ycolumn] == filter_value]

                if len(df) == 0:
                    warnings.warn(
                        f"series[{i}] produced no rows after applying mask/filter; skipping series.",
                        RuntimeWarning,
                    )
                    continue

                x = np.asarray(df[xcolumn_s].to_numpy(), dtype=float) * xfactor_s

                # y can be scalar-valued or array-valued (e.g., bagging predictions)
                y_cells = df[ycolumn_s].tolist()
                y_arrays = [_to_1d_float_array(v) * yfactor_s for v in y_cells]
                is_array_valued = any(len(v) > 1 for v in y_arrays)
                if weight_by_multiplicity:
                    row_weights = _validate_multiplicity_weights(df["multiplicity"].to_numpy(), i)
                else:
                    row_weights = np.ones(len(y_arrays), dtype=float)
                y_weights = [np.full(len(values), weight) for values, weight in zip(y_arrays, row_weights)]

                binned = bool(item.get("binned", False))
                xtol = item.get("xtol", None)

                if binned:
                    # Snap x to a regular grid when a tolerance is given, then group
                    # all y samples that fall in the same x bin into one array.
                    if xtol is not None:
                        x_key = np.round(x / float(xtol)) * float(xtol)
                    else:
                        x_key = x.copy()
                    unique_x, inverse = np.unique(x_key, return_inverse=True)
                    y_arrays = [
                        np.concatenate([y_arrays[j] for j in np.where(inverse == k)[0]]) for k in range(len(unique_x))
                    ]
                    y_weights = [
                        np.concatenate([y_weights[j] for j in np.where(inverse == k)[0]]) for k in range(len(unique_x))
                    ]
                    x = unique_x
                    is_array_valued = any(len(v) > 1 for v in y_arrays)

                scatter_kwargs = {}

                style = item.get("style")
                if style is not None:
                    if not isinstance(style, dict):
                        raise ValueError(f"series[{i}].style must be a dictionary.")

                    # accept user-friendly keys in style dict
                    if "size" in style and "s" not in style:
                        scatter_kwargs["s"] = style["size"]
                    if "linewidth" in style and "linewidths" not in style:
                        scatter_kwargs["linewidths"] = style["linewidth"]
                    if "edgecolor" in style and "edgecolors" not in style:
                        scatter_kwargs["edgecolors"] = style["edgecolor"]
                    if "facecolor" in style and "facecolors" not in style:
                        scatter_kwargs["facecolors"] = style["facecolor"]

                    for k in (
                        "color",
                        "marker",
                        "s",
                        "edgecolors",
                        "facecolors",
                        "linewidths",
                        "alpha",
                    ):
                        if k in style:
                            scatter_kwargs[k] = style[k]

                # direct keys override style keys
                if "color" in item:
                    color = item["color"]
                elif "color" in scatter_kwargs:
                    color = scatter_kwargs["color"]
                else:
                    color = palette[i % len(palette)]
                scatter_kwargs["color"] = color

                if "marker" in item:
                    scatter_kwargs["marker"] = item["marker"]
                if "size" in item:
                    scatter_kwargs["s"] = item["size"]
                if "edgecolor" in item:
                    scatter_kwargs["edgecolors"] = item["edgecolor"]
                if "facecolor" in item:
                    scatter_kwargs["facecolors"] = item["facecolor"]
                if "linewidth" in item:
                    scatter_kwargs["linewidths"] = item["linewidth"]
                if "alpha" in item:
                    scatter_kwargs["alpha"] = item["alpha"]

                ymode = str(item.get("ymode", "auto")).lower()
                if ymode == "auto":
                    ymode = "cloud" if is_array_valued else "marker"
                if ymode not in ("marker", "cloud", "violin", "boxplot", "box"):
                    raise ValueError(f"series[{i}].ymode must be one of marker, cloud, violin, boxplot, auto.")

                aggregation = str(item.get("aggregation", "mean")).lower()
                if aggregation not in ("mean", "median"):
                    raise ValueError("aggregation must be 'mean' or 'median'.")

                mean_line = bool(item.get("mean_line", False))
                mean_line_label = item.get("mean_line_label")
                mean_line_style = item.get("mean_line_style", {})
                if not isinstance(mean_line_style, dict):
                    raise ValueError(f"series[{i}].mean_line_style must be a dictionary.")

                label = item.get("label", f"series_{i}")

                # Representative scalar y per x (used by marker mode and for exports)
                y_repr = np.asarray(
                    [_aggregate(v, aggregation, weights) for v, weights in zip(y_arrays, y_weights)],
                    dtype=float,
                )
                has_distribution_per_x = any(len(v) > 1 for v in y_arrays)
                plotted_x_values.append(x)
                if ymode in ("cloud", "violin", "boxplot", "box"):
                    plotted_y_values.append(np.concatenate([values for values, weights in zip(y_arrays, y_weights)]))
                else:
                    plotted_y_values.append(y_repr)

                if ymode == "marker":
                    ax.scatter(x, y_repr, label=label, **scatter_kwargs)
                elif ymode == "cloud":
                    cloud_kwargs = scatter_kwargs.copy()
                    xjitter = float(item.get("xjitter", 0.0))
                    cloud_kwargs.setdefault("s", 8)
                    cloud_kwargs.setdefault("alpha", 0.35)

                    all_x = []
                    all_y = []
                    for xi, yi, weights in zip(x, y_arrays, y_weights):
                        if weight_by_multiplicity:
                            yi = _weighted_samples(yi, weights)
                        if xjitter > 0:
                            dx = np.random.uniform(-xjitter, xjitter, size=len(yi))
                            xvals = xi + dx
                        else:
                            xvals = np.full(len(yi), xi)
                        all_x.append(xvals)
                        all_y.append(yi)

                    cloud_x = np.concatenate(all_x)
                    cloud_y = np.concatenate(all_y)
                    plotted_x_values[-1] = cloud_x
                    plotted_y_values[-1] = cloud_y
                    ax.scatter(cloud_x, cloud_y, label=label, **cloud_kwargs)
                elif ymode == "violin":
                    width = float(item.get("width", _infer_box_violin_width(x)))
                    if weight_by_multiplicity:
                        for position, values, weights in zip(x, y_arrays, y_weights):
                            _weighted_violin(ax, values, weights, position, width, color, scatter_kwargs)
                    else:
                        v = ax.violinplot(y_arrays, positions=x, widths=width, showextrema=False)
                        vbodies = v.get("bodies", [])
                        if not isinstance(vbodies, list):
                            vbodies = [vbodies]
                        for body in vbodies:
                            body.set_facecolor(scatter_kwargs.get("facecolors", color))
                            body.set_edgecolor(scatter_kwargs.get("edgecolors", color))
                            body.set_linewidth(float(scatter_kwargs.get("linewidths", 0.8)))
                            body.set_alpha(float(scatter_kwargs.get("alpha", 0.3)))
                    ax.plot([], [], color=color, label=label)
                else:  # boxplot / box
                    width = float(item.get("width", _infer_box_violin_width(x)))
                    box_arrays = [
                        _weighted_samples(values, weights) if weight_by_multiplicity else values
                        for values, weights in zip(y_arrays, y_weights)
                    ]
                    bp = ax.boxplot(
                        box_arrays,
                        positions=x,
                        widths=width,
                        patch_artist=True,
                        showfliers=bool(item.get("showfliers", False)),
                        manage_ticks=False,
                    )
                    alpha = float(scatter_kwargs.get("alpha", 0.8))
                    for patch in bp["boxes"]:
                        patch.set_facecolor(scatter_kwargs.get("facecolors", "none"))
                        patch.set_edgecolor(scatter_kwargs.get("edgecolors", color))
                        patch.set_linewidth(float(scatter_kwargs.get("linewidths", 0.8)))
                        patch.set_alpha(alpha)
                    for med in bp["medians"]:
                        med.set_color(color)
                        med.set_alpha(alpha)
                    for whisk in bp["whiskers"]:
                        whisk.set_color(color)
                        whisk.set_alpha(alpha)
                    for cap in bp["caps"]:
                        cap.set_color(color)
                        cap.set_alpha(alpha)
                    ax.plot([], [], color=color, label=label)

                if mean_line:
                    if not has_distribution_per_x:
                        warnings.warn(
                            f"series[{i}] mean_line requested but no per-x distribution was found "
                            "(each x has a single y value). Drawing the line anyway.",
                            RuntimeWarning,
                        )

                    order = np.argsort(x)
                    line_kwargs = {
                        "color": color,
                        "linewidth": 1.2,
                        "linestyle": "-",
                    }
                    line_kwargs.update(mean_line_style)
                    line_kwargs["label"] = mean_line_label if mean_line_label is not None else "_nolegend_"
                    ax.plot(x[order], y_repr[order], **line_kwargs)

                plot_payload["series"].append(
                    {
                        "source_filepath": src,
                        "xcolumn": xcolumn_s,
                        "ycolumn": ycolumn_s,
                        "xfactor": xfactor_s,
                        "yfactor": yfactor_s,
                        "mask_name": mask_name,
                        "filter_ycolumn": filter_ycolumn,
                        "filter_value": filter_value if has_filter_value else None,
                        "label": label,
                        "ymode": ymode,
                        "aggregation": aggregation,
                        "weight_by_multiplicity": weight_by_multiplicity,
                        "binned": binned,
                        "xtol": xtol,
                        "mean_line": mean_line,
                        "mean_line_label": mean_line_label,
                        "mean_line_style": mean_line_style,
                        "color": color,
                        "marker": scatter_kwargs.get("marker"),
                        "size": scatter_kwargs.get("s"),
                        "edgecolor": scatter_kwargs.get("edgecolors"),
                        "facecolor": scatter_kwargs.get("facecolors"),
                        "linewidth": scatter_kwargs.get("linewidths"),
                        "alpha": scatter_kwargs.get("alpha"),
                        "x": x.tolist(),
                        "y": y_repr.tolist(),
                        "y_samples": [v.tolist() for v in y_arrays],
                    }
                )

            ax.set_xlabel(xlabel if xlabel is not None else ("Target" if target_vs_predictions else "x"))
            ax.set_ylabel(ylabel if ylabel is not None else ("Predictions" if target_vs_predictions else "y"))

            if xaxis_top:
                ax.xaxis.set_label_position("top")
                ax.tick_params(axis="x", which="both", bottom=False, top=True, labelbottom=False, labeltop=True)

            if yaxis_right:
                ax.yaxis.set_label_position("right")
                ax.tick_params(axis="y", which="both", left=False, right=True, labelleft=False, labelright=True)

            if not show_xlabel:
                ax.set_xlabel("")
            if not show_ylabel:
                ax.set_ylabel("")
            if not show_xticklabels:
                ax.tick_params(axis="x", which="both", labelbottom=False, labeltop=False)
            if not show_yticklabels:
                ax.tick_params(axis="y", which="both", labelleft=False, labelright=False)
            if title is not None:
                ax.set_title(title)

            if xticks is not None:
                ax.set_xticks([float(value) for value in xticks])
            if yticks is not None:
                ax.set_yticks([float(value) for value in yticks])

            xrange_values = _validate_axis_range(xrange, "xrange")
            yrange_values = _validate_axis_range(yrange, "yrange")
            finite_x_values = _finite_values(plotted_x_values)
            finite_y_values = _finite_values(plotted_y_values)

            if loglog:
                _require_positive(finite_x_values, "x data")
                _require_positive(finite_y_values, "y data")
                if xrange_values is not None:
                    _require_positive(np.asarray(xrange_values, dtype=float), "xrange")
                if yrange_values is not None:
                    _require_positive(np.asarray(yrange_values, dtype=float), "yrange")
                ax.set_xscale("log")
                ax.set_yscale("log")

            if symlog:
                symlog_values = _finite_values([finite_x_values, finite_y_values])
                if xrange_values is not None:
                    symlog_values = _finite_values([symlog_values, np.asarray(xrange_values, dtype=float)])
                if yrange_values is not None:
                    symlog_values = _finite_values([symlog_values, np.asarray(yrange_values, dtype=float)])
                linthresh = (
                    float(symlog_linthresh) if symlog_linthresh is not None else _infer_symlog_linthresh(symlog_values)
                )
                ax.set_xscale("symlog", linthresh=linthresh)
                ax.set_yscale("symlog", linthresh=linthresh)
                plot_payload["symlog_linthresh"] = linthresh

            if target_vs_predictions:
                shared_limits = _shared_target_prediction_limits(
                    finite_x_values,
                    finite_y_values,
                    xrange_values,
                    yrange_values,
                    loglog,
                )
                ax.set_xlim(shared_limits)
                ax.set_ylim(shared_limits)
                ax.set_box_aspect(1)
                ax.plot(
                    shared_limits,
                    shared_limits,
                    color="k",
                    linewidth=0.8,
                    linestyle="--",
                    label="y=x",
                    zorder=0,
                )
                plot_payload["axis_limits"] = shared_limits
            else:
                if xrange_values is not None:
                    ax.set_xlim(xrange_values)

                if yrange_values is not None:
                    ax.set_ylim(yrange_values)

            if show_yzero_axis:
                if loglog:
                    warnings.warn(
                        "show_yzero_axis=True is ignored when loglog=True.",
                        RuntimeWarning,
                    )
                else:
                    ax.axhline(y=0, color="k", linewidth=0.5)

            if grid:
                ax.set_axisbelow(True)
                ax.grid(True, which=grid_which, axis=grid_axis, **grid_style)

            if legend:
                legend_kwargs = {
                    "frameon": show_legend_frame,
                    "loc": legend_location_normalized,
                    "ncols": legend_ncols,
                }
                if legend_bbox_to_anchor is not None:
                    legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
                    legend_kwargs["bbox_transform"] = ax.transAxes
                if legend_handlelength is not None:
                    legend_kwargs["handlelength"] = legend_handlelength
                if legend_handletextpad is not None:
                    legend_kwargs["handletextpad"] = legend_handletextpad
                ax.legend(**legend_kwargs)

            for item in annotations:
                kwargs = {key: value for key, value in item.items() if key not in {"text", "x", "y", "coordinates"}}
                transform = ax.transAxes if item["coordinates"] == "axes fraction" else ax.transData
                ax.text(item["x"], item["y"], item["text"], transform=transform, **kwargs)

            fig.tight_layout()

            if savefig:
                save_kwargs = {"dpi": float(dpi)}
                if savefig_bbox_inches is not None:
                    save_kwargs["bbox_inches"] = savefig_bbox_inches
                fig.savefig(savefig, **save_kwargs)

            if figdata_fname is not None:
                ext = os.path.splitext(figdata_fname)[1].lower()
                if ext == ".json":
                    with open(figdata_fname, "w", encoding="utf-8") as f:
                        json.dump(plot_payload, f, indent=2)
                elif ext == ".npz":
                    out = {}
                    for i, s in enumerate(plot_payload["series"]):
                        out[f"x_{i}"] = np.asarray(s["x"], dtype=float)
                        out[f"y_{i}"] = np.asarray(s["y"], dtype=float)
                        out[f"label_{i}"] = np.asarray([s["label"]], dtype=object)
                        out[f"source_filepath_{i}"] = np.asarray([s["source_filepath"]], dtype=object)
                        out[f"xfactor_{i}"] = np.asarray(s["xfactor"])
                        out[f"yfactor_{i}"] = np.asarray(s["yfactor"])
                    out["plot_metadata"] = np.asarray(json.dumps(plot_payload))
                    np.savez(figdata_fname, **out)
                elif ext in (".txt", ".dat"):
                    rows = []
                    for i, s in enumerate(plot_payload["series"]):
                        for x, y in zip(s["x"], s["y"]):
                            rows.append([i, x, y])
                    factors = [{"xfactor": s["xfactor"], "yfactor": s["yfactor"]} for s in plot_payload["series"]]
                    header = "series_index x y\nseries_factors=" + json.dumps(factors)
                    np.savetxt(figdata_fname, np.asarray(rows, dtype=float), header=header)
                else:
                    raise ValueError(
                        f"Unsupported figdata_fname extension '{ext}'. Expected .npz, .json, .txt, or .dat."
                    )

            if show_plot:
                plt.show()
        finally:
            if not show_plot:
                plt.close(fig)


def _load_plot_source(source_filepath):
    try:
        return DerivativeStructures.from_file(source_filepath)
    except Exception as dss_error:
        try:
            return DerivativeStructures.from_structures_set_file(source_filepath)
        except Exception as sset_error:
            raise ValueError(
                f"Could not load '{source_filepath}' as a serialized "
                "DerivativeStructures or StructuresSet file. "
                f"DerivativeStructures error: {dss_error}. "
                f"StructuresSet error: {sset_error}"
            ) from sset_error
