import json
import os

import numpy as np

marker_styles = {
    "Calculated": {
        "marker": "o",
        "edgecolor": "k",
        "facecolor": "none",
        "size": 20,
        "linewidth": 0.65,
    },
    "Predicted-fit": {
        "marker": "o",
        "edgecolor": "none",
        "facecolor": "k",
        "size": 10,
        "linewidth": 0.65,
    },
    "Predicted-CV": {
        "marker": "o",
        "edgecolor": "r",
        "facecolor": "none",
        "size": 10,
        "linewidth": 0.65,
    },
    "Enumeration": {
        "marker": ".",
        "edgecolor": "none",
        "facecolor": "gray",
        "size": 10,
        "linewidth": 0,
    },
    "Predicted-GS": {
        "marker": "o",
        "edgecolor": "green",
        "facecolor": "none",
        "size": 6,
        "linewidth": 0.65,
    },
}


def scatter_plot(
    ax,
    x_data,
    y_data,
    marker_style=None,
    label=None,
):
    """Helper function to create a scatter plot."""

    # Default symbol configuration if none is provided
    if marker_style is None:
        marker_style = {"marker": ".", "color": "blue", "size": 15, "linewidth": 1.0}

    scatter_args = {
        "x": x_data,
        "y": y_data,
        "marker": marker_style["marker"],
        "s": marker_style["size"],
        "edgecolors": marker_style["edgecolor"],
        "facecolors": marker_style["facecolor"],
        "linewidth": marker_style["linewidth"],
        "label": label,
    }

    # Adjust scatter arguments if the marker is "."
    if marker_style["marker"] == ".":
        scatter_args["facecolors"] = marker_style["facecolor"]
        scatter_args["edgecolors"] = "none"

    ax.scatter(**scatter_args)


def save_plot_data(data, filename):
    """Helper function to save plot data to a file, depending on extension."""
    if not filename:
        return

    # Extract extension
    ext = os.path.splitext(filename)[1].lower()

    # Define mapping for consistency
    fields = {
        "concentrations_property": data["concentration"],
        "concentrations_enum": data["concentration-enum"],
        "property": data["property"],
        "predictions": data["predicted-property"],
        "predictions_cv": data["predicted-property-cv"],
        "predictions_enum": data["predicted-property-enumeration"],
    }

    if ext == ".npz":
        np.savez(filename, **fields)

    elif ext == ".json":
        # Convert numpy arrays to lists for JSON serialization
        json_ready = {k: np.asarray(v).tolist() for k, v in fields.items()}
        with open(filename, "w") as f:
            json.dump(json_ready, f, indent=2)

    elif ext in (".dat", ".txt"):
        # Flatten arrays and discard empty ones
        flattened = {k: np.ravel(v) for k, v in fields.items() if len(np.ravel(v)) > 1}

        if not flattened:
            raise ValueError("No non-empty data arrays to save.")

        # Check that remaining arrays have consistent lengths
        lengths = [len(v) for v in flattened.values()]
        if len(set(lengths)) != 1:
            raise ValueError(
                "All non-empty data arrays must have the same length to save as text."
            )

        # Stack and save with headers
        arr = np.column_stack(list(flattened.values()))
        header = " ".join(flattened.keys())
        np.savetxt(filename, arr, header=header, fmt="%.6g")

    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Expected .npz, .json, .dat, or .txt."
        )


def auto_scale_yaxis(property_data, data):
    """Helper function to auto scale the y-axis."""
    ymin = None
    ymax = None

    keys = [
        "property",
        "predicted-property",
        "predicted-property-cv",
        "predicted-property-enumeration",
        "predicted-property-gss",
    ]
    for key in keys:
        values = data.get(key)
        if values is not None:
            key_min = np.amin(values)
            key_max = np.amax(values)

            if ymin is None or ymax is None:
                ymin, ymax = key_min, key_max
            else:
                ymin = min(ymin, key_min)
                ymax = max(ymax, key_max)

    dy = ymax - ymin
    return [ymin - 0.1 * dy, ymax + 0.3 * dy]


def _set_rc_params():
    from matplotlib import rcParams

    rcParams["figure.figsize"] = (4.0, 3.0)
    rcParams["figure.dpi"] = 300
    rcParams["savefig.format"] = "png"
    rcParams["xtick.major.size"] = 2.5  # major tick size in points
    rcParams["xtick.minor.size"] = 1.1  # minor tick size in points
    rcParams["xtick.major.width"] = 1.5  # major tick width in points
    rcParams["xtick.minor.width"] = 0.6  # minor tick width in points
    rcParams["ytick.major.size"] = 2.5  # major tick size in points
    rcParams["ytick.minor.size"] = 1.1  # minor tick size in points
    rcParams["ytick.major.width"] = 1.5  # major tick width in points
    rcParams["ytick.minor.width"] = 0.6  # minor tick width in points
    rcParams["lines.linewidth"] = 2.0
    rcParams["lines.markersize"] = 6
    rcParams["lines.markeredgewidth"] = 1
    rcParams["xtick.labelsize"] = 11
    rcParams["ytick.labelsize"] = 11
    rcParams["axes.formatter.useoffset"] = False

    rcParams["axes.titlesize"] = 24
    rcParams["axes.labelsize"] = 11
    rcParams["axes.labelpad"] = 2.0
    rcParams["axes.linewidth"] = 1.1

    rcParams["legend.fontsize"] = 10
    rcParams["legend.frameon"] = True
    rcParams["legend.framealpha"] = 1.0
    rcParams["legend.handletextpad"] = 0.35
    rcParams["legend.labelspacing"] = 0.15
    rcParams["legend.borderpad"] = 0.30
    rcParams["legend.edgecolor"] = "0.0"

    rcParams["xtick.major.width"] = 1.0  # major tick width in points
    rcParams["xtick.minor.width"] = 0.3  # minor tick width in points
    rcParams["ytick.major.width"] = 1.0  # major tick width in points
    rcParams["ytick.minor.width"] = 0.3  # minor tick width in points

    rcParams["xtick.major.pad"] = 1.0
    rcParams["ytick.major.pad"] = 1.0
    rcParams["axes.labelpad"] = 4.0

    rcParams["figure.subplot.left"] = 0.15
    rcParams["figure.subplot.right"] = 0.98
    rcParams["figure.subplot.bottom"] = 0.15
    rcParams["figure.subplot.top"] = 0.98
    rcParams["figure.subplot.wspace"] = 0.2
    rcParams["figure.subplot.hspace"] = 0.2
