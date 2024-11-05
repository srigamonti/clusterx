import numpy as np


marker_styles = {
    "Calculated": {"marker": "o", "color": "k", "size": 13, "linewidth": 0.65},
    "Predicted-fit": {"marker": "o", "color": "k", "size": 6, "linewidth": 0.65},
    "Predicted-CV": {"marker": "o", "color": "r", "size": 6, "linewidth": 0.65},
    "Enumeration": {"marker": ".", "color": "gray", "size": 8, "linewidth": 0},
    "Predicted-GS": {"marker": "o", "color": "green", "size": 6, "linewidth": 0.65},
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
        "edgecolors": marker_style["color"],
        "facecolors": "none",
        "linewidth": marker_style["linewidth"],
        "label": label,
    }

    # Adjust scatter arguments if the marker is "."
    if marker_style["marker"] == ".":
        scatter_args["facecolors"] = marker_style["color"]
        scatter_args["edgecolors"] = "none"

    ax.scatter(**scatter_args)


def save_plot_data(data, filename):
    """Helper function to save plot data to a file."""
    if filename:
        np.savez(
            filename,
            concentrations_property=data["concentration"],
            property=data["property"],
            predictions=data["predicted-property"],
            predictions_cv=data["predicted-property-cv"],
            concentrations_enum=data["concentration-enum"],
            predictions_enum=data["predicted-property-enumeration"],
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
