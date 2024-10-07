import numpy as np


def scatter_plot(ax, x_data, y_data, marker, color, label):
    """Helper function to create a scatter plot."""
    ax.scatter(
        x_data,
        y_data,
        marker=marker,
        s=15,
        edgecolors=color,
        facecolors="none",
        label=label,
    )


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

    rcParams["lines.markersize"] = 6

    rcParams["xtick.major.pad"] = 1.0
    rcParams["ytick.major.pad"] = 1.0
    rcParams["axes.labelpad"] = 4.0

    rcParams["figure.subplot.left"] = 0.15
    rcParams["figure.subplot.right"] = 0.98
    rcParams["figure.subplot.bottom"] = 0.15
    rcParams["figure.subplot.top"] = 0.98
    rcParams["figure.subplot.wspace"] = 0.2
    rcParams["figure.subplot.hspace"] = 0.2
