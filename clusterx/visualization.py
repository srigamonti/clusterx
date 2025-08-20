# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from typing import Optional, List, Dict, Union
from numpy.typing import NDArray
import numpy as np
from ase.data import chemical_symbols as cs
import matplotlib.pyplot as plt
from clusterx.structures_set import StructuresSet
from clusterx.model import Model
from clusterx.visualization_utils import (
    scatter_plot,
    save_plot_data,
    auto_scale_yaxis,
    _set_rc_params,
    marker_styles,
)


def juview(plat, n=None):
    """Visualize structure object in Jupyter notebook

    **Parameters:**

    ``plat``: any of Atoms(ASE), ParentLattice, Supercell, Structure, StructureSet
        structure object to be plotted
    ``n``: integer or list of integer
        If integer, plot the first ``n`` structures. If list, plot range of structures.
        If ``None``, return all structures.

    **Return:**

        nglview display object
    """
    import nglview
    from clusterx.parent_lattice import ParentLattice
    from clusterx.structures_set import StructuresSet
    from clusterx.structures_set import Structure
    from clusterx.clusters.clusters_pool import ClustersPool

    if isinstance(plat, Structure):
        return _makeview([plat.get_atoms()])

    if isinstance(plat, ParentLattice):
        return _makeview(plat.get_all_atoms())

    if isinstance(plat, StructuresSet):
        if n is None:
            return _makeview(plat.get_images())
        if np.shape(n) == ():
            return _makeview(plat.get_images(n=n))
        if np.shape(n) == (2,):
            return _makeview(plat.get_images()[n[0] : n[1]])

    if isinstance(plat, ClustersPool):
        atoms = plat.get_cpool_atoms()
        if n is not None:
            if np.shape(n) == ():
                return _makeview(atoms[0:n])
            if np.shape(n) == (2,):
                return _makeview(atoms[n[0] : n[1]])
        else:
            return _makeview(atoms)

    if isinstance(plat, list):
        if n is not None:
            return _makeview(plat[0:n])
        else:
            return _makeview(plat)

    if not isinstance(plat, list):
        view = nglview.show_ase(plat)
        _juview_applystyle(view)
        # view.add_unitcell()
        # view.add_ball_and_stick()
        # view.parameters=dict(clipDist=0,color_scheme="element")
        return view


def _makeview(images):
    """Nglview setup for images arrays

    Parts taken from https://github.com/arose/nglview/issues/554

    ``images``: Array of Atoms (or descendant) objects
    """
    import nglview
    import math

    views = []
    for im in images:
        view = nglview.show_ase(im)
        view._remote_call("setSize", target="Widget", args=["300px", "300px"])
        _juview_applystyle(view)
        views.append(view)

    import ipywidgets

    # hboxes = [ipywidgets.HBox(views[i*3:i*3+3]) for i in range(int(math.ceil(len(views)/3.0)))]
    hboxes = [
        ipywidgets.HBox([views[j] for j in range(i * 3, min(i * 3 + 3, len(views)))])
        for i in range(int(math.ceil(len(views) / 3.0)))
    ]
    vbox = ipywidgets.VBox(hboxes)
    return vbox


def _juview_applystyle(view):
    view.parameters = dict(
        backgroundColor="white", clipDist=-100, color_scheme="element"
    )
    view.add_unitcell()
    view.camera = "orthographic"
    # view.center()
    view.control.rotate([0, 1, 0, 0])
    view.add_ball_and_stick()
    view.add_spacefill(radius_type="vdw", scale=0.3)


def plot_optimization_vs_number_of_clusters(
    clsel,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    yfactor=1.0,
    show_yzero_axis=True,
    show_plot=True,
    yaxis_label="Errors",
    fig_fname=None,
    data_fname=None,
):
    """Plot cluster optimization with matplotlib

    The plot shows the prediction and fitting errors as a function of the clusters
    pool size resulting from a cluster optimization done with a ClustersSelector object.

    The range of cluster pool sizes in the x-axis is determined by nclmin (minimum size)
    and nclmax (maximum size)

    **Parameters:**

    ``clsel``: ClustersSelector object
        The ClustersSelector oject which was used for the cluster optimization.

    ``xmin``: integer (Default: None)
        Minimum cluster size in x-axis.

    ``xmax``: integer (Default: None)
        Maximum cluster size in x-axis.

    ``yfactor``: float (Default:1.0)
        Multipliplicative factor for y-values. Useful to pass unit conversion factors.

    ``scale``: float (Default: 1.0)
        Adjust this parameter to change font size, axes line width, and other details of the plot.

    ``yaxis_label``: string (Default: "Errors")
        Label for the y-axis of the plot

    """
    from matplotlib.ticker import MaxNLocator

    _set_rc_params()

    set_sizes = sorted(clsel.set_sizes)
    indexes = [i[0] for i in sorted(enumerate(clsel.set_sizes), key=lambda x: x[1])]

    rmse = [clsel.rmse[ind] * yfactor for ind in indexes]
    cvs = [clsel.cvs[ind] * yfactor for ind in indexes]

    nclmax = max(set_sizes)
    nclmin = min(set_sizes)

    if xmin is None:
        xmin = nclmin

    if xmax is None:
        xmax = nclmax

    e_min = min([min(rmse), min(cvs)])
    e_max = max([max(rmse), max(cvs)])

    if ymin is None:
        ymin = e_min

    if ymax is None:
        ymax = e_max

    ncl_opt = set_sizes[cvs.index(min(cvs))]

    fig = plt.figure(figsize=(4.0, 3.0))
    ax = fig.add_axes([0.19, 0.16, 0.78, 0.80])

    ax.set_ylim([ymin, ymax])
    ax.set_xlim([xmin, xmax])

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.plot(
        [ncl_opt],
        [min(cvs)],
        "o",
        markeredgecolor="r",
        markerfacecolor="None",
        label="lowest RMSE-CV",
    )
    opt_s = None
    opt_cv = None
    opt_r = None
    if (clsel.clusters_sets == "combinations") or (
        clsel.clusters_sets == "size+combinations"
    ):
        opt_r = []
        opt_cv = []
        opt_s = []
        x = set_sizes[0]
        ormse = rmse[0]
        ocvs = cvs[0]
        for i, s in enumerate(set_sizes):
            if s == x:
                if cvs[i] < ocvs:
                    ocvs = cvs[i]
                    ormse = rmse[i]
            elif s > x:
                opt_s.append(x)
                opt_r.append(ormse)
                opt_cv.append(ocvs)
                x = s
                ormse = rmse[i]
                ocvs = cvs[i]

        opt_s.append(x)
        opt_r.append(ormse)
        opt_cv.append(ocvs)

        plt.plot(
            set_sizes,
            rmse,
            marker=".",
            color="blue",
            zorder=1,
            linestyle="",
            label="RMSE-fit",
        )
        plt.plot(
            set_sizes,
            cvs,
            marker=".",
            color="black",
            zorder=1,
            linestyle="",
            label="RMSE-CV",
        )
        plt.plot(opt_s, opt_r, marker=".", color="blue", zorder=1, linestyle="-")
        plt.plot(opt_s, opt_cv, marker=".", color="black", zorder=1, linestyle="-")
    else:
        ax.plot(
            set_sizes,
            rmse,
            marker=".",
            color="blue",
            zorder=1,
            linestyle="-",
            label="RMSE-fit",
        )
        ax.plot(
            set_sizes,
            cvs,
            marker=".",
            color="black",
            zorder=1,
            linestyle="-",
            label="RMSE-CV",
        )

    plt.ylabel(yaxis_label)
    plt.xlabel("Number of clusters")
    plt.legend()
    # leg=ax.legend(loc='best',borderaxespad=2,borderpad=2,labelspacing=1,handlelength=3, handletextpad=2)
    ax.legend(loc="upper center")

    if show_yzero_axis:
        ax.axhline(y=0, color="k", linewidth=0.5)

    if data_fname is not None:
        np.savez(
            data_fname,
            cluster_set_sizes=set_sizes,
            rmse_fit=rmse,
            rmse_cv=cvs,
            optimal_number_of_clusters=ncl_opt,
            optimal_rmse_cv=min(cvs),
            cluster_set_sizes_for_lowest_cv=opt_s,
            rmse_fit_for_lowest_cv=opt_r,
            rmse_cv_for_lowest_cv=opt_cv,
        )

    if fig_fname is not None:
        plt.savefig(fig_fname)

    if show_plot:
        plt.show()


def plot_optimization_vs_sparsity(
    clsel,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    xaxis_label="Sparsity",
    yaxis_label="Energy [arb. units]",
    show_plot=True,
    fname="plot_optimization_vs_sparsity",
):
    """Plot cluster optimization with matplotlib

    The plot shows the prediction and fitting errors as a function of the
    sparsity parameter when the LASSO method is used.

    **Parameters:**

    ``clsel``: ClustersSelector object
        The ClustersSelector oject which was used for the optimization to be plotted.
    """
    _set_rc_params()

    set_sparsity = clsel.lasso_sparsities

    rmse = clsel.rmse
    cvs = clsel.cvs

    nclmax = max(set_sparsity)
    nclmin = min(set_sparsity)
    print(nclmin, nclmax)
    if xmin is None:
        xmin = nclmin

    if xmax is None:
        xmax = nclmax

    e_min = min([min(rmse), min(cvs)])
    e_max = max([max(rmse), max(cvs)])

    if ymin is None:
        ymin = e_min

    if ymax is None:
        ymax = e_max

    opt = set_sparsity[clsel.cvs.index(min(cvs))]

    fig = plt.figure(figsize=(4.0, 3.0))
    ax = fig.add_axes([0.19, 0.16, 0.75, 0.80])

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    plt.semilogx(
        [opt],
        [min(cvs)],
        "o",
        markeredgecolor="r",
        markerfacecolor="None",
        label="lowest RMSE-CV",
    )

    plt.semilogx(
        set_sparsity,
        rmse,
        marker=".",
        color="blue",
        zorder=1,
        linestyle="-",
        label="RMSE-fit",
    )
    plt.semilogx(
        set_sparsity,
        cvs,
        marker=".",
        color="black",
        zorder=1,
        linestyle="-",
        label="RMSE-CV",
    )

    plt.ylabel(yaxis_label)
    plt.xlabel(xaxis_label)
    plt.legend()
    ax.legend(loc="best")

    plt.savefig(fname)
    if show_plot:
        plt.show()


def plot_predictions_vs_target(
    sset,
    cemodel,
    prop_name,
    scale=1.0,
    xaxis_label="Calculated energy [arb. units]",
    yaxis_label="Predicted energy [arb. units]",
):
    """Plot predictions versus target with matplotlib

    The plot shows the prediction versus the target

    **Parameters:**

    ``sset``: StructuresSet object
    ``cemodel``: Model object
    ``prop_name``: string
    """
    from matplotlib import rc
    import math

    energies = sset.get_property_values(property_name=prop_name)
    predictions = sset.get_predictions(cemodel)

    e_min = min([min(energies), min(predictions)])
    e_max = max([max(energies), max(predictions)])
    e_range = e_max - e_min

    width = 15.0 * scale
    fs = int(width * 1.8)
    ticksize = fs
    golden_ratio = (math.sqrt(5) - 0.9) / 2.0
    height = float(width * golden_ratio)

    plt.figure(figsize=(width, height))

    rc("axes", linewidth=3 * scale)

    plt.ylim(e_min - e_range / 8, e_max + e_range / 10)
    plt.xlim(e_min - e_range / 8, e_max + e_range / 10)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    ax = plt.gca()
    ax.tick_params(width=3 * scale, size=10 * scale, pad=10 * scale)

    plt.plot(
        [e_min - 2 * e_range / 10, e_max + 2 * e_range / 10],
        [e_min - 2 * e_range / 10, e_max + 2 * e_range / 10],
        marker="",
        color="black",
        zorder=1,
        linestyle="-",
        label="reference",
        linewidth=2.1 * scale,
    )

    plt.plot(
        energies,
        predictions,
        "o",
        markersize=15 * scale,
        markeredgewidth=2 * scale,
        markeredgecolor="b",
        markerfacecolor="None",
        label="structures",
    )
    # scatter([ncl_opt],[min(cv)], s=400,facecolors='none', edgecolors='r',)

    plt.ylabel(yaxis_label, fontsize=fs)
    plt.xlabel(xaxis_label, fontsize=fs)
    plt.legend()
    leg = ax.legend(
        loc="best",
        borderaxespad=2 * scale,
        borderpad=2 * scale,
        labelspacing=1 * scale,
        handlelength=3 * scale,
        handletextpad=2 * scale,
    )
    leg.get_frame().set_linewidth(3 * scale)

    for t in leg.get_texts():
        t.set_fontsize(fs)

    # plt.savefig("plot_optimization.png")
    plt.show()


def plot_property_vs_concentration(
    sset: StructuresSet,
    property_name: str,
    site_type: int = 0,
    sigma: int = 1,
    cemodel: Optional[Model] = None,
    show_loo_predictions: bool = True,
    sset_enum: Optional[StructuresSet] = None,
    properties_enum: Optional[np.ndarray] = None,
    concentrations_enum: Optional[np.ndarray] = None,
    sset_gss: Optional[StructuresSet] = None,
    show_plot: bool = True,
    refs: Union[List[float], NDArray[np.float64]] = [0.0, 0.0],
    yaxis_label: Optional[str] = None,
    show_yzero_axis: bool = True,
    data_fname: Optional[str] = None,
    fig_fname: Optional[str] = None,
) -> Dict[str, Union[np.ndarray, str]]:
    """Plot property values versus concentration and return dictionary with data

    The call to this functions generates a plot with matplotlib. It also returns a dictionary
    with the data used to generate the plot. This is useful in the case that the
    user wants to format the plot in a different way, or to write the data to a file
    for postprocessing (in the case that only the data is neeed, set ``show_plot`` to ``False``).

    **Parameters:**

    ``sset``: StructuresSet object
        The property values will be plotted for structures in ``sset``.
    ``property_name``: string
        The calculated property ``property_name`` will be extracted
        from the ``sset`` and depicted in the plot. If ``cemodel`` is not ``None``
        as well, then both predicted and calculated data are plot.
    ``site_type``: integer
        The x axis of the plot will indicate the fractional concentration for
        site type ``site_type``
    ``sigma``: integer
        The x axis of the plot will indicate the fractional concentration for
        the atomic species ``sigma`` in site type ``site_type``
    ``cemodel``: Model object
        If not ``None``, the property values as predicted by ``cemodel`` will be
        depicted.
    ``refs``: 1D Array containing two float (optional, default: None)
        the values of a reference energy at concentrations 0 and 1.
    ``show_loo_predictions``: Boolean
        If true, show predicted properties corresponding to leave-one-out CV.
        That is, the predictions for the left-out samples in the CV procedure are
        also plotted.

    **Returns:**

        Returns a dictionary with the data used to build the plot, with the following
        elements:

        - ``data["concentration"]``: Array of floats. The x-axis of the plot,
          containing the concentration of the substitutional species.
        - ``data["property"]``: Array of floats, same lenght as ``data["concentration"]``.
          Contains the values returned by ``sset.get_property_values(property_name = property_name)``.
        - ``data["predicted-property"]``: The prediced properties with ``cemodel``.
        - ``data["predicted-property-cv"]``: The prediced properties with ``cemodel`` on CV.
        - ``data["concentration-enum"]``: Concentrations for enumeration.
        - ``data["predicted-property-enum"]``: Predicted values for enumeration.
        - ``data["concentration-gss"]``: Concentrations for ground-state search.
        - ``data["predicted-property-gss"]``: Predicted values for ground-state-search.

        Depending on the arguments to the call to ``plot_property_vs_concentration``, some of the
        returned dictionary elements may be missing.

    """
    _set_rc_params()

    data = {}
    fig, ax = plt.subplots()

    if isinstance(refs, list):
        refs = np.array(refs, dtype=np.float64)

    energies = (
        np.array(sset.get_property_values(property_name=property_name))
        if sset is not None
        else None
    )
    predictions = (
        np.array(sset.get_predictions(cemodel)) if cemodel is not None else None
    )
    pred_enum = (
        np.array(sset_enum.get_predictions(cemodel))
        if sset_enum is not None
        else properties_enum
    )

    frconc = sset.get_concentrations(site_type, sigma) if sset is not None else None
    vl_en = (
        refs[0] * (1 - np.array(frconc)) + np.array(frconc) * refs[1]
        if sset is not None
        else None
    )

    frconc_enum = (
        sset_enum.get_concentrations(site_type, sigma)
        if sset_enum is not None
        else concentrations_enum
    )
    vl_en_enum = (
        refs[0] * (1 - np.array(frconc_enum)) + np.array(frconc_enum) * refs[1]
        if frconc_enum is not None
        else None
    )

    if cemodel is not None and show_loo_predictions:
        cvs = cemodel.get_cv_score(sset)
        pred_cv = np.array(cvs["Predictions-CV"])
    else:
        pred_cv = None

    frconc_gss = (
        sset_gss.get_concentrations(site_type, sigma) if sset_gss is not None else None
    )
    pred_gss = (
        np.array(sset_gss.get_predictions(cemodel)) if sset_gss is not None else None
    )
    vl_en_gss = (
        refs[0] * (1 - np.array(frconc_gss)) + np.array(frconc_gss) * refs[1]
        if frconc_gss is not None
        else None
    )

    data["concentration"] = frconc
    data["property"] = energies - vl_en if sset is not None else None
    data["predicted-property"] = (
        predictions - vl_en if predictions is not None else None
    )
    data["predicted-property-cv"] = pred_cv - vl_en if pred_cv is not None else None
    data["concentration-enum"] = frconc_enum
    data["predicted-property-enumeration"] = (
        pred_enum - vl_en_enum if pred_enum is not None else None
    )
    data["concentration-gss"] = frconc_gss
    data["predicted-property-gss"] = (
        pred_gss - vl_en_gss if pred_gss is not None else None
    )

    _ = (
        scatter_plot(
            ax, frconc, energies - vl_en, marker_styles["Calculated"], "Calculated"
        )
        if sset is not None
        else None
    )
    if predictions is not None:
        scatter_plot(
            ax,
            frconc,
            predictions - vl_en,
            marker_styles["Predicted-fit"],
            "Predicted-fit",
        )
    if pred_cv is not None:
        scatter_plot(
            ax,
            frconc,
            pred_cv - vl_en,
            marker_styles["Predicted-CV"],
            "Predicted-CV",
        )
    if pred_enum is not None:
        scatter_plot(
            ax,
            frconc_enum,
            pred_enum - vl_en_enum,
            marker_styles["Enumeration"],
            "Enumeration",
        )
    if pred_gss is not None:
        scatter_plot(
            ax,
            frconc_gss,
            pred_gss - vl_en_gss,
            marker_styles["Predicted GS"],
            "Predicted GS",
        )

    species_name = (
        sset.get_parent_lattice().get_sublattice_types()[site_type][sigma]
        if sset is not None
        else 0
    )
    xlabel = (
        f"Concentration of {cs[species_name]}"
        if species_name != 0
        else "Concentration of substituent"
    )
    plt.xlabel(xlabel)
    plt.ylabel(yaxis_label if yaxis_label else property_name)
    data["xlabel"] = xlabel

    ax.set_ylim(auto_scale_yaxis(data["property"], data))

    if show_yzero_axis:
        ax.axhline(y=0, color="k", linewidth=0.5)

    plt.legend()
    save_plot_data(data, data_fname)
    if fig_fname is not None:
        plt.savefig(fig_fname)
    if show_plot:
        plt.show()

    plt.close()
    return data


def plot_property(
    xvalues,
    yvalues,
    prop_name=None,
    xaxis_label=None,
    yaxis_label=None,
    show_plot=True,
    scale=1.0,
):
    """yvalues versus xvalues"""
    import math
    from matplotlib import rc, rcParams

    width = 15.0 * scale
    fs = int(width * 1.8)
    ticksize = fs
    golden_ratio = (math.sqrt(5) - 0.9) / 2.0
    height = float(width * golden_ratio)

    rc("axes", linewidth=3 * scale)

    fig = plt.figure(figsize=(width, height))
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    ax = plt.gca()
    ax.tick_params(width=3 * scale, size=10 * scale, pad=10 * scale)

    rcParams["savefig.format"] = "png"

    plt.plot(
        xvalues,
        yvalues,
        marker=".",
        color="b",
        markersize=15 * scale,
        markeredgewidth=2.0 * scale,
        linewidth=2.2 * scale,
        label=prop_name,
    )
    if xaxis_label is not None:
        plt.xlabel(xaxis_label, fontsize=fs)
    if yaxis_label is not None:
        plt.ylabel(yaxis_label, fontsize=fs)

    if prop_name is not None:
        plt.legend()
        leg = ax.legend(
            loc="best",
            borderaxespad=scale,
            borderpad=scale,
            labelspacing=1 * scale,
            handlelength=2 * scale,
            handletextpad=scale,
            fontsize=fs,
        )
        leg.get_frame().set_linewidth(3 * scale)

    if show_plot:
        plt.show()
    else:
        plt.savefig(prop_name + "_plot.png")


def _wls_normalize_histogram_for_plotting(histogram, shift_y_first_nonzero=False):
    hist = histogram.copy()
    nbins = len(hist)
    mean = 0
    n_nonzero = 0

    if shift_y_first_nonzero:
        for i in range(nbins):
            if hist[i] != 0:
                hfirst = hist[i]
                break
        for i in range(nbins):
            if hist[i] != 0:
                hist[i] -= hfirst

    for i in range(nbins):
        if hist[i] != 0:
            mean += hist[i]
            n_nonzero += 1

    if n_nonzero == 0:
        return hist
    else:
        mean /= n_nonzero
        for i in range(nbins):
            hist[i] /= np.abs(mean)

        return hist


def plot_histograms_wang_landau(cdos_object, index=-1):
    """Plot histograms for Wang-Landau"""

    hist = np.array(cdos_object._stored_cdos[index]["histogram"])
    cdos = np.array(cdos_object._stored_cdos[index]["cdos"])

    ener_arr = np.array(cdos_object._energy_bins)
    cdos_arr = _wls_normalize_histogram_for_plotting(cdos, shift_y_first_nonzero=True)
    hist_arr = _wls_normalize_histogram_for_plotting(hist)
    ones_arr = np.ones(len(ener_arr))

    figure, ax = plt.subplots(figsize=(10, 8))

    ax.bar(
        ener_arr, ones_arr, width=cdos_object._energy_bin_width * 0.80, color="silver"
    )
    ax.bar(ener_arr, hist_arr, width=cdos_object._energy_bin_width * 0.80)
    ax.bar(ener_arr, cdos_arr, width=cdos_object._energy_bin_width * 0.40)

    plt.show()
