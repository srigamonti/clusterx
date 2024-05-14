# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np

def juview(plat,n=None):
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

    if isinstance(plat,Structure):
        return _makeview( [ plat.get_atoms() ] )

    if isinstance(plat,ParentLattice):
        return _makeview(plat.get_all_atoms())

    if isinstance(plat,StructuresSet):
        if n is None:
            return _makeview(plat.get_images())
        if np.shape(n) == ():
            return _makeview(plat.get_images(n=n))
        if np.shape(n) == (2,):
            return _makeview(plat.get_images()[n[0]:n[1]])

    if isinstance(plat,ClustersPool):
        atoms = plat.get_cpool_atoms()
        if n is not None:
            if np.shape(n) == ():
                return _makeview(atoms[0:n])
            if np.shape(n) == (2,):
                return _makeview(atoms[n[0]:n[1]])
        else:
            return _makeview(atoms)

    if isinstance(plat,list):
        if n is not None:
            return _makeview(plat[0:n])
        else:
            return _makeview(plat)

    if not isinstance(plat,list):
        view = nglview.show_ase(plat)
        _juview_applystyle(view)
        #view.add_unitcell()
        #view.add_ball_and_stick()
        #view.parameters=dict(clipDist=0,color_scheme="element")
        return view

def _makeview(images):
    """Nglview setup for images arrays

    Parts taken from https://github.com/arose/nglview/issues/554

    ``images``: Array of Atoms (or descendant) objects
    """
    import nglview, math
    views = []
    for im in images:
        view = nglview.show_ase(im)
        view._remote_call("setSize", target="Widget", args=["300px","300px"])
        _juview_applystyle(view)
        views.append(view)

    import ipywidgets
    #hboxes = [ipywidgets.HBox(views[i*3:i*3+3]) for i in range(int(math.ceil(len(views)/3.0)))]
    hboxes = [ipywidgets.HBox([views[j] for j in range(i*3,min(i*3+3,len(views)))]) for i in range(int(math.ceil(len(views)/3.0)))]
    vbox = ipywidgets.VBox(hboxes)
    return vbox

def _juview_applystyle(view):
    view.parameters = dict(backgroundColor='white',clipDist=-100,color_scheme="element")
    view.add_unitcell()
    view.camera = 'orthographic'
    #view.center()
    view.control.rotate([0,1,0,0])
    view.add_ball_and_stick()
    view.add_spacefill(radius_type='vdw',scale=0.3)


def plot_optimization_vs_number_of_clusters(
        clsel,
        xmin = None,
        xmax = None,
        ymin = None,
        ymax = None,
        yfactor = 1.0,
        scale = 1.0,
        show_yzero_axis = True,
        show_plot = True,
        yaxis_label = "Errors",
        fig_fname=None,
        data_fname=None):
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
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    _set_rc_params()

    set_sizes = sorted(clsel.set_sizes)
    indexes = [i[0] for i in sorted(enumerate(clsel.set_sizes), key=lambda x:x[1])]

    rmse = [clsel.rmse[ind] * yfactor for ind in indexes]
    cvs = [clsel.cvs[ind] * yfactor for ind in indexes]

    nclmax = max(set_sizes)
    nclmin = min(set_sizes)

    if xmin is None:
        xmin = nclmin

    if xmax is None:
        xmax = nclmax
         
    e_min = min([min(rmse),min(cvs)])
    e_max = max([max(rmse),max(cvs)])
    
    if ymin is None:
        ymin = e_min

    if ymax is None:
        ymax = e_max

    ncl_opt = set_sizes[cvs.index(min(cvs))]

    fig = plt.figure(figsize=(4.0,3.0))
    ax = fig.add_axes([0.19, 0.16, 0.78, 0.80])

    ax.set_ylim([ymin, ymax])
    ax.set_xlim([xmin, xmax])

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.plot([ncl_opt],[min(cvs)], 'o',markeredgecolor='r', markerfacecolor='None' , label='lowest RMSE-CV' )
    opt_s = None
    opt_cv = None
    opt_r = None
    if (clsel.clusters_sets=="combinations") or (clsel.clusters_sets=="size+combinations"):

        opt_r=[]
        opt_cv=[]
        opt_s=[]
        x=set_sizes[0]
        ormse=rmse[0]
        ocvs=cvs[0]
        for i,s in enumerate(set_sizes):
            if s == x:
                if cvs[i] < ocvs:
                    ocvs=cvs[i]
                    ormse=rmse[i]
            elif s > x:
                opt_s.append(x)
                opt_r.append(ormse)
                opt_cv.append(ocvs)
                x=s
                ormse=rmse[i]
                ocvs=cvs[i]

        opt_s.append(x)
        opt_r.append(ormse)
        opt_cv.append(ocvs)

        plt.plot(set_sizes, rmse, marker='.', color='blue', zorder=1,  linestyle='',label='RMSE-fit')
        plt.plot(set_sizes, cvs, marker='.', color='black', zorder=1, linestyle='',label='RMSE-CV')
        plt.plot(opt_s, opt_r, marker='.', color='blue', zorder=1,  linestyle='-')
        plt.plot(opt_s, opt_cv, marker='.',  color='black', zorder=1, linestyle='-')
    else:
        ax.plot(set_sizes, rmse, marker='.', color='blue', zorder=1,  linestyle='-',label='RMSE-fit')
        ax.plot(set_sizes, cvs, marker='.', color='black', zorder=1, linestyle='-',label='RMSE-CV')

    plt.ylabel(yaxis_label )
    plt.xlabel('Number of clusters')
    plt.legend()
    #leg=ax.legend(loc='best',borderaxespad=2,borderpad=2,labelspacing=1,handlelength=3, handletextpad=2)
    ax.legend(loc='upper center')

        
    if show_yzero_axis:
        ax.axhline(y=0, color='k', linewidth=0.5)

    if data_fname is not None:
        np.savez(
            data_fname,
            cluster_set_sizes = set_sizes,
            rmse_fit = rmse,
            rmse_cv = cvs,
            optimal_number_of_clusters = ncl_opt,
            optimal_rmse_cv = min(cvs),
            cluster_set_sizes_for_lowest_cv = opt_s,
            rmse_fit_for_lowest_cv = opt_r,
            rmse_cv_for_lowest_cv = opt_cv,
            )
    
    if fig_fname is not None:
        plt.savefig(fig_fname)

    if show_plot:
        plt.show()

    
def plot_optimization_vs_sparsity(
        clsel,
        xmin = None,
        xmax = None,
        ymin = None,
        ymax = None,
        xaxis_label = 'Sparsity',
        yaxis_label = "Energy [arb. units]",
        show_plot = True,
        fname = "plot_optimization_vs_sparsity"):
    """Plot cluster optimization with matplotlib

    The plot shows the prediction and fitting errors as a function of the

    **Parameters:**

    ``clsel``: ClustersSelector object
        The ClustersSelector oject which was used for the optimization to be plotted.
        sparsity parameter when the LASSO method is used.
    """

    import matplotlib.pyplot as plt

    _set_rc_params()

    set_sparsity=clsel.lasso_sparsities

    rmse=clsel.rmse
    cvs=clsel.cvs

    nclmax=max(set_sparsity)
    nclmin=min(set_sparsity)
    print(nclmin,nclmax)
    if xmin is None:
        xmin = nclmin

    if xmax is None:
        xmax = nclmax
         
    e_min = min([min(rmse),min(cvs)])
    e_max = max([max(rmse),max(cvs)])
    
    if ymin is None:
        ymin = e_min

    if ymax is None:
        ymax = e_max

    opt=set_sparsity[clsel.cvs.index(min(cvs))]

    fig = plt.figure(figsize=(4.0,3.0))
    ax = fig.add_axes([0.19, 0.16, 0.75, 0.80])

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    
    plt.semilogx([opt],[min(cvs)], 'o', markeredgecolor='r', markerfacecolor='None', label='lowest RMSE-CV')

    plt.semilogx(set_sparsity, rmse, marker='.', color='blue', zorder=1,  linestyle='-',label='RMSE-fit')
    plt.semilogx(set_sparsity, cvs, marker='.', color='black', zorder=1, linestyle='-',label='RMSE-CV')

    plt.ylabel(yaxis_label)
    plt.xlabel(xaxis_label)
    plt.legend()
    ax.legend(loc='best')

    plt.savefig(fname)
    if show_plot:
        plt.show()


def plot_predictions_vs_target(sset, cemodel, prop_name, scale=1.0, xaxis_label = 'Calculated energy [arb. units]', yaxis_label = "Predicted energy [arb. units]"):
    """Plot predictions versus target with matplotlib

    The plot shows the prediction versus the target

    **Parameters:**

    ``sset``: StructuresSet object
    ``cemodel``: Model object
    ``prop_name``: string
    """
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from matplotlib import rc,rcParams
    import math

    energies = sset.get_property_values(property_name = prop_name)
    predictions = sset.get_predictions(cemodel)

    e_min=min([min(energies),min(predictions)])
    e_max=max([max(energies),max(predictions)])
    e_range=e_max - e_min

    width=15.0*scale
    fs=int(width*1.8)
    ticksize = fs
    golden_ratio = (math.sqrt(5) - 0.9) / 2.0
    height = float(width * golden_ratio)

    plt.figure(figsize=(width,height))

    rc('axes', linewidth=3*scale)

    plt.ylim(e_min-e_range/8,e_max+e_range/10)
    plt.xlim(e_min-e_range/8,e_max+e_range/10)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    ax = plt.gca()
    ax.tick_params(width=3*scale,size=10*scale,pad=10*scale)

    plt.plot( [e_min-2*e_range/10, e_max+2*e_range/10 ], [e_min-2*e_range/10, e_max+2*e_range/10 ], marker='', color='black', zorder=1,  linestyle='-',label='reference', linewidth=2.1*scale)

    plt.plot(energies,predictions, 'o', markersize=15*scale,  markeredgewidth=2*scale,markeredgecolor='b', markerfacecolor='None' , label="structures")
    #scatter([ncl_opt],[min(cv)], s=400,facecolors='none', edgecolors='r',)

    plt.ylabel(yaxis_label ,fontsize=fs)
    plt.xlabel(xaxis_label ,fontsize=fs)
    plt.legend()
    leg=ax.legend(loc='best',borderaxespad=2*scale,borderpad=2*scale,labelspacing=1*scale,handlelength=3*scale, handletextpad=2*scale)
    leg.get_frame().set_linewidth(3*scale)

    for l in leg.get_texts():
        l.set_fontsize(fs)

    #plt.savefig("plot_optimization.png")
    plt.show()

def _set_rc_params():
    from matplotlib import rcParams
    
    rcParams['figure.figsize'] = (4.0,3.0)
    rcParams['figure.dpi'] = 300
    rcParams['savefig.format'] = 'png'
    rcParams['xtick.major.size'] =    2.5     # major tick size in points
    rcParams['xtick.minor.size'] =    1.1       # minor tick size in points
    rcParams['xtick.major.width'] =   1.5     # major tick width in points
    rcParams['xtick.minor.width'] =   0.6     # minor tick width in points
    rcParams['ytick.major.size'] =    2.5     # major tick size in points
    rcParams['ytick.minor.size'] =    1.1       # minor tick size in points
    rcParams['ytick.major.width'] =   1.5     # major tick width in points
    rcParams['ytick.minor.width'] =   0.6     # minor tick width in points
    rcParams['lines.linewidth'] = 2.0
    rcParams['lines.markersize'] = 6
    rcParams['xtick.labelsize'] = 11
    rcParams['ytick.labelsize'] = 11
    rcParams['axes.formatter.useoffset'] = False
    
    rcParams['axes.titlesize'] = 24
    rcParams['axes.labelsize'] = 11
    rcParams['axes.labelpad'] = 2.0
    rcParams['axes.linewidth'] = 1.1
    
    rcParams['legend.fontsize'] = 10
    rcParams['legend.frameon'] = True
    rcParams['legend.framealpha'] = 1.0
    rcParams['legend.handletextpad'] = 0.35
    rcParams['legend.labelspacing'] = 0.15
    rcParams['legend.borderpad'] = 0.30
    rcParams['legend.edgecolor'] = '0.0'

    rcParams['xtick.major.width'] =   1.0     # major tick width in points
    rcParams['xtick.minor.width'] =   0.3     # minor tick width in points
    rcParams['ytick.major.width'] =   1.0     # major tick width in points
    rcParams['ytick.minor.width'] =   0.3     # minor tick width in points

    rcParams['lines.markersize'] = 6

    rcParams['xtick.major.pad'] = 1.0
    rcParams['ytick.major.pad'] = 1.0
    rcParams['axes.labelpad'] = 4.0

def plot_property_vs_concentration(sset,
                                   site_type=0,
                                   sigma=1,
                                   cemodel=None,
                                   property_name=None,
                                   show_loo_predictions=True,
                                   sset_enum=None,
                                   properties_enum=None,
                                   concentrations_enum=None,
                                   sset_gss=None,
                                   show_plot = True,
                                   refs=None,
                                   scale=1.0,
                                   yaxis_label = None,
                                   yfactor = 1.0,
                                   show_yzero_axis = True,
                                   data_fname=None,
                                   fig_fname=None):
    """Plot property values versus concentration and return dictionary with data

    The call to this functions generates a plot with matplotlib. It also returns a dictionary
    with the data used to generate the plot. This is useful in the case that the
    user wants to format the plot in a different way, or to write the data to a file
    for postprocessing (in the case that only the data is neeed, set ``show_plot`` to ``False``).

    **Parameters:**

    ``sset``: StructuresSet object
        The property values will be plotted for structures in ``sset``.
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
    ``property_name``: string
        If not ``None``, the calculated property ``property_name`` will be extracted
        from the ``sset`` and depicted in the plot. If ``cemodel`` is not ``None``
        as well, then both predicted and calculated data are plot.
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
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from clusterx.utils import findmax, findmin
    
    _set_rc_params()

    data = {}

    fig = plt.figure(figsize=(4.0,3.0))

    ax = fig.add_axes([0.19, 0.16, 0.78, 0.80])
    
    if refs is None:
        refs = [0.0,0.0]
    else:
        refs = np.array(refs) * yfactor 
        
    energies = np.array(sset.get_property_values(property_name = property_name)) * yfactor
    if cemodel is not None:
        predictions = np.array(sset.get_predictions(cemodel)) * yfactor

    if sset_enum is not None:
        pred_enum = np.array(sset_enum.get_predictions(cemodel)) * yfactor
    
    if properties_enum is not None and sset_enum is None:
        pred_enum = properties_enum
        frconc_enum = concentrations_enum
        vl_en_enum = refs[0]*(1-np.array(frconc_enum)) + np.array(frconc_enum)*refs[1]

    if sset_gss is not None:
        pred_gss = np.array(sset_gss.get_predictions(cemodel)) * yfactor

    pred_cv = None
    if show_loo_predictions and cemodel is not None:
        cvs = cemodel.get_cv_score(sset)
        pred_cv = np.array(cvs["Predictions-CV"]) * yfactor 

    frconc = sset.get_concentrations(site_type,sigma)
    vl_en = refs[0]*(1-np.array(frconc)) + np.array(frconc)*refs[1]

    if sset_enum is not None:
        frconc_enum = sset_enum.get_concentrations(site_type,sigma)
        vl_en_enum = refs[0]*(1-np.array(frconc_enum)) + np.array(frconc_enum)*refs[1]
    if sset_gss is not None:
        frconc_gss = sset_gss.get_concentrations(site_type,sigma)
        vl_en_gss = refs[0]*(1-np.array(frconc_gss)) + np.array(frconc_gss)*refs[1]

    data["concentration-enum"] = None
    data["predicted-property-enumeration"] = None
    data["predicted-property"] = None
    data["predicted-property-cv"] = None
    data["concentration"] = frconc
    data["property"] = energies-vl_en
    ymax = np.amax(data["property"])
    ymin = np.amin(data["property"])
    ax.scatter(frconc,energies-vl_en,marker='o', s=25, zorder=0, facecolors='none', edgecolors='k',label='Calculated')
    if cemodel is not None and pred_cv is not None:
        data["predicted-property"] = predictions-vl_en
        data["predicted-property-cv"] = pred_cv-vl_en
        plt.scatter(frconc,predictions-vl_en,marker='.', s=20, zorder=1, facecolors='k', edgecolors=None,label='Predicted-fit')
        plt.scatter(frconc,pred_cv-vl_en,marker='.', s=10, zorder=2, facecolors='red', edgecolors=None,label='Predicted-CV')
        ymax = findmax(ymax,data["predicted-property"],data["predicted-property-cv"])
        ymin = findmin(ymin,data["predicted-property"],data["predicted-property-cv"])
    if cemodel is not None and pred_cv is None:
        data["predicted-property"] = predictions-vl_en
        plt.scatter(frconc,predictions-vl_en,marker='o', s=20, edgecolors='none', facecolors='blue',label='Predicted-fit')
        ymax = findmax(ymax,data["predicted-property"])
        ymin = findmin(ymin,data["predicted-property"])
    if sset_enum is not None or properties_enum is not None:
        data["concentration-enum"] = frconc_enum
        data["predicted-property-enumeration"] = pred_enum-vl_en_enum
        plt.scatter(frconc_enum,pred_enum-vl_en_enum,marker='o', edgecolors='none', facecolors='gray',label='Enumeration')
        ymax = findmax(ymax,data["predicted-property-enumeration"])
        ymin = findmin(ymin,data["predicted-property-enumeration"])
    if sset_gss is not None:
        data["concentration-gss"] = frconc_gss
        data["predicted-property-gss"] = pred_gss-vl_en_gss
        plt.scatter(frconc_gss,pred_gss-vl_en_gss,marker='o', edgecolors='none', facecolors='red',label='Predicted GS')
        ymax = findmax(ymax,data["predicted-property-gss"])
        ymin = findmin(ymin,data["predicted-property-gss"])

    from ase.data import chemical_symbols as cs
    cs = np.array(cs)
    sublattice_types = sset.get_parent_lattice().get_sublattice_types()
    species_name = sublattice_types[site_type][sigma]
    xlabel = "Concentration of "+cs[species_name]
    plt.xlabel(xlabel)
    if yaxis_label is not None:
        plt.ylabel(yaxis_label)
    else:
        plt.ylabel(property_name)

    data["xlabel"] = xlabel

    dy = ymax-ymin
    ax.set_ylim([ymin - 0.1 * dy, ymax + 0.30 * dy])
    
    if show_yzero_axis:
        ax.axhline(y=0, color='k', linewidth=0.5)
    
    plt.legend()

    if data_fname is not None:
        np.savez(
            data_fname,
            concentrations_property = data["concentration"],
            property = data["property"],
            predictions = data['predicted-property'],
            predictions_cv = data['predicted-property-cv'],
            concentrations_enum = data["concentration-enum"],
            predictions_enum = data["predicted-property-enumeration"]
            )

    if show_plot:
        plt.show()

    if fig_fname is not None:
        plt.savefig(fig_fname) 

    plt.close()
    mpl.rcParams.update(mpl.rcParamsDefault)
    return data

def plot_property(xvalues, yvalues, prop_name = None, xaxis_label = None, yaxis_label = None, show_plot = True, scale = 1.0):
    """yvalues versus xvalues 
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import math
    from matplotlib import rc,rcParams
    
    width=15.0*scale
    fs=int(width*1.8)
    ticksize = fs
    golden_ratio = (math.sqrt(5) - 0.9) / 2.0
    labelsize = fs
    height = float(width * golden_ratio)
    
    rc('axes', linewidth=3*scale)
    
    fig = plt.figure(figsize=(width,height))
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    ax = plt.gca()
    ax.tick_params(width=3*scale,size=10*scale,pad=10*scale)
                   
    rcParams["savefig.format"] = 'png'
    
    plt.plot(xvalues,yvalues, marker = '.',color = 'b', markersize= 15*scale, markeredgewidth = 2.0*scale, linewidth = 2.2*scale, label = prop_name)
    if xaxis_label is not None:
        plt.xlabel(xaxis_label,fontsize=fs)
    if yaxis_label is not None:
        plt.ylabel(yaxis_label,fontsize=fs)

    if prop_name is not None:
        plt.legend()
        leg=ax.legend(loc='best',borderaxespad=scale,borderpad=scale,labelspacing=1*scale,handlelength=2*scale, handletextpad=scale, fontsize = fs)
        leg.get_frame().set_linewidth(3*scale)
        
    if show_plot:
        plt.show()
    else:
        plt.savefig(prop_name+"_plot.png")


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
    import matplotlib.pyplot as plt
    
    hist = np.array(cdos_object._stored_cdos[index]['histogram'])
    cdos = np.array(cdos_object._stored_cdos[index]['cdos'])
    
    ener_arr = np.array(cdos_object._energy_bins)
    cdos_arr = _wls_normalize_histogram_for_plotting(cdos, shift_y_first_nonzero=True)
    hist_arr = _wls_normalize_histogram_for_plotting(hist)
    ones_arr = np.ones(len(ener_arr))
    
    figure, ax = plt.subplots(figsize=(10, 8))
    
    ax.bar(ener_arr, ones_arr, width=cdos_object._energy_bin_width*0.80, color="silver")
    ax.bar(ener_arr, hist_arr, width=cdos_object._energy_bin_width*0.80)
    ax.bar(ener_arr, cdos_arr, width=cdos_object._energy_bin_width*0.40)

    plt.show()
