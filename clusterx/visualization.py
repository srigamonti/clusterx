# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np

def juview(plat,n=None):
    """Visualize structure object in Jupyter notebook

    **Parameters:**

    ``plat``: any of Atoms(ASE), ParentLattice, Supercell, Structure, StructureSet
        structure object to be plotted
    ``n``: integer
        plot the first ``n`` structures. If ``None``, return all structures.

    **Return:**

        nglview display object
    """
    import nglview
    from clusterx.parent_lattice import ParentLattice
    from clusterx.structures_set import StructuresSet
    from clusterx.structures_set import Structure
    from clusterx.clusters.clusters_pool import ClustersPool
    from clusterx.clusters.cluster import Cluster

    if isinstance(plat,Structure):
        return _makeview( [ plat.get_atoms() ] )

    if isinstance(plat,ParentLattice):
        return _makeview(plat.get_all_atoms())

    if isinstance(plat,StructuresSet):
        return _makeview(plat.get_images(n=n))

    if isinstance(plat,ClustersPool):
        atoms = plat.get_cpool_atoms()
        if n is not None:
            return _makeview(atoms[0:n])
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
    hboxes = [ipywidgets.HBox(views[i*3:i*3+3]) for i in range(int(math.ceil(len(views)/3.0)))]
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


def plot_optimization_vs_number_of_clusters(clsel,nclmin=0,scale=1.0):
    """Plot cluster optimization with matplotlib

    The plot shows the prediction and fitting errors as a function of the clusters
    pool size.

    **Parameters:**

    ``clsel``: ClustersSelector object
        The ClustersSelector oject which was used for the optimization to be plotted.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from matplotlib import rc,rcParams
    import math


    set_sizes=sorted(clsel.set_sizes)
    indizes=[i[0] for i in sorted(enumerate(clsel.set_sizes), key=lambda x:x[1])]

    rmse=[clsel.rmse[ind] for ind in indizes]
    cvs=[clsel.cvs[ind] for ind in indizes]

    nclmax=max(set_sizes)
    nclmin=min(set_sizes)
    ncl_range=nclmax-nclmin

    e_min=min([min(rmse),min(cvs)])
    e_max=max([max(rmse),max(cvs)])
    e_range=e_max - e_min

    ncl_opt=set_sizes[clsel.cvs.index(min(cvs))]

    width=15.0*scale
    fs=int(width*1.8)
    ticksize = fs
    golden_ratio = (math.sqrt(5) - 0.9) / 2.0
    labelsize = fs
    height = float(width * golden_ratio)

    plt.figure(figsize=(width,height))

    rc('axes', linewidth=3*scale)

    plt.ylim(e_min-e_range/8,e_max+e_range/10)
    plt.xlim(nclmin-ncl_range/10,nclmax+ncl_range/10)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    ax = plt.gca()
    ax.tick_params(width=3*scale,size=10*scale,pad=10*scale)

    #ax.tick_params(axis="x",which="minor",width=3,size=10,pad=10)


    #print((clsel.clusters_sets=="combinations") or (clsel.clusters_sets=="size+combinations"))

    plt.plot([ncl_opt],[min(cvs)], 'o', markersize=25*scale, markeredgewidth=4*scale,markeredgecolor='r', markerfacecolor='None' , label='lowest cv-RMSE' )


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

        plt.plot(set_sizes, rmse, markersize=25*scale, marker='.', color='blue', zorder=1,  linestyle='',label='training-RMSE')
        plt.plot(set_sizes, cvs, markersize=25*scale, marker='.', color='black', zorder=1, linestyle='',label='cv-RMSE')
        plt.plot(opt_s, opt_r, markersize=25*scale, marker='.', color='blue', zorder=1,  linestyle='-', linewidth=4*scale)
        plt.plot(opt_s, opt_cv, markersize=25*scale, marker='.',  color='black', zorder=1, linestyle='-', linewidth=4*scale)

    else:

        #plt.plot([ncl_opt],[min(cvs)], 'o', markersize=25, markeredgewidth=4,markeredgecolor='r', markerfacecolor='None' , label='lowest cv-RMSE' )
        #scatter([ncl_opt],[min(cv)], s=400,facecolors='none', edgecolors='r',)

        plt.plot(set_sizes, rmse, markersize=25*scale, marker='.', color='blue', zorder=1,  linestyle='-',label='training-RMSE', linewidth=4*scale)
        plt.plot(set_sizes, cvs, markersize=25*scale, marker='.', color='black', zorder=1, linestyle='-',label='cv-RMSE',linewidth=4*scale)

    plt.ylabel("Energy [arb. units]",fontsize=fs)
    plt.xlabel('Number of clusters',fontsize=fs)
    plt.legend()
    leg=ax.legend(loc='best',borderaxespad=2*scale,borderpad=2*scale,labelspacing=1*scale,handlelength=3*scale, handletextpad=2*scale)
    leg.get_frame().set_linewidth(3*scale)

    for l in leg.get_texts():
        l.set_fontsize(fs)

    #plt.savefig("plot_optimization.png")
    plt.show()


def plot_optimization_vs_sparsity(clsel,scale=1.0):
    """Plot cluster optimization with matplotlib

    The plot shows the prediction and fitting errors as a function of the

    **Parameters:**

    ``clsel``: ClustersSelector object
        The ClustersSelector oject which was used for the optimization to be plotted.
        sparsity parameter when the LASSO method is used.
    """

    import matplotlib.pyplot as plt
    from matplotlib import rc
    from matplotlib import rc,rcParams
    import math

    set_sparsity=clsel.lasso_sparsities

    rmse=clsel.rmse
    cvs=clsel.cvs

    nclmax=max(set_sparsity)
    nclmin=min(set_sparsity)
    ncl_range=nclmax-nclmin

    e_min=min([min(rmse),min(cvs)])
    e_max=max([max(rmse),max(cvs)])
    e_range=e_max - e_min

    opt=set_sparsity[clsel.cvs.index(min(cvs))]

    width=15.0*scale
    fs=int(width*1.8)
    ticksize = fs
    golden_ratio = (math.sqrt(5) - 0.9) / 2.0
    labelsize = fs
    height = float(width * golden_ratio)

    plt.figure(figsize=(width,height))

    rc('axes', linewidth=3*scale)

    plt.ylim(e_min-e_range/8,e_max+e_range/10)
    plt.xlim(nclmin-0.1*nclmin,nclmax+0.1*nclmax)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    ax = plt.gca()
    ax.tick_params(axis="both",which="both",width=3*scale,size=10*scale,pad=10*scale)


    plt.semilogx([opt],[min(cvs)], 'o', markersize=25*scale, markeredgewidth=4*scale,markeredgecolor='r', markerfacecolor='None' , label='lowest cv-RMSE')
    #scatter([ncl_opt],[min(cv)], s=400,facecolors='none', edgecolors='r',)

    plt.semilogx(set_sparsity, rmse, markersize=25*scale, marker='.', color='blue', zorder=1,  linestyle='-',label='training-RMSE', linewidth=4*scale)
    plt.semilogx(set_sparsity, cvs, markersize=25*scale, marker='.', color='black', zorder=1, linestyle='-',label='cv-RMSE',linewidth=4*scale)

    plt.ylabel("Energy [arb. units]",fontsize=fs)
    plt.xlabel('Sparsity',fontsize=fs)
    plt.legend()
    leg=ax.legend(loc='best',borderaxespad=2*scale,borderpad=2*scale,labelspacing=1*scale,handlelength=3*scale, handletextpad=2*scale)
    leg.get_frame().set_linewidth(3*scale)

    for l in leg.get_texts():
        l.set_fontsize(fs)

    #plt.savefig("plot_optimization.png")
    plt.show()


def plot_predictions_vs_target(sset,cemodel, prop_name, scale=1.0):
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
    labelsize = fs
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

    plt.ylabel("Predicted energy [arb. units]",fontsize=fs)
    plt.xlabel('Calculated energy [arb. units]',fontsize=fs)
    plt.legend()
    leg=ax.legend(loc='best',borderaxespad=2*scale,borderpad=2*scale,labelspacing=1*scale,handlelength=3*scale, handletextpad=2*scale)
    leg.get_frame().set_linewidth(3*scale)

    for l in leg.get_texts():
        l.set_fontsize(fs)

    #plt.savefig("plot_optimization.png")
    plt.show()

def plot_property_vs_concentration(sset, site_type=0, sigma=1, cemodel=None, property_name = None, show_loo_predictions = True, sset_enum=None, sset_gss=None, show_plot = True, refs=None, scale=1.0):
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
    import math
    from matplotlib import rc,rcParams

    data = {}

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

    if refs is None:
        refs = [0.0,0.0]

    energies = sset.get_property_values(property_name = property_name)
    if cemodel is not None:
        predictions = sset.get_predictions(cemodel)

    if sset_enum is not None:
        pred_enum = sset_enum.get_predictions(cemodel)

    if sset_gss is not None:
        pred_gss = sset_gss.get_predictions(cemodel)

    if show_loo_predictions and cemodel is not None:
        cvs = cemodel.get_cv_score(sset)
        pred_cv = cvs["Predictions-CV"]

    frconc = sset.get_concentrations(site_type,sigma)
    vl_en = refs[0]*(1-np.array(frconc)) + np.array(frconc)*refs[1]

    if sset_enum is not None:
        frconc_enum = sset_enum.get_concentrations(site_type,sigma)
        vl_en_enum = refs[0]*(1-np.array(frconc_enum)) + np.array(frconc_enum)*refs[1]
    if sset_gss is not None:
        frconc_gss = sset_gss.get_concentrations(site_type,sigma)
        vl_en_gss = refs[0]*(1-np.array(frconc_gss)) + np.array(frconc_gss)*refs[1]


    #fig = plt.figure()
    #fig.suptitle("Property vs. concentration")
    data["concentration"] = frconc
    data["property"] = energies-vl_en
    plt.scatter(frconc,energies-vl_en,marker='o', s=150*scale, edgecolors='green', facecolors='none',label='Calculated')
    if cemodel is not None:
        data["predicted-property"] = predictions-vl_en
        data["predicted-property-cv"] = pred_cv-vl_en
        plt.scatter(frconc,pred_cv-vl_en,marker='x', s=75*scale, edgecolors='none', facecolors='red',label='Predicted-CV')
        plt.scatter(frconc,predictions-vl_en,marker='o', s=50*scale, edgecolors='none', facecolors='blue',label='Predicted')
    if sset_enum is not None:
        data["concentration-enum"] = frconc_enum
        data["predicted-property-enumeration"] = pred_enum-vl_en_enum
        plt.scatter(frconc_enum,pred_enum-vl_en_enum,marker='o', s=30*scale, edgecolors='none', facecolors='gray',label='Enumeration')
    if sset_gss is not None:
        data["concentration-gss"] = frconc_gss
        data["predicted-property-gss"] = pred_gss-vl_en_gss
        plt.scatter(frconc_gss,pred_gss-vl_en_gss,marker='o', s=40*scale, edgecolors='none', facecolors='red',label='Predicted GS')

    plt.xlabel("Concentration",fontsize=fs)
    plt.ylabel(property_name,fontsize=fs)

    plt.legend()
    leg=ax.legend(loc='best',borderaxespad=2*scale,borderpad=2*scale,labelspacing=1*scale,handlelength=3*scale, handletextpad=2*scale)
    leg.get_frame().set_linewidth(3*scale)

    for l in leg.get_texts():
        l.set_fontsize(fs)

    if show_plot:
        plt.show()

    plt.close()
    mpl.rcParams.update(mpl.rcParamsDefault)
    return data
