import numpy as np

def juview(plat):
    """Visualize structure object in Jupyter notebook

    **Parameters:**

    ``plat``: any of Atoms(ASE), ParentLattice, Supercell, Structure, StructureSet
        structure object to be plotted

    **Return:**

        nglview display object
    """
    import nglview
    from clusterx.parent_lattice import ParentLattice
    from clusterx.structures_set import StructuresSet

    if isinstance(plat,ParentLattice):
        return _makeview(plat.get_all_atoms())

    if isinstance(plat,StructuresSet):
        return _makeview(plat.get_images())

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


def plot_optimization_vs_number_of_clusters(clsel):
    """Plot cluster optimization with matplotlib

    The plot shows the prediction and fitting errors as a function of the clusters
    pool size.
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

    width=15.0
    fs=int(width*1.8)
    ticksize = fs
    golden_ratio = (math.sqrt(5) - 1.0) / 2.0
    labelsize = fs
    height = float(width * golden_ratio)

    plt.figure(figsize=(width,height))

    rc('axes', linewidth=3)

    plt.ylim(e_min-e_range/8,e_max+e_range/10)
    plt.xlim(nclmin-ncl_range/10,nclmax+ncl_range/10)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    ax = plt.gca()
    ax.tick_params(width=3,size=10,pad=10)

    plt.plot([ncl_opt],[min(cvs)], 'o', markersize=25, markeredgewidth=4,markeredgecolor='r', markerfacecolor='None' )
    #scatter([ncl_opt],[min(cv)], s=400,facecolors='none', edgecolors='r',)

    plt.plot(set_sizes, rmse, markersize=25, marker='.', color='blue', zorder=1,  linestyle='-',label='training-RMSE', linewidth=4)
    plt.plot(set_sizes, cvs, markersize=25, marker='.', color='black', zorder=1, linestyle='-',label='cv-RMSE',linewidth=4)

    plt.ylabel("Energy [arb. units]",fontsize=fs)
    plt.xlabel('Number of clusters',fontsize=fs)
    plt.legend()
    leg=ax.legend(loc='upper left',borderaxespad=2,borderpad=2,labelspacing=1,handlelength=3, handletextpad=2)
    leg.get_frame().set_linewidth(3)

    for l in leg.get_texts():
        l.set_fontsize(25)

    #plt.savefig("plot_optimization.png")
    plt.show()


def plot_optimization_vs_sparsity(clsel):
    """Plot cluster optimization with matplotlib

    The plot shows the prediction and fitting errors as a function of the
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

    width=15.0
    fs=int(width*1.8)
    ticksize = fs
    golden_ratio = (math.sqrt(5) - 1.0) / 2.0
    labelsize = fs
    height = float(width * golden_ratio)

    plt.figure(figsize=(width,height))

    rc('axes', linewidth=3)

    plt.ylim(e_min-e_range/8,e_max+e_range/10)
    plt.xlim(nclmin-ncl_range/10,nclmax+ncl_range/10)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    ax = plt.gca()
    ax.tick_params(width=3,size=10,pad=10)

    plt.semilogx([opt],[min(cvs)], 'o', markersize=25, markeredgewidth=4,markeredgecolor='r', markerfacecolor='None' )
    #scatter([ncl_opt],[min(cv)], s=400,facecolors='none', edgecolors='r',)

    plt.semilogx(set_sparsity, rmse, markersize=25, marker='.', color='blue', zorder=1,  linestyle='-',label='training-RMSE', linewidth=4)
    plt.semilogx(set_sparsity, cvs, markersize=25, marker='.', color='black', zorder=1, linestyle='-',label='cv-RMSE',linewidth=4)

    plt.ylabel("Energy [arb. units]",fontsize=fs)
    plt.xlabel('Number of clusters',fontsize=fs)
    plt.legend()
    leg=ax.legend(loc='upper left',borderaxespad=2,borderpad=2,labelspacing=1,handlelength=3, handletextpad=2)
    leg.get_frame().set_linewidth(3)

    for l in leg.get_texts():
        l.set_fontsize(25)

    #plt.savefig("plot_optimization.png")
    plt.show()
