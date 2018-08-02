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
