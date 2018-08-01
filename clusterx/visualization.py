


def juview(plat):
    """Visualize structure object in Jupyter notebook

    **Parameters:**

    ``plat``: any of Atoms(ASE), ParentLattice, Supercell, Structure, or an array of them
        structure object to be plotted

    **Return:**

        nglview display object
    """
    import nglview
    from clusterx.parent_lattice import ParentLattice

    if isinstance(plat,ParentLattice):
        return _makeview(plat.get_all_atoms())

    if not isinstance(plat,list):
        view = nglview.show_ase(plat)
        view.add_unitcell()
        view.add_ball_and_stick()
        view.parameters=dict(clipDist=0,color_scheme="element")
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
        view.parameters = dict(backgroundColor='white',clipDist=-100,color_scheme="element")
        view.add_unitcell()
        view.camera = 'orthographic'
        #view.center()
        view.add_ball_and_stick()
        view.add_spacefill(radius_type='vdw',scale=0.3)
        views.append(view)

    import ipywidgets
    hboxes = [ipywidgets.HBox(views[i*3:i*3+3]) for i in range(int(math.ceil(len(views)/3.0)))]
    vbox = ipywidgets.VBox(hboxes)
    return vbox
