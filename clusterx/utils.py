import numpy as np
import os

def isclose(r1,r2,rtol=1e-4):
    """Determine if to vectors are similar

    **Parameters:**

    r1,r2: 1D arrays of integer or float
        Vectors to be compared
    rtol: float
        Tolerance

    **Returns:**

    Boolean: if the euclidean distance between r1 and r2 is smaller than rtol,
    it returns ``True``, otherwise returns ``False``.
    """
    return np.linalg.norm(np.subtract(r1,r2)) < rtol


def dict_compare(d1, d2):
    """Compare two dictionaries containing mutable objects.

    This compares two dictionaries. Two dictionaries are considered equal
    even if the position of keys differ. Handles mutable values in the dict.
    Some parts are taken from:

    https://stackoverflow.com/questions/4527942/comparing-two-dictionaries-in-python

    Parameters:

    d1,d2: python dictionaries
        dictionaries to be compared

    Return: boolean
        True if dicts are equal, False if they are different.
    """
    areeq = True

    if len(d1) != len(d2):
        return False

    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    if len(d1) != len(intersect_keys):
        return False

    for k in d1_keys:
        for v1,v2 in zip(d1[k],d2[k]):
            if v1 != v2:
                return False

    return areeq


def sub_folders(root):
    """Return list of subfolders

    Parameters:

    root: string
        path of the absolute folder for which you want to get the list of
        subfolders.
    """
    #return os.walk(root).next()[1] #python 2.7
    return next(os.walk(root))[1]

def is_integer(s):
    """Determine whether argument is integer or can be converted to integer

    Parameters:

    s: object
        Can be any object for which we want to determine whether is integer or
        can be converted to integer.
    """
    try:
        int(s)
        return True
    except ValueError:
        return False

def list_integer_named_folders(root=".",prepend='',containing_files=[],not_containing_files=[]):
    """Return array of integer named folders.

    Scans folders in ``root`` and detects those which are named
    with an integer number. It returns then a sorted list of the
    found integers, e.g. , if the folder structure is::

        root/
            1/
            2/
            5/
            7/
            8/
            run_2/
            run_3/
            run_4/
            run_5/
            run_old/
            folderx/
            foldery/


    then the following array is returned::

        [1,2,5,7,8].

    If ``prepend`` is set to some string, then it returns a (sorted)
    list of strings. For example, if prepend is set to ``"run_"``, then this
    will return the array::

        ["run_2","run_3","run_4","run_5"]

    Parameters:

    root: string
        path of the root folder in which to scan for integer named folders.
    prepend: string
        scan for folders whose name starts with the string ``prepend`` and
        ends with an integer number.
    containing_files: array of strings
        a list of file names that should be contained in the returned folders.
    not_containing_files: array of strings
        a list of file names that should not be contained in the returned folders.

    Returns:
        array of integers if default value for ``prepend`` is used. Otherwise
        returns an array of strings

    """
    folders = sub_folders(root)

    flist = []
    for folder in folders:
        include = True
        for fn in not_containing_files:
            if os.path.exists(os.path.join(root,folder,fn)):
                include = False
                break
        if not include:
            continue

        for fn in containing_files:
            if not os.path.exists(os.path.join(root,folder,fn)):
                include = False
                break
        if not include:
            continue

        if prepend != '':
            if folder.startswith(prepend):
                d = folder[len(prepend):]
                if is_integer(d):
                    flist.append(int(d))
        else:
            if is_integer(folder):
                flist.append(int(folder))

    flist.sort()

    if prepend != '':
        slist = []
        for f in flist:
            slist.append(prepend + str(f))

        return slist

    else:
        return flist



def atat_to_cell(file_path="lat.in", interpret_as="parent_lattice", parent_lattice=None,pbc=None):
    """Parse a ``lat.in`` or ``str.out`` file from ``ATAT``.

    ``ATAT`` users may use the input files from ``ATAT`` to perform a cluster
    expansion with **CELL**. This function allows to convert an input ``lat.in``
    file from ``ATAT`` to a ``ParentLattice`` object in CELL. One may also read
    ``str.out`` files, which are converted to ``Structure`` objects in ``CELL``.

    **Parameters:**

    file_path: string
        string containing the path of the file to be parsed. The file must have
        format corresponding to a ``lat.in`` or ``str.out`` file from ATAT.
    interpret_as: string or None
        Indicate how to interpret the file in ``file_path``. Possible values are:

        * ``None``:
            Three arrays are returned: ``cell``, ``r``, and ``species``.
        * ``parent_lattice``:
            The recommended value if ``file_path`` corresponds to
            a ``lat.in`` input file from ATAT. A ``ParentLattice`` **CELL**
            object will be returned.
        * ``super_cell``:
            The recommended value if ``file_path`` corresponds to
            a ``lat.in`` file with a matrix of lattice vectors (``u,v,w`` in
            `ATAT doc <https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/manual/node35.html>`_)
            different to the identity, or if it is known that the structure is
            a periodic repetition of a parent lattice. For this option, the parent
            lattice must be provided (see ``parent_lattice`` parameter below).
            A ``SuperCell`` object will be returned.
        * ``structure``:
            The recommended value if ``file_path`` corresponds to
            a ``str.out`` file from ATAT. For this option, the parent
            lattice must be provided (see ``parent_lattice`` parameter below).
            In this case a ``Structure`` object will be returned.

    parent_lattice: ParentLattice object
        If ``interpret_as`` is ``super_cell`` or ``structure``, a parent lattice
        must be provided. This must be compatible with the information in the
        ``lat.in`` file that was used to create the ``str.out`` files. See the
        examples below.

    pbc: one or three bool (same as ASE's Atoms object)
        Periodic boundary conditions flags. Examples: True, False, 0, 1,
        (1, 1, 0), (True, False, False). Default value: False. The returned
        **CELL** objects will have these pbc's set up.

    **Returns:**

    Depending on the value of ``interpret_as``, the returned object can be python
    arrays (``interpret_as=None``), a ``ParentLattice``
    (``interpret_as="parent_lattice"``), a ``SuperCell``
    (``interpret_as="super_cell"``), or a ``Structure``
    (``interpret_as="Structure"``).

    **Examples:**

    """
    from ase.data import atomic_numbers
    from copy import deepcopy
    from ase import Atoms
    f = open(file_path)
    lines = f.readlines()
    f.close()

    lat = []

    lat.append([])
    lnn=0
    for ln,line in enumerate(lines):
        ls = line.split()
        if len(ls)>0:
            if ln < 3:
                lat[0].append([float(ls[0]),float(ls[1]),float(ls[2])])
            if ln == 2: lat.append([])
            if ln > 2 and ln < 6:
                lat[1].append([float(ls[0]),float(ls[1]),float(ls[2])])
            if ln == 5: lat.append([])
            if ln >= 6:
                lat[2].append([])
                s=''.join(ls[3:]).split(',')
                lat[2][lnn].append([float(ls[0]),float(ls[1]),float(ls[2])])
                lat[2][lnn].append(s)
                lnn = lnn + 1

    struc = deepcopy(lat)

    # Write coordinate system
    parent=[]
    for i in range(3):
            parent.append([struc[0][i][0],struc[0][i][1],struc[0][i][2]])

    # Write lattice
    scell = []
    for i in range(3):
        scell.append([struc[1][i][0],struc[1][i][1],struc[1][i][2]])

    # Write atoms
    x = []
    species = []
    for i in range(len(struc[2])):
        x.append([struc[2][i][0][0],struc[2][i][0][1],struc[2][i][0][2]])
        nrs = []
        for sp in struc[2][i][1]:
            if sp == "Vac":
                sp = "X"
            nrs.append(atomic_numbers[sp])
        species.append(nrs)

    x = np.matrix(x)
    b = np.matrix(parent)
    a = np.matrix(scell)

    ####################
    # Calculate real cartesian coordinates
    r = x*b
    cell = a*b
    if interpret_as == None:
        return cell, r, species
    if interpret_as == "parent_lattice":
        from clusterx.parent_lattice import ParentLattice
        nrs = np.zeros(len(species))
        for inr, nr in enumerate(species):
            nrs[inr]=int(nr[0])

        plat = ParentLattice(Atoms(positions=r,cell=cell,numbers=nrs,pbc=pbc),sites=species,pbc=pbc)
        return plat
    if interpret_as == "super_cell":
        from clusterx.structure import Structure
        pcell = parent_lattice.get_cell()
        tmat = np.asarray(np.rint(np.dot(cell,np.linalg.inv(pcell))).astype(int))
        from clusterx.super_cell import SuperCell
        scell = SuperCell(parent_lattice,tmat)
        return scell
    if interpret_as == "structure":
        from clusterx.structure import Structure
        pcell = parent_lattice.get_cell()
        tmat = np.asarray(np.rint(np.dot(cell,np.linalg.inv(pcell))).astype(int))
        from clusterx.super_cell import SuperCell
        scell = SuperCell(parent_lattice,tmat)
        pris = scell.get_atoms()

        pos = pris.get_positions()
        new_nrs = np.zeros(len(pos))
        for ip,p in enumerate(r):
            nr = species[ip][0]
            for i2,p2 in enumerate(pos):
                if isclose(p,p2):
                    new_nrs[i2] = nr

        struc = Structure(scell,decoration=new_nrs)
        return struc
