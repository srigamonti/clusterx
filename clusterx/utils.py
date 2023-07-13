# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
import os
import copy
from ase.data import chemical_symbols as cs
from ase.build.supercells import clean_matrix, lattice_points_in_supercell # needed by make_supercell
from ase import Atoms

class SupercellError(Exception):
    pass

def findmax(*args):
    vals = []
    for v in args:
        if hasattr(v,"__len__"):
            vals.append(np.amax(v))
        else:
            vals.append(v)

    return np.amax(vals)

def findmin(*args):
    vals = []
    for v in args:
        if hasattr(v,"__len__"):
            vals.append(np.amin(v))
        else:
            vals.append(v)

    return np.amin(vals)
        
def isclose(r1,r2,rtol=1e-4):
    """Determine whether two vectors are similar

    **Parameters:**

    ``r1,r2``: 1D arrays of integer or float
        Vectors to be compared
    ``rtol``: float
        Tolerance.

    **Returns:**

    Boolean: if the euclidean distance between ``r1`` and ``r2`` is smaller than
    ``rtol``, ``True`` is returned, otherwise ``False`` is returned.
    """
    try:
        if len(r1) != len(r2):
            return False
        return np.linalg.norm(np.subtract(r1,r2)) < rtol
    except:
        return np.linalg.norm(np.subtract(r1,r2)) < rtol


def dict_compare(d1, d2, tol=None):
    """Compare two dictionaries containing mutable objects.

    This compares two dictionaries. Two dictionaries are considered equal
    even if the position of keys differ. Handles mutable values in the dict.
    Some parts are taken from:

    https://stackoverflow.com/questions/4527942/comparing-two-dictionaries-in-python

    **Parameters:**

    ``d1,d2``: python dictionaries
        dictionaries to be compared

    ``tol``: float
        a small float number. If not ``None``, the comparison of dictionary
        values is regarded as a vector comparison and done with
        ``utils.isclose()``. For the meaning of ``tol``, read the documentation
        for ``rtol`` parameter of ``utils.isclose()``.

    Return: boolean
        ``True`` if dicts are equal, ``False`` if they are different.
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
        try:
            for v1,v2 in zip(d1[k],d2[k]):
                if (isinstance(v1, list)) or (isinstance(v1,np.ndarray)):
                    for i,v in enumerate(v1):
                        if tol is None:
                            if v != v2[i]:
                                return False
                        else:
                            if not isclose(v ,v2[i], tol):
                                return False
                else:
                    if tol is None:
                        if v1 != v2:
                            return False
                    else:
                        if not isclose(v1,v2,tol):
                            return False

        except TypeError:
            try:
                if _is_integrable(d1[k]):
                    if not isclose(d1[k],d2[k],tol):
                        return False
            except:
                if type(d1[k]) is dict:
                    for subkeys in d1[k].keys():
                        if not isclose(d1[k][subkeys],d2[k][subkeys],tol):
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

def _is_integrable(s):
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

def list_integer_named_folders(root=".", prefix='', suffix='', containing_files=[], not_containing_files=[]):
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

    If ``prefix`` is set to some string, then it returns a (sorted)
    list of strings. For example, if prefix is set to ``"run_"``, then this
    will return the array::

        ["run_2","run_3","run_4","run_5"]

    Parameters:

    ``root``: string
        path of the root folder in which to scan for integer named folders.
    ``prefix``: string
        scan for folders whose name starts with the string ``prefix`` followed
        by and integer number (and possibly the string ``suffix``).
    ``suffix``: string
        scan for folders whose name ends with the string ``suffix`` and
        is preceded by an integer number (and possibly the string ``prefix``).
    ``containing_files``: array of strings
        a list of file names that should be contained in the returned folders.
    ``not_containing_files``: array of strings
        a list of file names that should not be contained in the returned folders.

    Returns:
        array of integers if default value for ``prefix`` is used. Otherwise
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

        if prefix == folder[:len(prefix)] and suffix == folder[-len(suffix):]:
            if len(prefix) != 0 and len(suffix) != 0:
                d = folder[len(prefix):][:-len(suffix)]
            elif len(prefix) != 0 and len(suffix) == 0:
                d = folder[len(prefix):]
            elif len(prefix) == 0 and len(suffix) != 0:
                d = folder[:-len(suffix)]
            else:
                d = folder[:]

            if _is_integrable(d):
                flist.append(int(d))
        else:
            if _is_integrable(folder):
                flist.append(int(folder))

    flist.sort()

    if prefix != '' or suffix != '':
        slist = []
        for f in flist:
            slist.append(prefix + str(f) + suffix)

        return slist

    else:
        return flist



def atat_to_cell(file_path="lat.in", interpret_as="parent_lattice", parent_lattice=None,pbc=None,wrap=True):
    """Parse a ``lat.in`` or ``str.out`` file from ``ATAT``.

    ``ATAT`` users may use the input files from ``ATAT`` to perform a cluster
    expansion with **CELL**. This function allows to convert an input ``lat.in``
    file from ``ATAT`` to a ``ParentLattice`` object in CELL. One may also read
    ``str.out`` files, which are converted to ``Structure`` objects in ``CELL``.

    **Parameters:**

    ``file_path``: string
        string containing the path of the file to be parsed. The file must have
        format corresponding to a ``lat.in`` or ``str.out`` file from ATAT.
    ``interpret_as``: string or None
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

    ``parent_lattice``: ParentLattice object
        If ``interpret_as`` is ``super_cell`` or ``structure``, a parent lattice
        must be provided. This must be compatible with the information in the
        ``lat.in`` file that was used to create the ``str.out`` files. See the
        examples below.
    ``pbc``: one or three bool (same as ASE's Atoms object)
        Periodic boundary conditions flags. Examples: True, False, 0, 1,
        (1, 1, 0), (True, False, False). Default value: False. The returned
        **CELL** objects will have these pbc's set up.
    ``wrap``: boolean (default:``True``)
        Wrap atomic coordinates. If pbc is ``None``, pbc is set to (1,1,1).
        Set ``wrap`` to ``False`` if structure corresponds
        to a supercell, i.e., if the second matrix of the structure definition
        in either the lat.in or str.out file is different from the identity
        matrix.

    **Returns:**

    Depending on the value of ``interpret_as``, the returned object can be python
    arrays (``interpret_as=None``), a ``ParentLattice``
    (``interpret_as="parent_lattice"``), a ``SuperCell``
    (``interpret_as="super_cell"``), or a ``Structure``
    (``interpret_as="Structure"``).

    **Examples:**

    .. todo::
        Clarify wrap option. Right now it is not guaranteed to
        work when the supercell definition in lat.in is not the
        identity matrix. This is so, because in the lat.in format
        the scaled coordinates are given in reference to the
        cartesian vectors of the parent lattice, but may define a
        supercell. Fix it such that the x matrix below referrs to
        a*b and not b as it is now.

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
            if sp == "Vac" or sp == "V":
                sp = "X"
            nrs.append(atomic_numbers[sp])
        species.append(nrs)

    x = np.matrix(x)
    b = np.matrix(parent)
    a = np.matrix(scell)

    ####################
    # Calculate real cartesian coordinates
    if wrap:
        from clusterx.symmetry import wrap_scaled_positions
        if pbc == None:
            pbc = (1,1,1)
        x = wrap_scaled_positions(x,pbc)
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
        pris = scell.get_pristine()

        pos = pris.get_positions()
        new_nrs = np.zeros(len(pos))
        for ip,p in enumerate(r):
            nr = species[ip][0]
            for i2,p2 in enumerate(pos):
                if isclose(np.asarray(p).reshape(-1),p2):
                    new_nrs[i2] = nr

        struc = Structure(scell,decoration=new_nrs)
        return struc


def _is_integer_matrix(m,rnd=5):
    mi = np.around(m,rnd)
    for k in range(3):
        for l in range(3):
            #if not round(m[k,l],rnd).is_integer():
            if not mi[k,l].is_integer():
                return False
    return True

def get_cl_idx_sc(cl, sc, method=0, tol=1e-3):
    """Return atom indexes of cluster points in SuperCell

    **Parameters:**

    ``cl``: npoints x 3 matrix
        matrix of cartesian or scaled coordinates of cluster points. Cluster
        positions are expected to be wrapped inside supercell ``sc``
    ``sc``: natoms x 3 matrix
        matrix of cartesian or scaled coordinates of supercell atomic positions.
        The type of coordinates (either cartesion or scaled) must coincide with
        that of ``cl``
    ``method``: integer
        Method to use. 0: (slow) nested for loop using numpy allclose. 1: (fast)
        calculates all distances from points in ``cl`` to atoms in ``sc``, and
        return indices for which distances are zero.
    ``tol``: real positive number
        tolerance to determine whether cluster and atom positions are the same.
    """
    from scipy.spatial.distance import cdist
    sdistances = None
    method = 1
    if method == 0:
        idxs = np.zeros(len(cl),dtype="int")
        for icl,clp in enumerate(cl):
            for isc, scp in enumerate(sc):
                if np.allclose(clp,scp,atol=tol):
                    idxs[icl] = isc
                    break

    if method == 1:
        sdistances = cdist(cl, sc, metric='euclidean') # Evaluate all (scaled) distances between cluster points to scell sites
        idxs = np.argwhere(np.abs(sdistances) < tol)[:,1] # Atom indexes of the transformed cluster

    return idxs

def add_noise(v,noise_level):
    """Add randomly distributed noise to vector coordinates

    To each coordinate of the input vector ``v``, random noise uniformly
    distributed between -``noise_level`` and ``noise_level`` is added. The input
    vector ``v`` is left unchanged. The modified vector is returned.

    **Parameters:**

    ``v``: list of floats
        The input vector

    ``noise_level``: float
        Width of the uniform distribution used to add noise.
    """
    import random
    energies = []
    for e in v:
        energies.append(e+random.uniform(-1,1)*noise_level)
    return energies

def calculate_trafo_matrix(pcell,scell,rnd=5):
    """Calculate integer transformation matrix given a primitive cell and a super-cell

    If :math:`S` and :math:`V` are, respectively, a matrix whose rows are the cartesian coordinates
    of the parent lattice vectors and a matrix whose rows are the cartesian coordinates
    of the super-cell lattice vectors; then, this function returns the matrix :math:`P=SV^{-1}`.
    If the resulting matrix is not integer (see ``rnd`` parameter), then ``None`` is returned.

    **Parameters:**

    ``pcell``: 3x3 array of float
        The rows of this matrix correspond to the cartesian coordinates of a parent lattice

    ``scell``: 3x3 array of float
        The rows of this matrix correspond to the cartesian coordinates of a supercell

    ``rnd``: integer (optional)
         The matrix :math:`P=SV^{-1}` is rounded to ``rnd`` decimal places and checked for integrity.
    """
    tmat = np.dot(scell,np.linalg.inv(pcell))
    if _is_integer_matrix(tmat,rnd):
        return np.asarray(np.rint(tmat).astype(int))
    else:
        return None

def _str_grep(input_str, search_str, prepend=''):
    out_str = ''
    for l in input_str.split('\n'):
        if search_str in l:
            out_str = out_str + prepend + l.strip() + '\n'

    return out_str.rstrip()

def mgrep(fpath, search_array, prepend='',root='.'):
    """
    Grep strings in file and return matching lines.

    **Parameters:**

    ``fpath``: string
        File path to grep
    ``search_array``: array of strings
        Each element of the array is an string to grep in ``fpath``.
    ``prepend``: string
        prepend string ``prepend`` to each matching line.
    ``root``: string
        File to grep should be in ``root/fpath``.
    """
    abs_path = os.path.join(root,fpath)
    out_str = ''
    if os.path.isfile(abs_path):
        fstr = open(abs_path).read()

        for attr in search_array:
            ostr = _str_grep(fstr,attr,prepend=prepend)

            if ostr != '':
                out_str = out_str + prepend + ostr.strip() + '\n'

    return out_str.rstrip()


def parent_lattice_to_atat(plat, out_fname="lat.in", for_str = False):
    """Serializes ParentLattice object to ATAT input file

    **Parameters:**

    ``plat``: ParentLattice object
        ParentLattice object to be serialized
    ``out_fname``: string
        Output file path
    """
    cell = plat.get_cell()
    positions = plat.get_scaled_positions()
    if for_str:
        sites = plat.get_atomic_numbers()
    else:
        sites = plat.get_sites()

    f = open(out_fname,'w+')

    for cellv in cell:
        f.write(u"%2.12f\t%2.12f\t%2.12f\n"%(cellv[0],cellv[1],cellv[2]))

    f.write(u"1.000000000000\t0.000000000000\t0.000000000000\n")
    f.write(u"0.000000000000\t1.000000000000\t0.000000000000\n")
    f.write(u"0.000000000000\t0.000000000000\t1.000000000000\n")

    for i,pos in enumerate(positions):
        stri = u"%2.12f\t%2.12f\t%2.12f\t"%(pos[0],pos[1],pos[2])
        if for_str:
            stri = stri + "%s\n"%cs[sites[i]]
        else:
            if len(sites[i])>1:
                for z in sites[i][:-1]:
                    stri = stri + "%s,"%cs[z]
            stri = stri + "%s\n"%cs[sites[i][-1]]

        f.write(stri)

    f.close()

class Exponential():
    """Basic exponential object of type ``coefficient`` * ``x`` ^ ``exponent`` . Numerically evalueted using the method ``evaluate`` ( ``x`` ) .

    **Parameters:**

    ``exponent``: exponent of the exponential

    ``coefficient``: coefficient of the exponential

    """

    def __init__(self, exponent, coefficient = 1):
        self.exponent = exponent
        self.coefficient = coefficient

    def evaluate(self, x):
        return self.coefficient * np.power(x,self.exponent)

    def divide_scalar(self, value):
        self.coefficient = self.coefficient / value

    def multiply_scalar(self, scalar):
        self.coefficient = self.coefficient * scalar


class PolynomialFunction():
    """Polynomial function, build from ``Exponential()`` .
    """


    def __init__(self):
        self.exponentials = []

    def add_exponential(self, order, coefficient = 1):
        in_sum = False
        for exponential in self.exponentials:
            if exponential.exponent == order:
                exponential.coefficient += coefficient
                in_sum = True
                break
        if not in_sum:
            self.exponentials.append(Exponential(order, coefficient))

    def clear_exponentials(self):
        for exponential in self.exponentials:
            if abs(exponential.coefficient) < 10**(-15):
                self.exponentials.remove(exponential)

    def evaluate(self, x):
        value = 0
        for exponential in self.exponentials:
            value += exponential.evaluate(x)
        return value

    def normalize(self, scalar_product, m):
        length = scalar_product(self.evaluate, self.evaluate, m)
        length = np.sqrt(length)
        for exponential in self.exponentials:
            exponential.divide_scalar(length)

    def multiply_scalar(self, scalar):
        for exponential in self.exponentials:
            exponential.multiply_scalar(scalar)

    def add_polynomial_function(self, polynomial_function):
        for exponential in polynomial_function.exponentials:
            self.add_exponential(exponential.exponent, exponential.coefficient)

    def print_polynomial(self):
        outstring = ''
        for exponential in self.exponentials:
            outstring += str(exponential.coefficient) + ' *  x^' + str(exponential.exponent) + ' + '
        outstring = outstring[:-3]
        print(outstring)


class PolynomialBasis():
    """Polynomial basis, constructed from several ``PolynomialFunction()``.
    Constructs orthonormal basis sets using ``scalcar_product`` .
    When initialized, all orthonormal basis sets to the order ``max_order`` are generated.

    **Parameters**

    ``max_order``: Maximal order to which the basis set is initialized.

    ``symmetric``: Defines if the scalar product uses sigmas symmetrized around 0 or ascending from 0.

    """

    def __init__(self, max_order = 10, symmetric = False):
        self.m = max_order
        self.symmetric = symmetric
        self.basis_function_set = {}
        for order in range(1,max_order+1):
            self.basis_function_set[str(order)] = self.construct(order)

    def construct(self, m):
        basis_functions = []
        for degree in range(m):
            new_function = PolynomialFunction()
            new_function.add_exponential(degree)
            chi = Exponential(degree)
            for basis_function in basis_functions:
                overlap = self.scalar_product(basis_function.evaluate,chi.evaluate, m)
                overlap = overlap * (-1)
                summand = copy.deepcopy(basis_function)
                summand.multiply_scalar(overlap)
                new_function.add_polynomial_function(summand)
            new_function.normalize(self.scalar_product, m)
            new_function.clear_exponentials()
            basis_functions.append(new_function)
        return basis_functions

    def scalar_product(self, function1, function2, m):
        """
        returns the result of the scalar product 1/m sum_{sigma} f_1(sigma) * f_2(sigma)
        """
        scaling = 1 / m
        if self.symmetric:
            sigmas = [x for x in range(-int(m/2), int(m/2)+1)]
            if m % 2 == 0:
                sigmas.remove(0)
        else:
            sigmas = [x for x in range(m)]
        scalar_product = 0
        for sigma in sigmas:
            scalar_product += function1(sigma) * function2(sigma)
        scalar_product = scalar_product * scaling
        return scalar_product

    def print_basis_functions(self, m):
        for function in self.basis_function_set[str(m)]:
            function.print_polynomial()

    def evaluate(self, alpha, sigma, m):
        if m > self.m:
            self.basis_function_set[str(m)] = self.construct(m)
        return self.basis_function_set[str(m)][int(alpha)].evaluate(sigma)


def poppush(x, val):
    """Left-shift array one position and writes new value to the right. Returns average.

    **Parameters:**
    ``x``: numpy array
    ``val``: int or float
    """
    x[:-1] = x[1:]; x[-1] = val
    return x.mean()

def sort_atoms(atoms, key = (2,1,0)):
    """Return atoms object with sorted atomic coordinates

    The default sorting is: increasing z-coordinate first, increasing
    y-coordinate second, increasing x-coordinate third. Useful to get
    well ordered slab structures, for instance. Sorting can be changed
    by appropriately setting the ``key`` argument, with the same
    effect as in::

        from operator import itemgetter
        sp = sorted(p, key=itemgetter(2,1,0))

    where p is a Nx3 array of vector coordinates.
    """
    if key is None:
        return atoms
    else:
        nrs = atoms.get_atomic_numbers()
        poss = atoms.get_positions()
        pn = []
        for p,n in zip(poss,nrs):
            pn.append([p[0],p[1],p[2],n])
        from operator import itemgetter
        import numpy as np
        _pn = sorted(pn, key=itemgetter(*key))
        _poss = np.delete(np.array(_pn),3,1)
        _nrs = np.delete(np.array(_pn),[0,1,2],1).flatten()
        from ase import Atoms
        return Atoms(positions=_poss, numbers=_nrs, cell=atoms.get_cell(), pbc=atoms.get_pbc())

def remove_vacancies(at):
    """Remove every Atom containing 'X' as species symbol or 0 as atomic number
    from an Atoms object and return the resulting Atoms object.

    **Parameters:**
    ``at``: Atoms object

    .. todo::
        find cleaner way to do this...
    """
    from ase import Atoms
    positions = []
    numbers = []
    indices = []
    tags = []
    tags0 = at.get_tags()
    momenta = []
    momenta0 = at.get_momenta()
    masses = []
    masses0 = at.get_masses()

    for i,atom in enumerate(at):
        nr = atom.get('number')
        if nr != 0:
            indices.append(i)
            positions.append(atom.get('position'))
            tags.append(tags0[i])
            momenta.append(momenta0[i])
            masses.append(masses0[i])
            numbers.append(nr)

    return Atoms(cell=at.get_cell(),
                 pbc=at.get_pbc(),
                 numbers=numbers,
                 positions=positions,
                 calculator=at.get_calculator(),
                 tags=tags,
                 momenta=momenta,
                 masses=masses,
                 celldisp=at.get_celldisp(),
                 constraint=at.constraints,
                 info=at.info)



def make_supercell(prim, P, wrap=True, tol=1e-5):
    """Generate a supercell by applying a general transformation (*P*) to
    the input configuration (*prim*).

    This function is a modified version of ASEs build/supercells.py.
    The modification here fixes a bug in ASEs implementation, introduced in
    ASEs version 3.18.0: for certain transformation matrices, the determinant of
    the matrix is negative, and the function exits with error, since a negative
    number of atoms is obtained.
    An example script demonstrating the error is::

       from ase.build import make_supercell
       from ase.atoms  import Atoms

       prim = Atoms(symbols='Cu',
             pbc=True,
             cell=[[0.0, 1.805, 1.805], [1.805, 0.0, 1.805], [1.805, 1.805, 0.0]])

       sc = make_supercell(prim, P = [[ 2,  2, -2],[ 2, -2,  2],[-2,  2,  2]], wrap = True, tol = 1e-05)

    The release 3.22.0 of ASE still contains the bug.
    It was informed to ASEs developers on the 3.7.2021

    The transformation is described by a 3x3 integer matrix
    `\mathbf{P}`. Specifically, the new cell metric
    `\mathbf{h}` is given in terms of the metric of the input
    configuration `\mathbf{h}_p` by `\mathbf{P h}_p =
    \mathbf{h}`.

    **Parameters:**

    ``prim``: ASE Atoms object
        Input configuration.
    ``P``: 3x3 integer matrix
        Transformation matrix `\mathbf{P}`.
    ``wrap``: bool
        wrap in the end
    ``tol``: float
        tolerance for wrapping
    """

    supercell_matrix = P
    supercell = clean_matrix(supercell_matrix @ prim.cell)

    # cartesian lattice points
    lattice_points_frac = lattice_points_in_supercell(supercell_matrix)
    lattice_points = np.dot(lattice_points_frac, supercell)

    superatoms = Atoms(cell=supercell, pbc=prim.pbc)

    for lp in lattice_points:
        shifted_atoms = prim.copy()
        shifted_atoms.positions += lp
        superatoms.extend(shifted_atoms)

    # check number of atoms is correct
    n_target = np.abs(int(np.round(np.linalg.det(supercell_matrix) * len(prim))))
    if n_target != len(superatoms):
        msg = "Number of atoms in supercell: {}, expected: {}".format(
            n_target, len(superatoms)
        )
        raise SupercellError(msg)

    if wrap:
        superatoms.wrap(eps=tol)

    return superatoms

def decorate_supercell(scell, atoms):
    """ Create a Structure instance by decorating a SuperCell with an Atoms object from ASE.
    """
    from clusterx.structure import Structure

    ans = []
    for i1, p1 in enumerate(scell.get_positions()):
        for i2, p2 in enumerate(atoms.get_positions()):
            if np.linalg.norm(p1-p2) < 1e-5:
                ans.append(atoms.get_atomic_numbers()[i2])

    return Structure(scell, ans)

def super_structure(struc0, d):
    """ Create a super structure

    This function takes a ``Structure`` instance (``struc0``) and creates a new structure
    which is obtained as the periodic repetition of the original structure
    along its unit cell vectors. The number of repetitions along each cell vector is given
    by the three components of the input integer vector ``d``.

    **Parameters:**

    ``struc0``: Structure object
        Original structure for the superstructure.
    ``d``: int, three-component integer array, or 3x3 integer array
        The super structure is obtained by the transformation d S, with d a
        3x3 matrix of integer and S the supercell cell vectors.
    """
    from clusterx.structure import Structure
    from clusterx.super_cell import SuperCell
    from ase.build import make_supercell

    if np.shape(d) == ():
        n = np.zeros((3,3), int)
        np.fill_diagonal(n, [d,d,d])
    elif np.shape(d) == (3,):
        n = np.zeros((3,3), int)
        np.fill_diagonal(n, d)
    elif np.shape(d) == (3,3):
        n = np.array(d)
    else:
        print("ERROR (clusterx.utils.super_structure()): ")

    p0 = struc0.get_supercell().get_transformation()
    p1 = n @ p0

    atoms0 = struc0.get_atoms()
    atoms1 = make_supercell(atoms0, n)

    scell1 = SuperCell(struc0.get_parent_lattice(), p1)

    return decorate_supercell(scell1, atoms1)

def sset_equivalence_check(sset, to_primitive = True, cpool = None, basis = "trigonometric", comat = None, pretty_print = False):
    """Find equivalent structures in a StructuresSet object

    Equivalence is determined *i)* in terms of symmetry between structures
    or *ii)* in terms of cluster basis representation (if a ClustersPool object
    or a correlation matrix is given).

    In the first case, the `SymmetryEquivalenceCheck tool of ASE <https://wiki.fysik.dtu.dk/ase/ase/utils.html#ase.utils.structure_comparator.SymmetryEquivalenceCheck>`_ is used.

    **Parameters:**

    ``sset``: StructuresSet object
        The structures set object to be analyzed.

    ``to_primitive``: Boolean (default: ``True``)
        If ``True`` the structures are reduced to their primitive cells.
        This feature requires ``spglib`` to installed
        (*cf.* `ASE's SymmetryEquivalenceCheck <https://wiki.fysik.dtu.dk/ase/ase/utils.html?highlight=to_primitive#ase.utils.structure_comparator.SymmetryEquivalenceCheck>`_)

    ``cpool``: ClustersPool object (default: ``None``)
        This parameter is optional. If given, the equivalence of a pair of structures
        is determined according to their cluster basis representation: two
        structures with the same cluster correlations for the clusters in ``cpool``
        are considered equivalent.

    ``basis``: string (default: ``"trigonometric"``)
        Only used if ``cpool`` is not ``None``. Site basis functions used in the determination
        of cluster correlations.

    ``comat``: 2D array of floats (default: ``None``)
        It overrides ``cpool`` and ``basis``. If a correlation matrix for the
        structures set sset was pre-computed, the equivalence in terms of cluster
        basis representation is performed by comparing the rows of this matrix (each
        row must correspond to a structure in the structures set).


    **Returns:**
    Returns a dictionary. The keys (k) are structure indices of unique representative structures,
    and the values (v) are arrays of integer, indicating all structure indices equivalent to k
    (containing k itself too). For instance, the dictionary::

        {"0": [0, 1, 3, 8, 9],
         "2": [2, 5, 6],
         "4": [4, 7]}

    | indicates that in the structures set with indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] there are
      just three distinct structures. These can be represented by strucutres 0, 2 and 4.
      The structures [0, 1, 3, 8, 9] are all equivalent, etc.
    | Notice that here equivalence is used in the sense explained above: It is symmetrical
      equivalence only if ``cpool`` and ``comat`` are None.
    """
    import numpy as np

    comp = None
    from clusterx.utils import isclose

    if cpool is None and comat is None:
        from ase.utils.structure_comparator import SymmetryEquivalenceCheck
        comp = SymmetryEquivalenceCheck(to_primitive = to_primitive)
    elif comat is None:
        from clusterx.correlations import CorrelationsCalculator
        ccalc = CorrelationsCalculator(basis = basis, parent_lattice = sset.get_parent_lattice(), clusters_pool = cpool)
        comat = ccalc.get_correlation_matrix(sset)

    nstr = len(sset)

    crossedout = []
    id_str_list = {}
    for i in range(nstr):
        if i not in crossedout:
            crossedout.append(i)
            subset = [i]

            if comat is not None:
                corr_i = comat[i,:]
            else:
                atoms_i = sset[i].get_atoms()

            for j in range(i+1, nstr):

                if comat is not None:
                    corr_j = comat[j,:]
                    check = isclose(corr_i, corr_j)
                else:
                    atoms_j = sset[j].get_atoms()
                    check = comp.compare(atoms_i, atoms_j)

                if check:
                    crossedout.append(j)
                    subset.append(j)

            id_str_list[i] = np.array(subset, dtype='i4')

    if pretty_print:
        print(id_str_list)

    return id_str_list

def atoms_equivalence_check(atoms, to_primitive = True, pretty_print = False):
    """Find equivalent structures in an array of Atoms objects

    Equivalence is determined in terms of symmetry between structures

    The `SymmetryEquivalenceCheck tool of ASE <https://wiki.fysik.dtu.dk/ase/ase/utils.html#ase.utils.structure_comparator.SymmetryEquivalenceCheck>`_ is used.

    **Parameters:**

    ``atoms``: array of Atoms objects
        The structures to be analyzed.

    ``to_primitive``: Boolean (default: ``True``)
        If ``True`` the structures are reduced to their primitive cells.
        This feature requires ``spglib`` to installed
        (*cf.* `ASE's SymmetryEquivalenceCheck <https://wiki.fysik.dtu.dk/ase/ase/utils.html?highlight=to_primitive#ase.utils.structure_comparator.SymmetryEquivalenceCheck>`_)


    **Returns:**
    Returns a dictionary. The keys (k) are structure indices of unique representative structures,
    and the values (v) are arrays of integer, indicating all structure indices equivalent to k
    (containing k itself too). For instance, the dictionary::

        {"0": [0, 1, 3, 8, 9],
         "2": [2, 5, 6],
         "4": [4, 7]}

    | indicates that in the structures set with indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] there are
      just three distinct structures. These can be represented by strucutres 0, 2 and 4.
      The structures [0, 1, 3, 8, 9] are all equivalent, etc.
    | Notice that here equivalence is used in the sense explained above: It is symmetrical
      equivalence only if ``cpool`` and ``comat`` are None.
    """
    import numpy as np

    from clusterx.utils import isclose

    from ase.utils.structure_comparator import SymmetryEquivalenceCheck
    comp = SymmetryEquivalenceCheck(to_primitive = to_primitive)

    nstr = len(atoms)

    crossedout = []
    id_str_list = {}
    for i in range(nstr):
        if i not in crossedout:
            crossedout.append(i)
            subset = [i]

            atoms_i = atoms[i]

            for j in range(i+1, nstr):

                atoms_j = atoms[j]
                check = comp.compare(atoms_i, atoms_j)

                if check:
                    crossedout.append(j)
                    subset.append(j)

            id_str_list[i] = np.array(subset, dtype='i4')

    if pretty_print:
        print(id_str_list)

    return id_str_list


def report_sset_equivalence_check(sset, sset_equivalence_check_output, property_name = None, tol = 0.0):
    """Generate report of equivalent structures

    Writes to files: ``sset_unique_sym.json`` and ``sset_unique_gss.json``.

    The first contains the structures whose index are given by the keys of the dictionary
    returned by ``sset_equivalence_check()``.

    The second contains the structures of every equivalence subset where the value of
    ``property_name`` is minimal. So, if the property is an energy, the final set will contain all
    lowest energy structures of every equivalence subset.

    """
    import operator

    folders = sset.get_folders()

    id_str_list = sset_equivalence_check_output

    unique = []
    for k, v in id_str_list.items():
        unique.append(v[0])

    sset_unique = sset.get_subset(unique)
    sset_unique.serialize("sset_unique_sym.json", overwrite = True)

    unique_gss = [] # Collect the lowest energy structure from each subset
    pvals = sset.get_property_values(property_name)
    for k, v in id_str_list.items():
        if len(v) > 1:
            pvals_subset = operator.itemgetter(*v)(pvals)
            i_min = np.argmin(pvals_subset)
            #unique_gss.append(i_min)
            unique_gss.append(v[i_min])
        else:
            unique_gss.append(v[0])

    sset_unique_gss = sset.get_subset(unique_gss)
    sset_unique_gss.serialize("sset_unique_gss.json", overwrite = True)

    decim = 5
    tol = tol
    print("Structure indices start from 1 (corresponding to sset[0]).")
    print(f'floats shown below are rounded to {decim} decimals.')
    print(f'Show only sets where maximum energy variation is larger than {tol}.')
    for k, v in id_str_list.items():
        subset = sset.get_subset(v)
        pvals = subset.get_property_values(property_name)
        ediff = round(np.amax(pvals)-np.amin(pvals),decim)
        if len(v) > 1 and ediff>=tol:
            print("\n========================================================")
            #subset = sset.get_subset(v)
            subset.serialize(f'repetitions-{k}.json')
            #pvals = subset.get_property_values('energy_mixing_atom')
            #print(k, pvals)
            #print(k, round(np.amax(pvals)-np.amin(pvals),5), np.around(pvals, decimals = 5))
            print(f'Found {len(pvals)} structures possibly equivalent to structure:')
            print(k+1)
            print("List of str indices:")
            print(np.array(id_str_list[k])+1)
            print("Energies for these structures:")
            print(np.around(pvals, decimals = decim))
            print("Difference between maximum and minimum energies:")
            #print(round(np.amax(pvals)-np.amin(pvals),decim))
            print(ediff)
            print("Folders:")
            print(operator.itemgetter(*id_str_list[k])(folders))
            #print(folders)
            #print(map(folders.__getitem__, np.array(id_str_list[k]).tolist()))
