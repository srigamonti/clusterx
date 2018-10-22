import numpy as np
import os
import copy

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
                if tol is None:
                    if v1 != v2:
                        return False
                else:
                    if not isclose(v1,v2,tol):
                        return False

        except TypeError:
            #if isinstance(d1[k],dict):
            #    for subk in d1[k].keys():
                    
            if _is_integrable(d1[k]):
                if d1[k]!=d2[k]:
                    return False
            else:
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
    list of strings. For example, if prepend is set to ``"run_"``, then this
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

        if prefix != '' or suffix != '':
            d = folder[len(prefix):][:-len(suffix)]
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

def get_unique_supercells(n,parent_lattice):
    """Find full list of unique supercells of index n.

    Following Ref.[1], the complete set of symmetrically inequivalent HNFs of
    index ``n``, for a given ``parent_lattice``, is determined and returned.

    [1] Gus L. W. Hart and Rodney W. Forcade *Phys. Rev. B*
    **80**, 014120 (2009).

    **Parameters:**

    ``n``: integer
        index of the supercells, i.e., the number of atoms in the supercells is
        ``n*parent_lattice.get_natoms()``.
    ``parent_lattice``: ParentLattice object
        The parent lattice

    **Returns:** two arrays containing 3x3 matrices. The matrices ``S`` of the
    first array contain the cartesian coordinates of the supercell vectors (row
    wise), while the matrices ``H`` in the second array are the (integer)
    transormation matrices with respect to the parent lattice ``U``. That is,
    :math:`S=HU`

    **Example:**
    In the following example, all the supercells of index 4 for the FCC lattice
    are found. The supercells are stored in the file
    ``unique_supercells-fcc.json`` for visualization with the command
    ``$>ase gui unique_supercells-fcc.json`` ::

        from clusterx import utils
        from clusterx.parent_lattice import ParentLattice
        from clusterx.structures_set import StructuresSet
        from clusterx.super_cell import SuperCell
        from clusterx.structure import Structure
        from ase.data import atomic_numbers as an
        from ase import Atoms
        import numpy as np

        a=3
        cell = np.array([[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]])
        positions = np.array([[0,0,0]])
        sites = [[an["Cu"],an["Au"]]]
        pris_fcc = Atoms(cell=cell*a,positions=positions*a)

        pl = ParentLattice(pris_fcc,sites=sites)

        unique_scs, unique_trafos = utils.get_unique_supercells(4,pl)

        sset = StructuresSet(pl,filename="unique_supercells-fcc.json")
        for t in unique_trafos:
            scell = SuperCell(pl,t)
            sset.add_structure(Structure(scell,scell.get_atomic_numbers()),write_to_db = True)

    The generated structures are the same as those found in Fig. 2 and Table IV
    of Phys. Rev. B 77, 224115 2008.

    The next example, shows a case of reduced dimensionality, that of a 2D
    square lattice::

        a=3.1
        cell = np.array([[1,0,0],[0,1,0],[0,0,1]])
        positions = np.array([[0,0,0]])
        sites = [[12,13]]
        pris = Atoms(cell=cell*a, positions=positions*a)

        pl = ParentLattice(pris, sites=sites, pbc=(1,1,0))

        unique_scs, unique_trafos = utils.get_unique_supercells(4,pl)

        sset = StructuresSet(pl,filename="test_get_unique_supercells-square_lattice.json")
        for t in unique_trafos:
            scell = SuperCell(pl,t)
            sset.add_structure(Structure(scell,scell.get_atomic_numbers()),write_to_db = True)

        #isok0 = len(unique_scs) == 4 and
        print("n: ",len(unique_scs))
        print("SCS: ", unique_scs)
        print("TRA: ", unique_trafos)

    The resulting supercells in this example correspond to Fig. 1 of
    Computational Materials Science 59 (2012) 101–107

    """
    from clusterx.symmetry import get_spacegroup
    pl_cell = parent_lattice.get_cell()

    hnfs = get_all_HNF(n,pbc=parent_lattice.get_pbc())

    all_scs = []
    for hnf in hnfs:
        all_scs.append(np.dot(hnf,pl_cell))

    n_scs = len(all_scs)
    unique_scs = []
    unique_trafos = []
    sc_sg, sc_sym = get_spacegroup(parent_lattice) # Scaled to parent_lattice
    nexts = np.asarray(np.arange(n_scs))
    unique_scs.append(all_scs[0])
    unique_trafos.append(hnfs[0])

    while len(nexts) > 1:
        i = nexts[0]
        the_list = nexts[1:]
        nexts=[]
        bi = all_scs[i]
        for j in the_list:
            bj = all_scs[j]
            j_is_next = True
            for r in sc_sym['rotations']:
                rr = np.dot(pl_cell,np.dot(r,np.linalg.inv(pl_cell))) # Rotations are in lattice coordinates, so we have to transorm them to cartesian.
                m = np.around(np.dot(np.linalg.inv(bi.T),np.dot(rr,bj.T)),5)
                if _is_integer_matrix(m):
                    j_is_next = False
                    break

            if j_is_next:
                nexts.append(j)

        if len(nexts) > 0:
            unique_scs.append(all_scs[nexts[0]])
            unique_trafos.append(hnfs[nexts[0]])

    return unique_scs, unique_trafos

def _is_integer_matrix(m,rnd=5):
    mi = np.around(m,rnd)
    for k in range(3):
        for l in range(3):
            #if not round(m[k,l],rnd).is_integer():
            if not mi[k,l].is_integer():
                return False
    return True

def get_all_HNF(n,pbc=(1,1,1)):
    """Return complete set of Hermite normal form (HNF) :math:`3x3` matrices
    of index ``n``.

    The algorithm here is based on Equation 1 of Gus L. W. Hart and Rodney W.
    Forcade, *Phys. Rev. B* **77**, 224115 (2008).

    **Parameters:**

    ``n``: integer
        index of the HNF matrices.
    ``pbc``: three bool
        Periodic boundary conditions flags. Examples:
        (1, 1, 0), (True, False, False). Default value: (1,1,1)
    """

    _hnfs = []
    for a in _divisors(n):
        for c in _divisors(int(n/a)):
            f = int(n/(a*c))
            for b in range(c):
                for d in range(f):
                    for e in range(f):
                        hnf = []
                        hnf.append([a,0,0])
                        hnf.append([b,c,0])
                        hnf.append([d,e,f])
                        _hnfs.append(np.array(hnf).T)

    hnfs = []
    for hnf in _hnfs:
        include = True
        for i, bc in enumerate(pbc):
            if not bc and (hnf[i] != np.identity(3,dtype="int")[i]).any():
                include = False
                break
        if include:
            hnfs.append(hnf)

    return(hnfs)

def _divisors(n):
    # get factors and their counts
    # Mainly taken from https://stackoverflow.com/a/37058745
    factors = {}
    nn = n
    i = 2
    while i*i <= nn:
        while nn % i == 0:
            if not i in factors:
                factors[i] = 0
            factors[i] += 1
            nn //= i
        i += 1
    if nn > 1:
        factors[nn] = 1

    primes = list(factors.keys())

    # generates factors from primes[k:] subset
    def generate(k):
        if k == len(primes):
            yield 1
        else:
            rest = generate(k+1)
            prime = primes[k]
            for factor in rest:
                prime_to_i = 1
                # prime_to_i iterates prime**i values, i being all possible exponents
                for _ in range(factors[prime] + 1):
                    yield factor * prime_to_i
                    prime_to_i *= prime

    # in python3, `yield from generate(0)` would also work
    #for factor in generate(0):
    #    yield factor

    r = []
    for factor in generate(0):
        r.append(factor)

    return sorted(r)

def get_cl_idx_sc(cl, sc, method=0, tol=1e-3):
    """Return atom indexes of cluster points in SuperCell

    **Parameters:**

    ``cl``: npoints x 3 matrix
        matrix of cartesian or scaled coordinates of cluster points. Cluster
        positions are expected to be wrapped inside supercell ``sc``
    ``sc``: natoms x 3 matrix
        matrix of cartesian or scaled coordinates of supercell atomic positions.
    ``method``: integer
        Method to use. 0: (slow) nested for loop using numpy allclose. 1: (fast)
        calculates all distances from points in ``cl`` to atoms in ``sc``, and
        return indices for which distances are zero.
    ``tol``: real positive number
        tolerance to determine whether cluster and atom positions are the same.
    """
    from scipy.spatial.distance import cdist

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


def parent_lattice_to_atat(plat, out_fname="lat.in"):
    """Serializes ParentLattice object to ATAT input file

    **Parameters:**

    ``plat``: ParentLattice object
        ParentLattice object to be serialized
    ``out_fname``: string
        Output file path
    """
    cell = plat.get_cell()
    positions = plat.get_scaled_positions()
    sites = plat.get_sites()

    f = open(out_fname,'w+')

    for cellv in cell:
        f.write(u"%2.12f\t%2.12f\t%2.12f\n"%(cellv[0],cellv[1],cellv[2]))

    f.write(u"1.000000000000\t0.000000000000\t0.000000000000\n")
    f.write(u"0.000000000000\t1.000000000000\t0.000000000000\n")
    f.write(u"0.000000000000\t0.000000000000\t1.000000000000\n")

    for i,pos in enumerate(positions):
        stri = u"%2.12f\t%2.12f\t%2.12f\t"%(pos[0],pos[1],pos[2])
        if len(sites[i])>1:
            for z in sites[i][:-1]:
                stri = stri + "%s,\t"%cs[z]
        stri = stri + "%s\n"%cs[sites[i][-1]]

        f.write(stri)

    f.close()

class Exponential():

    def __init__(self, exponent, coefficient = 1):
        self.exponent = exponent
        self.coefficient = coefficient

    def evaluate(self, x):
        return self.coefficient * np.power(x,self.exponent)

    def normalize(self, value):
        self.coefficient = self.coefficient / value

    def multiply_scalar(self, scalar):
        self.coefficient = self.coefficient * scalar



class PolynomialBasisFunction():

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
        rmlist = []
        for exponential in self.exponentials:
            if abs(exponential.coefficient) < 10**(-10):
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
            exponential.normalize(length)

    def multiply_scalar(self, scalar):
        for exponential in self.exponentials:
            exponential.multiply_scalar(scalar)

    def add_polynomial_basis_function(self, polynomial_basis_function):
        for exponential in polynomial_basis_function.exponentials:
            self.add_exponential(exponential.exponent, exponential.coefficient)

    def print_polynomial(self):
        outstring = ''
        for exponential in self.exponentials:
            outstring += str(exponential.coefficient) + ' *  x^' + str(exponential.exponent) + ' + '
        outstring = outstring[:-3]
        print(outstring)


class PolynomialBasis():

    def __init__(self, m = 10, symmetric = False):
        self.m = m
        self.symmetric = symmetric
        self.basis_function_set = {}
        for M in range(1,m+1):
            self.basis_function_set[str(M)] = self.construct(M)

    def construct(self, m):
        basis_functions = []
        for degree in range(m):
            new_function = PolynomialBasisFunction()
            new_function.add_exponential(degree)
            chi = Exponential(degree)
            for basis_function in basis_functions:
                overlap = self.scalar_product(basis_function.evaluate,chi.evaluate, m)
                overlap = overlap * (-1)
                summand = copy.deepcopy(basis_function)
                summand.multiply_scalar(overlap)
                new_function.add_polynomial_basis_function(summand)
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
            sigmas = range(m)
        scalar_product = 0
        for sigma in sigmas:
            scalar_product += function1(sigma) * function2(sigma)
        scalar_product = scalar_product * scaling
        return scalar_product

    def print_basis_functions(self, m):
        for function in self.basis_function_set[str(m)]:
            function.print_polynomial()

    def evaluate(self, alpha, sigma, M):
        if M > self.m:
            self.basis_function_set[str(M)] = self.construct(M)
        return self.basis_function_set[str(M)][int(alpha)].evaluate(sigma)


def poppush(x, val):
    """Left-shift array one position and writes new value to the right. Returns average.

    **Parameters:**
    ``x``: numpy array
    ``val``: int or float
    """
    x[:-1] = x[1:]; x[-1] = val
    return x.mean()
