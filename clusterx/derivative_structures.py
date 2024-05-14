# Copyright (c) 2015-2023, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
from clusterx.structures_set import StructuresSet
from clusterx.super_cell import SuperCell
from clusterx.utils import _is_integer_matrix
from sys import getsizeof
from itertools import combinations
import scipy
from tqdm import tqdm

class DSGenerator():
    """Generation of derivative structures
    
    A :class:`DSGenerator <clusterx.derivative_structure.DSGenerator>` object is used to generate 
    derivative structures from a :class:`parent lattice <clusterx.parent_lattice.ParentLattice>`. 

    **Parameters:**

    ``parent_lattice``: :class:`ParentLattice <clusterx.parent_lattice.ParentLattice>` object
        Parent lattice object. Derivative structures originate from this parent lattice.
    """
    def __init__(self, parent_lattice):
        self.plat = parent_lattice
        self.scell_sizes = []
        self.scell_shapes = []
        self.num_subs = []
        self.sigmas = []
        self.properties = {}

    def generate(self, supercell_sizes, num_subs_list):
        """Generate derivative structures

        **Parameters:**

        ``supercell_sizes``: list or array of int
            List  of integers indicating the number of unit cells in each derivative supercell
        ``num_subs_list``: ragged list of lists or arrays of integers
            every list or array in the ragged list, indicate the number of substituents to be 
            considered in a given supercell. The first dimension must coincide with the 
            dimension of ``supercell_sizes``.
        """
        for sc_size, num_subs in zip(supercell_sizes, num_subs_list): # sc_size: number of unit cells in supercell
            #unique_scs, unique_trafos = get_unique_supercells_large_angles(sc_size, self.plat, [-2,-1,0,1,2])
            unique_scs, unique_trafos = get_unique_supercells(sc_size, self.plat)

            for idx, t in enumerate(unique_trafos):
                print(f'Start scell shape {idx+1} of {len(unique_trafos)}')
                scell_fe = SuperCell(self.plat,t)
                natoms = scell_fe.get_natoms()
            
                symper = scell_fe.get_sym_perm()
            
                for nsubs in num_subs:

                    full_list = set()
                    ssites = scell_fe.get_substitutional_sites()

                    i=0
                    list_sigmas = []

                    n_max = int(scipy.special.binom(len(ssites), nsubs))
                    print(f'Max nr of configurations for {nsubs} substitutions in {natoms}-atom size scell (no sym accounted): {n_max}')

                    for con in tqdm(combinations(ssites,nsubs), total=n_max, desc="Finding unique sigmas"):
                        sigmas = np.zeros(natoms, dtype = "int")
                        np.put(sigmas, con, [1])

                        if tuple(sigmas.tolist()) not in full_list:
                            list_sigmas.append(sigmas)

                        _sigmass = []
                        for per in symper:
                            _sigmass.append(sigmas[np.ix_(per)])
                        sigmass = np.unique(_sigmass, axis=0)
                        for s_ in sigmass:
                            full_list.add(tuple(s_.tolist()))
                            
                        i = i+1
                        
                        """if i == 10000:
                            print("SSET_Size = "+str(getsizeof(list_sigmas)))
                            print("full_list_Size = "+str(getsizeof(full_list)))
                            i=0
                        """
                    self.scell_sizes.append(sc_size)
                    self.scell_shapes.append(t)
                    self.num_subs.append(nsubs)
                    self.sigmas.append(list_sigmas)

    def compute_properties(self, property_name, cemodel):
        self.properties[property_name] = []
        self.concentrations = []

        scsize0 = self.scell_sizes[0]
        scshape0 = self.scell_shapes[0]
        scell = SuperCell(self.plat, p=scshape0)
        for i, (scsize, scshape, nsubs, sigmas) in tqdm(enumerate(zip(self.scell_sizes, self.scell_shapes, self.num_subs, self.sigmas)), total=len(self.scell_sizes), desc="Computing properties"):
            
            self.properties[property_name].append([])
            self.concentrations.append([])

            if scsize != scsize0 or (scshape != scshape0).any():
                scell = SuperCell(self.plat, p=scshape)
                scsize0 = scsize
                scshape0 = scshape

            for sigma in sigmas:
                struc = scell.gen_structure(sigmas = sigma)
                conc = struc.get_fractional_concentrations()
                pval = cemodel.predict(struc)
                self.properties[property_name][i].append(pval)
                self.concentrations[i].append(conc[0][1])




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
    pl_cell = parent_lattice.get_cell()

    hnfs = get_HNFs(n,pbc=parent_lattice.get_pbc())

    all_scs = []
    for hnf in hnfs:
        all_scs.append(np.dot(hnf,pl_cell))

    n_scs = len(all_scs)
    unique_scs = []
    unique_trafos = []
    sc_sg, sc_sym = parent_lattice.get_sym() # Scaled to parent_lattice
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


def get_HNFs(n,pbc=(1,1,1)):
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

def _get_minimal_trafo(h, all_matrices = None, cell = None):

    def minimum_key(x):
        return max(_get_normalized_scalar_products(np.dot(x, cell)))

    trafos = [np.dot(np.reshape(mat, (3,3)), h) for mat in all_matrices]
    minimal = min(trafos, key = minimum_key)
    return minimal

def _get_normalized_scalar_products(s: np.ndarray):
    """
    For a matrix of column vectors, return the normalized scalar products.
    
    **Parameters:**
    
    ``s``: numpy.ndarray
        matrix of transformed cell vectors as columns
        
    **Returns:**
    
    Normalized scalar products between unique vector pairs
    """
    p_ij = []
    for i, j in [(0,1), (0,2), (1,2)]:
        S_i = s.T[i]
        S_j = s.T[j]
        denominator = np.linalg.norm(S_i) * np.linalg.norm(S_j)
        nominator = abs(np.dot(S_i, S_j))
        p_ij.append(nominator/denominator)
    p_ij = np.array(p_ij)
    return p_ij


def get_unique_supercells_large_angles(n, parent_lattice: object, elements: list):
    """
    Return all unique supercells with large angles. 
    Transformation of those supercells is done by unimodal matrices with matrix elements given by the paramter ``elements``   

    **Parameters:**
    
    ``elements``: list[int]
        transformation vector elements 

        example: [-3,-2,-1,0,1,2,3]

        The algorithm will search all combinations of 3x3 matrices composed of those ``elements``.

    """

    from itertools import product
    import multiprocessing
    from clusterx.super_cell import SuperCell # needed by make_supercell
    from functools import partial

    _, harray = get_unique_supercells(n, parent_lattice)

    parent_lattice_cell = parent_lattice.get_cell().array.T

    all_matrices = list(filter(lambda x: abs(np.linalg.det(np.reshape(x, (3,3)))) == 1, product(elements, repeat = 9)))

    with multiprocessing.Pool() as pool:
        small_angle_trafos = pool.map(partial(_get_minimal_trafo, all_matrices = all_matrices, cell = parent_lattice_cell), harray)

    return [SuperCell(parent_lattice, p) for p in small_angle_trafos], small_angle_trafos