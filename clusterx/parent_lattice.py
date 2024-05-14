# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from ase import Atoms
import numpy as np
import copy
from clusterx.utils import _is_integrable
from ase.db import connect
from clusterx.symmetry import get_spacegroup
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

def unique_non_sorted(a):
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]

class ParentLattice(Atoms):
    """**Parent lattice class**

    The parent lattice in a cluster expansion defines the atomic positions,
    the periodicity, the substitutional species, and the spectator sites, i.e. those
    atoms which are present in the crystal but are not substituted.
    ``ParentLattice`` subclasses ASE's ``Atoms`` class.

    **Parameters:**

    ``atoms``: ASE's ``Atoms`` object
        ``atoms`` object corresponding to the pristine lattice. If ``sites`` or
        ``site_symbols`` is not None, then the passed ``Atoms`` objects may not
        contain a definition of atomic species, i.e., can be initialized by only
        indicating atomic positions, unit cell, and periodic boundary conditions.
    ``substitutions``: list of ASE's ``Atoms`` objects.
        Every ``Atoms`` object in the list, corresponds to a possible full
        substitution of only one sublattice. If set, overrides ``sites`` (see
        below).
    ``numbers``: list of integer arrays or dict
        This is an array of length equal to the number of atoms in the parent
        lattice. Every element in the array is an array with species numbers,
        representing the species which can occupy the crystal
        sites (see examples below). This is overriden by ``substitutions``
        if set. If a dict, the dict keys must be the site indices.
    ``symbols``: list of strings
        This is an array of length equal to the number of atoms in the parent
        lattice. Every element in the array is an array with species symbols
        (e.g. ``[["Cu","Au"]]``),
        representing the species which can occupy the crystal
        sites (see examples below). This is overriden by ``substitutions``
        if set.
    ``json_db_filepath``: string
        Path to a Json database file, as created by ``ParentLattice.serialize()``.
        Overrides all the above. Allows to create a ``ParentLattice`` object
        from file.
    ``pbc``: three bool
        Periodic boundary conditions flags. Examples:
        (1, 1, 0), (True, False, False). Default value: (1,1,1)

    **Examples:**

    In all the examples below, you may visualize the created parent lattice
    by executing::

        $> ase gui plat.json

    In this example, the parent lattice for a simple binary fcc CuAl Alloy
    is built::

        from ase.build import bulk
        from clusterx.parent_lattice import ParentLattice

        pri = bulk('Cu', 'fcc', a=3.6)
        sub = bulk('Al', 'fcc', a=3.6)

        plat = ParentLattice(pri, substitutions=[sub], pbc=pri.get_pbc())
        plat.serialize(fname="plat.json")

    In the next example, a parent lattice for a complex clathrate compound is
    defined. This definition corresponds to the chemical formula
    :math:`Si_{46-x-y} Al_x Vac_y Ba_{8-z} Sr_z` ::

        from ase.spacegroup import crystal

        a = 10.515
        x = 0.185; y = 0.304; z = 0.116
        wyckoff = [
            (0, y, z), #24k
            (x, x, x), #16i
            (1/4., 0, 1/2.), #6c
            (1/4., 1/2., 0), #6d
            (0, 0 , 0) #2a
        ]

        from clusterx.parent_lattice import ParentLattice
        pri = crystal(['Si','Si','Si','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])
        sub1 = crystal(['Al','Al','Al','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])
        sub2 = crystal(['X','X','X','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])
        sub3 = crystal(['Si','Si','Si','Sr','Sr'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])

        plat = ParentLattice(atoms=pri,substitutions=[sub1,sub2,sub3])
        plat.serialize(fname="plat.json")

    This example uses the optional ``sites`` parameter instead of the ``substitutions``
    parameter::

        from ase import Atoms
        from clusterx.parent_lattice import ParentLattice

        a = 5.0
        cell = [[a,0,0],[0,a,0],[0,0,a]]
        scaled_positions = [[0,0,0],[0.25,0.25,0.25],[0.5,0.5,0.5],[0.75,0.75,0.75],[0.5,0.5,0]]
        sites = [[11,13],[25,0,27],[11,13],[13,11],[5]]

        plat = ParentLattice(Atoms(scaled_positions=scaled_positions,cell=cell), sites=sites)
        plat.serialize(fname="plat.json")

    Defined in this way, the crystal site located at [0,0,0] may be occupied by
    species numbers 11 and 13; the crystal site at [0.25,0.25,0.25] may be occupied
    by species numbers 25, 0 (vacancy) or 27; and so on. The pristine lattice has
    species numbers [11,25,11,13,5] (i.e. the first element in every of the lists in sites).
    Notice that sites with atom index 0 and 2 belong to a common sublattice (i.e. [11,13])
    while site with atom index 3 (i.e. [13,11]) determines a different sublattice.

    .. todo::

        * Add in metadata the tags, and the idx_tags relation.

        * override get_distance and get_distances from Atoms. Such that if get_all_distances
            was ever called, it sets a self.distances attribute which is used for get_distance
            and get_distances, saving computation time. Care should be paid in cases where
            positions are updated either by relaxation or deformation of the lattice.
        * remove deprecated sites and site_symbols parameters from __init__(), calls around the code,
            and documentation / tutorials

    **Methods:**
    """

    def __init__(self, atoms=None, substitutions=None, numbers=None, symbols=None, json_db_filepath=None, pbc=None, sites=None, site_symbols=None):
        if numbers is not None:
            if isinstance(numbers, dict):
                sites = []
                for i in range(len(atoms)):
                    sites.append(None)
                for site_index, nrs in numbers.items():
                    sites[int(site_index)] = list(nrs)
            else:
                sites = numbers
        if symbols is not None:
            site_symbols = symbols
        if json_db_filepath is not None:
            db = connect(json_db_filepath)
            substitutions = []
            for i,row in enumerate(db.select()):
                if i == 0:
                    atoms = row.toatoms()
                else:
                    substitutions.append(row.toatoms())

        if pbc is None:
            pbc = atoms.get_pbc()
            
        super(ParentLattice,self).__init__(symbols=atoms,pbc=pbc)

        if atoms is not None:
            self._atoms = atoms.copy()
            self._atoms.set_pbc(pbc)
        else:
            self._atoms = None

        self._natoms = len(self._atoms)

        if sites is not None or site_symbols is not None and substitutions is None:

            if site_symbols is not None:
                from ase.data import atomic_numbers as an
                sites = copy.deepcopy(site_symbols)
                for i, syms in enumerate(site_symbols):
                    for j, sym in enumerate(syms):
                        sites[i][j] = an[sym]

            if sites is not None:
                try:
                    unique_sites = np.unique(sites,axis=0)
                except:
                    try:
                        unique_sites = np.unique(np.array(sites,dtype=object))
                    except AttributeError:
                        raise AttributeError("sites array has problems, look at the documentation.")

            tags = np.zeros(self._natoms).astype(int)
            for ius, us in enumerate(unique_sites):
                for idx in range(self._natoms):
                    if (np.array_equal(sites[idx], us)):

                        tags[idx] = int(ius)

            numbers = np.zeros(self._natoms).astype(int)
            numbers_pris = np.zeros(self._natoms).astype(int)
            for idx in range(self._natoms):
                numbers_pris[idx] = sites[idx][0]
            self.set_atomic_numbers(numbers_pris)
            self._atoms.set_atomic_numbers(numbers_pris)

            substitutions = []
            for ius, us in enumerate(unique_sites):
                for isp in range(1,len(us)):
                    for idx in range(self._natoms):
                        if tags[idx] == ius:
                            numbers[idx] = us[isp]
                        else:
                            numbers[idx] = sites[idx][0]

                    substitutions.append(Atoms(positions=self.get_positions(),cell=self.get_cell(),pbc=self.get_pbc(),numbers=numbers))


        if substitutions is None:
            substitutions = []
        self._subs = []
        self._set_substitutions(substitutions)

    def __eq__(self, other):
        """Check identity of two ParentLattice objects
        """

        a1 = self.get_pristine()
        a2 = other.get_pristine()
        s1 = self.get_substitutions()
        s2 = other.get_substitutions()

        areeq = True
        if a1 != a2: return False

        for ss1,ss2 in zip(s1,s2):
            if ss1 != ss2: return False

        return True


    def copy(self):
        """Return a copy."""
        pl = self.__class__(atoms=self._atoms, substitutions=self._subs, pbc=self.get_pbc())

        pl.arrays = {}
        for name, a in self.arrays.items():
            pl.arrays[name] = a.copy()
        pl.constraints = copy.deepcopy(self.constraints)
        return pl

    """
    def get_atoms(self):
        "Get a copy of the Atoms object representing the pristine parent lattice."
        return self._atoms.copy()
    """

    def get_sym(self):
        """Get space symmetry of a ParentLattice object.
        """
        try:
            return self.sc_sg, self.sc_sym
        except:
            self.sc_sg, self.sc_sym = self._compute_sym()
            return self.sc_sg, self.sc_sym

    def _compute_sym(self):
        return get_spacegroup(self)

    def get_natoms(self):
        """Get the total number of atoms."""
        return len(self._atoms)

    def get_positions(self, wrap=False, **wrap_kw):
        return super(ParentLattice, self).get_positions(wrap, **wrap_kw)
    
    def get_cell(self, complete = False):
        return super(ParentLattice, self).get_cell(complete)

    def _set_substitutions(self,substitutions=[]):
        if len(substitutions) != 0:
            for atoms in substitutions:
                if self._natoms != len(atoms):
                    raise ValueError('Substitutions array has wrong length: %d != %d.' %
                                     (len(self._atoms), len(substitutions)))
                else:
                    self._subs.append(atoms)

        self._max_nsub = len(self._subs)
        self._set_tags()

    def is_nary(self,n):
        """Check whether lattice is n-ary

        Returns ``True`` if the structure is based on a n-ary ParentLattice object,
        i.e., whether it consists of a single n-ary substitutional sublattice (plus any number of
        spectator sublattices). Otherwise returns ``False``.

        **Example:**
        Check whether a structure is a ternary::

            from clusterx.parent_lattice import ParentLattice
            from clusterx.super_cell import SuperCell
            from ase.build import bulk

            cu = bulk("Cu","fcc")
            au = bulk("Au","fcc")
            ag = bulk("Ag","fcc")

            plat = ParentLattice(atoms=cu, substitutions=[au,ag])

            scell = SuperCell(plat, [[2,2,-2],[2,-2,2],[-2,2,2]])

            s = scell.gen_random(nsubs={0:[10,3]})

            If s.is_nary(3):
                print("The structure is ternary")


        **Parameters:**

        ``n``: int
            Integer number indicating how many different species can be hosted by
            the substitutional sublattice.
        """
        _is_nary = False
        sublt = self.get_sublattice_types()
        m = np.zeros(len(sublt),dtype=int)
        for i,(k,v) in enumerate(sublt.items()):
            m[i]= len(v)-1

        if (np.sum(m) == n-1) and (np.sum(m) in m):
            _is_nary = True

        return _is_nary



    def _set_tags(self):
        """Set the ``tags`` attribute of the Atoms object, and the ParentLattice
        attributes ``idx_subs`` and ``sites`` which define the substitutional
        framework of a ParentLattice.

        Example:

        Supose a ParentLattice object was initialized with four Atoms objects
        a1, a2, a3 and a4::

          parlat = ParentLattice(a1,[a2,a3,a4])

        i.e. a1 represents the pristine lattice and a2 to a4 are the
        possible substitutions.

        Now suppose that::

          a1.get_chemical_symbols() -> [Si,Si,Si,Ba,Ba,Na]
          a2.get_chemical_symbols() -> [Al,Al,Al,Ba,Ba,Na]
          a3.get_chemical_symbols() -> [Si,Si,Si, X, X,Na]
          a4.get_chemical_symbols() -> [Si,Si,Si,Sr,Sr,Na]

        Then::

          tags -> [0,0,0,1,1,2]
          idx_subs -> {0: [14,13], 1: [56,0,38], 2:[11]}
          sites -> {0: [14,13], 1: [14,13], 2: [14,13], 3: [56,0,38], 4: [56,0,38], 5:[11]}

        This means that sites with tag=0 can be occupied by species 14 and 13,
        and sites with tag=1 can be occupied by species 56, vacancy, or species 38.
        """
        all_numbers = np.zeros((self._max_nsub+1,self._natoms),dtype=int)

        all_numbers[0] = self._atoms.get_atomic_numbers()

        for i in range(self._max_nsub):
            all_numbers[i+1] = self._subs[i].get_atomic_numbers()

        all_numbers = all_numbers.T
        unique_subs, tags = np.unique(all_numbers,axis=0,return_inverse=True)
        self.set_tags(tags)

        self.idx_subs = {i: unique_non_sorted(unique_subs[i]) for i in range(len(unique_subs))}
        self.sites = { i:self.idx_subs[j] for i,j in enumerate(tags)}


    def get_substitutional_sites(self):
        """Return atom indexes which may be substituted
        """
        st = self.get_substitutional_tags()
        ss = []
        for i,tag in enumerate(self.get_tags()):
            if tag in st:
                ss.append(i)
        return ss

    def get_substitutional_atoms(self):
        """Return Atoms object for pristine lattice containing only sites which may be substituted
        """
        ats = Atoms(cell=self.get_cell(), pbc=self.get_pbc())
        pats = self.get_pristine()
        ss = self.get_substitutional_sites()
        for i in ss:
            ats.append(pats[i])
        return ats

    def get_n_sub_sites(self, unique=True):
        """Return total number of substitutional sites

        **Parameters:**

        ``unique``: Boolean (default ``True``)
            If ``True``, return the number of site-types which may be substituted.
            If ``False``, return the number of sites which may be substituted.
        """
        if unique:
            return len(self.get_substitutional_tags())
        else:
            st = []
            for tag in self.get_tags():
                if len(self.idx_subs[tag]) > 1:
                    st.append(tag)

            return len(st)

    def get_spectator_sites(self):
        """Return atom indexes which may not be substituted
        """
        st = self.get_spectator_tags()
        ss = []
        for i,tag in enumerate(self.get_tags()):
            if tag in st:
                ss.append(i)
        return ss

    def get_substitutional_tags(self):
        """Return site types for substitutional sites
        """
        st = []
        for tag in self.get_tags():
            if len(self.idx_subs[tag]) > 1:
                st.append(tag)
        return np.unique(np.asarray(st))

    def get_spectator_tags(self):
        """Return site types for non substitutional sites
        """
        st = []
        for tag in self.get_tags():
            if len(self.idx_subs[tag]) == 1:
                st.append(tag)
        return np.unique(np.asarray(st))

    def get_nsites_per_type(self):
        """Return number of sites per site type.

        For instance, suppose that the ``sites`` definition of the structure is::

            sites = {0: [14,13], 1: [14,13], 2: [14,13], 3: [56,0,38], 4: [56,0,38], 5:[11]}

        The ``site_types`` dictionary (as returned by ``idx_subs()``) for this will be::

            site_types = {0: [14,13], 1: [56,0,38], 2:[11]}

        Thus, the tags definition will be::

            tags = [0,0,0,1,1,2]

        and, the number of sites per site type (what this function returns)::

            {0: 3, 1: 2, 2: 1}

        That is, there are three sites of type 0, 2 sites of type 1, and 1 site of
        type 2.

        """
        nsites_per_type = {}
        tags = self.get_tags()

        idx_subs = self.get_idx_subs()
        for site_type, _ in idx_subs.items():
            nsites_per_type[site_type] = tags.tolist().count(int(site_type))

        return nsites_per_type

    def get_sites(self):
        """Return dictionary of sites

        The keys of the ``sites`` dictionary correspond to the atom indices in
        the ParentLattice object. The value for each key is an array of
        integer number representing the species-number that may occupy the site.
        For instance, if a ParentLattice object consists of six positions, such that
        the first three positions can be occupied by species 14 or 13, the 4th
        and 5th positions by species 56, vacant or species 38, and the 6th position
        can be only occupied by species 11, then the sites dictionary reads::

            sites = {0: [14,13], 1: [14,13], 2: [14,13], 3: [56,0,38], 4: [56,0,38], 5:[11]}
        """
        return self.sites
    
    def get_ems(self):
        nspt = self.get_sites()
        ems = np.zeros(len(nspt), dtype=int)
        for k,v in nspt.items():
            ems[k] = len(v)
        return ems

    def get_substitutions(self):
        """Return array of (references to) Atoms objects corresponding to fully
        substituted configurations.
        """
        return self._subs

    def get_pristine(self):
        """Return (reference to) Atoms object of pristine configuration
        """
        return self._atoms

    def get_all_atoms(self):
        """Return list of Atoms objects of pristine and fully substituted configurations.

        Returned Atoms objects are copies of members ``self._atoms`` and ``self._subs``.
        """
        a = []
        a.append(self._atoms.copy())
        for sub in self._subs:
            a.append(sub.copy())
        return a

    def get_sublattice_types(self, pretty_print = False):
        """Return dictionary of site type indexes and substitutional species

        The format of the returned dictionary is::

            {'0':[pri0,sub00,sub01,...],'1':[pri1,sub10,sub11,...]...}

        where the key indicates the site type, ``pri#`` the atomic number of the pristine structure and
        ``sub##`` the possible substitutional atomic numbers for the site.

        **Parameters:**

        ``pretty_print``: boolean
            Whether to return a dictionary or a self-explanatory and nicely
            formatted string.

            DEPRECATED: Use `print_sublattice_types()` instead.

        **Examples:**
        For instance, if a ParentLattice object consists of six positions, such that
        the first three positions can be occupied by species 14 or 13, the 4th
        and 5th positions by species 56, vacant or species 38, and the 6th position
        can be only occupied by species 11, then there are three site types, and the
        returned dictionary will look like::

            {0: [14,13], 1: [56,0,38], 2:[11]}

        .. todo::
            give option to also print the list of atom indexes for every site type.
        """
        if not pretty_print:
            return self.idx_subs
        else:
            warnings.warn("Using `pretty_print=True` is deprecated. Use `print_sublattice_types()` instead.", category=FutureWarning)
            self.print_sublattice_types()

    def print_sublattice_types(self):
        """Print chemical symbols and atomic numbers for each sublattice type.
        """
        from ase.data import chemical_symbols as cs
        sld = self.idx_subs
        cs = np.array(cs)
        print("\n+--------------------------------------------------------------------+")
        print("|{0:^68s}|".format("The structure consists of "+str(len(sld))+" sublattices"))
        print("+--------------------------------------------------------------------+")
        print("|{0:^17s}|{1:^30s}|{2:^19s}|".format("Sublattice type","Chemical symbols","Atomic numbers"))
        print("+--------------------------------------------------------------------+")

        for slind,slsps in self.idx_subs.items():
            print("|{0:^17s}|{1:^30s}|{2:^19s}|".format(str(slind),str(cs[slsps]),str(slsps)))
        print("+--------------------------------------------------------------------+\n")

    # Deprecated, use get_sublattice_types instead
    def get_idx_subs(self):
        return self.get_sublattice_types()

    def as_dict(self):
        """Return dictionary with object definition
        """
        dict = {}
        dict.update({"unit_cell" : self.get_cell()})
        dict.update({"pbc" : self.get_pbc()})
        dict.update({"positions" : self.get_positions()})
        dict.update({"numbers" : self.get_sites()})

        return dict

    def plat_from_dict_obsolete(dict):
        cell = np.array(dict["parent_lattice_pristine_unit_cell"])
        pbc = dict["parent_lattice_pbc"]
        positions = np.array(dict["parent_lattice_pristine_positions"])
        _site_types = dict["parent_lattice_idx_subs"]
        _tags = np.array(dict["parent_lattice_tags"])

        #numbers = np.zeros(len(positions), dtype=int)
        numbers = []
        for site_index,tag in enumerate(_tags):
            numbers.append(_site_types[tag])
            #numbers[site_index] = _site_types[tag]
        numbers_pris = []
        for nrs in numbers:
            numbers_pris.append(nrs[0])

        prist = Atoms(positions=positions, numbers=numbers_pris, cell=cell, pbc=pbc)
        return ParentLattice(atoms=prist, numbers=numbers, pbc=pbc)

    def plat_from_dict(dict):
        """Generates ParentLattice object from a dictionary as returned by ParentLattice.as_dict()
        """
        cell = np.array(dict["unit_cell"])
        pbc = dict["pbc"]
        positions = np.array(dict["positions"])
        _numbers = dict["numbers"]

        numbers = np.zeros(len(positions), dtype=int)
        for site_index, nrs in _numbers.items():
            numbers[site_index] = nrs[0]

        prist = Atoms(positions=positions, numbers=numbers, cell=cell, pbc=pbc)
        return ParentLattice(atoms=prist, numbers=_numbers, pbc=pbc)

    def get_atom_indices_for_site_type(self, site_type):
        """Return atom indices of the structure of a certain given site type
        """
        return np.where(self.get_tags() == site_type)

    def round_coords(self, decimals = 12):
        """ Round cell vectors and lattice positions coordinates.
            
        Uses ``numpy.around()`` method.
            
        **Parameters:**
        
        ``decimals``: integer
            Number of decimal places to round to (default: 0). If decimals is negative, 
            it specifies the number of positions to the left of the decimal point.
        
        **Return:**
            Returns a new ParentLattice object with rounded coordinates.
        """

        atomss = self.get_all_atoms()
        cell0 = np.around(self.get_cell(), decimals)
        pbc0 = self.get_pbc()
        
        atomss0 = []
        for at in atomss:
            pos0 = np.around(at.get_positions(), decimals)
            ans0 = at.get_atomic_numbers()
            atomss0.append(Atoms(positions = pos0, numbers = ans0, cell = cell0, pbc = pbc0))
            
        return ParentLattice(atoms = atomss0[0], substitutions = atomss0[1:], pbc = pbc0)
		
    def serialize(self, filepath="plat.json", fname=None):
        """Serialize a ParentLattice object

        Writes a `ASE's json database <https://wiki.fysik.dtu.dk/ase/ase/db/db.html>`_
        file containing a representation of the parent lattice.
        An instance of the ParentLattice class can be initialized from the created file.
        The created database can be visualized using
        `ASE's gui <https://wiki.fysik.dtu.dk/ase/ase/gui/gui.html>`_, e.g.: ::
        
            ase gui plat.json

        **Parameters:**

        ``filepath``: string
            Output file name.

        ``fname``: string
            *DEPRECATED*, use ``filepath`` instead. Output file name.
        """
        if fname is not None:
            filepath = fname
        
        db = connect(filepath, type="json", append=False)

        images = []
        db.write(self.get_pristine())
        for sub in self.get_substitutions():
            db.write(sub)

        db.metadata = self.as_dict()
