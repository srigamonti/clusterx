from ase import Atoms
import clusterx as c
import numpy as np
import sys
import io
import tempfile
import copy

def unique_non_sorted(a):
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]

class ParentLattice(Atoms):
    """Parent lattice class

    The parent lattice in a cluster expansion defines the atomic positions,
    the periodicity, the substitutional species, and the spectator sites, i.e. those
    atoms which are present in the crystal but are not substituted.
    ``ParentLattice`` subclasses the ASE's ``Atoms`` class.

    TODO:
        override get_distance and get_distances from Atoms. Such that if get_all_distances
        was ever called, it sets a self.distances attribute which is used for get_distance
        and get_distances, saving computation time. Care should be paid in cases where
        positions are updated either by relaxation or deformation of the lattice.
    """

    def __init__(self, atoms=None, substitutions=[],pbc=(1,1,1)):
        super(ParentLattice,self).__init__(symbols=atoms,pbc=pbc)
        #self._atoms = atoms.copy()
        self.set_atoms(atoms)
        self._fname = None
        self._fmt = None
        self._subs = []
        self._natoms = len(self._atoms)
        self.set_substitutions(substitutions)


    def copy(self):
        """Return a copy."""
        pl = self.__class__(atoms=self._atoms, substitutions=self._subs)

        pl.arrays = {}
        for name, a in self.arrays.items():
            pl.arrays[name] = a.copy()
        pl.constraints = copy.deepcopy(self.constraints)
        return pl

    def set_atoms(self, atoms=None):
        """Set the Atoms object representing the pristine parent lattice."""
        if atoms is not None:
            self._atoms = atoms
        else:
            self._atoms = None

    def get_atoms(self):
        """Get the Atoms object representing the pristine parent lattice."""
        return self._atoms

    def get_natoms(self):
        """Get the total number of atoms."""
        return len(self._atoms)

    def set_substitutions(self,substitutions=[]):
        if len(substitutions) != 0:
            for atoms in substitutions:
                if self._natoms != len(atoms):
                    raise ValueError('Substitutions array has wrong length: %d != %d.' %
                                     (len(self._atoms), len(substitutions)))
                else:
                    self._subs.append(atoms)

        self._max_nsub = len(self._subs)
        self._set_tags()


    def _set_tags(self):
        """
        Set the ``tags`` attribute of the Atoms object, and the attributes
        ``idx_subs`` and ``sites`` which define the substitutional framework
        of a ParentLattice.

        Example: Supose a ParentLattice object was initialized with four Atoms objects
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
        st = self.get_substitutional_tags()
        ss = []
        for i,tag in enumerate(self.get_tags()):
            if tag in st:
                ss.append(i)
        return ss

    def get_n_sub_sites(self):
        """Return total number of substitutional sites
        """
        return len(self.get_substitutional_tags())

    def get_spectator_sites(self):
        st = self.get_spectator_tags()
        ss = []
        for i,tag in enumerate(self.get_tags()):
            if tag in st:
                ss.append(i)
        return ss

    def get_substitutional_tags(self):
        st = []
        for tag in self.get_tags():
            if len(self.idx_subs[tag]) > 1:
                st.append(tag)
        return st

    def get_spectator_tags(self):
        st = []
        for tag in self.get_tags():
            if len(self.idx_subs[tag]) == 1:
                st.append(tag)
        return st

    def get_substitutions(self):
        return self._subs

    def get_sites(self):
        return self.sites

    def get_pristine(self):
        return self._atoms

    def get_idx_subs(self):
        """Return dictionary of site type indexes and substitutional species

        The format of the returned dictionary is ``{'0':[pri0,sub00,sub01,...],'1':[pri1,sub10,sub11,...]...}``
        where the key indicates the site type, ``pri#`` the chemical number of the pristine structure and
        ``sub##`` the possible substitutional chemical numbers for the site.
        """
        return self.idx_subs

    def get_fname(self):
        """
        Get file path of serialized parent lattice
        """
        return self._fname

    def get_fmt(self):
        """
        Get file format of serialized parent lattice
        """
        return self._fmt

    def serialize(self, fmt="db", tmp=False, fname=None):
        import os
        from ase.data import chemical_symbols as cs

        self._fmt=fmt

        # get basic info
        cell = self._atoms.get_cell()
        positions = self._atoms.get_scaled_positions()
        sites = self.get_sites()
        #symbols = self._atoms.get_chemical_symbols()

        # set file name
        prefix = "parlat"
        if fmt=="traj":
            suffix=".traj"
        elif fmt=="xyz":
            suffix=".xyz"
        elif fmt=="json":
            suffix=".json"
        elif fmt=="db":
            suffix=".db"
        elif fmt=="cif":
            suffix=".cif"
        elif fmt=="ATAT" or fmt=="atat":
            suffix=".in"


        # serialize
        if fmt == "ATAT" or fmt == "atat":
            if tmp is True:
                f = tempfile.NamedTemporaryFile(mode='w', bufsize=-1, suffix=suffix, prefix=prefix, dir=".")
            else:
                f = open(prefix+suffix,'w')

            for cellv in cell:
                f.write(u"%2.12f\t%2.12f\t%2.12f\n"%(cellv[0],cellv[1],cellv[2]))

            f.write(u"1.000000000000\t0.000000000000\t0.000000000000\n")
            f.write(u"0.000000000000\t1.000000000000\t0.000000000000\n")
            f.write(u"0.000000000000\t0.000000000000\t1.000000000000\n")

            i=0
            #for pos, s in zip(positions, symbols):
            for pos in positions:
                stri = u"%2.12f\t%2.12f\t%2.12f\t"%(pos[0],pos[1],pos[2])
                if len(self.sites[i])>1:
                    for z in self.sites[i][:-1]:
                        stri = stri + "%s,\t"%cs[z]
                stri = stri + "%s\n"%cs[self.sites[i][-1]]

                f.write(stri)
                i = i+1

            if tmp is True:
                return f
            else:
                self._fname = f.name
        elif fmt in  ["xyz", "cif", "traj","json","db"]:
            from ase.io import write

            if fname is None:
                fname = prefix+suffix

            images = []
            images.append(self.get_pristine())
            for sub in self.get_substitutions():
                images.append(sub)

            write(fname,images=images,format=fmt)

            self._fname = fname
