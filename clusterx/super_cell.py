from ase import Atoms
import clusterx as c
import numpy as np
import sys
from ase.visualize import view
from ase.build import make_supercell
from clusterx.parent_lattice import ParentLattice
import clusterx

class SuperCell(ParentLattice):
    """
    **Build a super cell**

    A SuperCell object acts as a blue-print for the creation of structures
    with arbitrary decorations.

    **Parameters:**

    parent_lattice: ParentLattice object
        In **CELL**, a super cell is a periodic repetition of a parent lattice
        object. This parameter defines such a parent lattice for the created
        super cell.
    p: 3x3 integer array
        Transformation matrix :math:`P`. The cartesian coordinates of the
        latttice vectors defining the created SuperCell object, are the rows of
        the matrix :math:`S` defined by :math:`S = PV` where the rows of
        :math:`V` are the cartesian coordinates of the lattice
        vectors of the ParentLattice object. That is, the value of
        `parent_lattice.get_cell()`. The given matrix ``p`` must be compatible
        with the periodic boundary conditions of the parent lattice, i.e. the
        resulting super cell must not contain translations along the
        non-periodic directions.

    .. todo::
        Take into account pbc when building the super cell. Either ignore
        translation in ``p`` along non-periodic directions or warn in some way
        if ``p`` is not compatible with pbc.

    **Methods:**
    """

    def __init__(self, parent_lattice, p):
        self._plat = parent_lattice
        self._p = p
        self.index = int(round(np.linalg.det(p)))
        #prist = make_supercell(parent_lattice.get_atoms(),p)
        prist = make_supercell(parent_lattice.get_pristine(),p)
        subs = [make_supercell(atoms,p) for atoms in parent_lattice.get_substitutions()]
        #ParentLattice.__init__(self, atoms = prist, substitutions = subs )
        super(SuperCell,self).__init__(atoms = prist, substitutions = subs )
        self._natoms = len(self)
        self.set_pbc(self._plat.get_pbc())

    def copy(self):
        #Return a copy.

        sc = self.__class__(self._plat, self._p)

        return sc

    def as_dict(self):
        """Return dictionary with object definition
        """
        plat_dict = super(SuperCell, self._plat).as_dict()
        self._dict = {"tmat":self._p,"parent_lattice":plat_dict}

        return self._dict

    def get_index(self):
        """Return index of the SuperCell

        The index of the supercell is an integer number, equal to the super cell
        volume in units of the parent cell volume.
        """
        return self.index

    def get_parent_lattice(self):
        """
        Return the parent lattice object which defines the supercell.
        """
        return self._plat

    def get_transformation(self):
        """
        Return the transformation matrix that defines the supercell from the
        defining parent lattice.
        """
        return self._p

    def plot(self):
        """
        Plot the pristine supercell object.
        """
        view(self)

    def gen_random(self,nsubs):
        """
        Generate a random Structure with given number of substitutions.

        Parameters:

        nsubs: dictionary
            The nsubs argument must have the format ``{site_type0:[nsub01,nsub02,...], site_type1: ...}``
            where ``site_type#`` and ``nsub##`` are, respectively, an integer indicating
            the site type index and the number of substitutional species of each kind, as returned by
            ``SuperCell.get_idx_subs()``.
        """
        import clusterx.structure

        decoration, sigmas = self.gen_random_decoration(nsubs)

        return clusterx.structure.Structure(SuperCell(self._plat,self._p),sigmas=sigmas)

    def gen_random_decoration(self,nsubs):
        """Generate a random decoration of the super cell with given number of substitutions.

        Parameters:

        nsubs: dictionary
            The format of the dictionary is as follows::

                {site_type1:[n11,n12,...], site_type2:[n21,n22,...], ...}

            it indicates how many substitutional atoms of every kind (n#1, n#2, ...)
            may replace the pristine species for sites of site_type#.

        Returns:

        decoration: list of int
            list with the same length as the pristine structure.
            decoration[atom_idx] = species number sitting at site idx.
        sigmas: list of int
            sigma[atom_index] is an integer between 0 and M-1, where M is the
            total number of different species which may substitute an atom
            in this position.
        """
        idx_subs = self.get_idx_subs()
        tags = self.get_tags() # tags[atom_index] = site_type

        decoration = self.get_atomic_numbers()
        sigmas = np.zeros(len(tags),dtype=np.int8)
        for tag, nsub in nsubs.items():
            #list all atom indices with the given tag or site_type
            sub_idxs = np.where(tags==tag)[0]

            #select nsub such indices for each substitutional
            #atom of given site tag
            sub_list = np.asarray([])
            for i,n in enumerate(nsub):
                sub_idxs = np.setdiff1d(sub_idxs,sub_list)
                sub_list = np.random.choice(sub_idxs,n,replace=False)

                for atom_index in sub_list:
                    decoration[atom_index] = idx_subs[tag][i+1]
                    sigmas[atom_index] = i+1
        return decoration, sigmas


    def enumerate_decorations(self, npoints=None, radii=None):
        from ase.db.jsondb import JSONDatabase
        from ase.neighborlist import NeighborList
        from subprocess import call

        atoms = self.get_pristine()
        natoms = len(atoms)
        sites = self.get_sites()
        call(["rm","-f","neighbors.json"])
        atoms_db = JSONDatabase(filename="neighbors.json") # For visualization

        rmax = np.full(len(atoms),np.amax(radii)/2.0)
        nl = NeighborList(rmax, self_interaction=True, bothways=True, skin=0.0)
        nl.build(atoms)
        distances = atoms.get_all_distances(mic=True)

        for id1 in range(natoms):
            neigs = atoms.copy()
            indices, offsets = nl.get_neighbors(id1)

            for i, offset in zip(indices,offsets):
                #pos.append(atoms.positions[i] + np.dot(offset, atoms.get_cell()))
                #pos.append(atoms.positions[i])
                #chem_num.append(atoms.numbers[i])
                spi = len(sites[i])-1
                neigs[i].number = sites[i][spi]
                neigs[i].position = atoms.positions[i] + np.dot(offset, atoms.get_cell())


            #neigs = Atoms(numbers=chem_num,positions=pos,cell=self.get_cell(),pbc=self.get_pbc())

            atoms_db.write(neigs)
