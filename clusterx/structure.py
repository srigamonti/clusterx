# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from clusterx.super_cell import SuperCell
from ase import Atoms
import numpy as np
from ase.data import atomic_numbers as an

class Structure(SuperCell):
    """Structure class

    The structure class inherits from the SuperCell class. A structure is
    a super cell with a unique decoration on the sublattices that can be
    substituted.

    **Parameters:**

    ``super_cell``: SuperCell object
        Super cell.
    ``decoration``: list of int
        Atomic numbers of the structure. Overriden by ``sigma`` and ``atomic_symbols``.
    ``decoration_symbols``: list of strings
        Atomic symbols of the structure. Overriden by ``sigma``.
    ``sigmas``: list of int
        Every site in a supercell is represented by an array of the species that
        can occupy the site. Thus, *taking as reference these arrays*, a possible
        representation of a decoration is by indicating the ordinal number of the
        corresponding species. For instance, if the "sites" representation of the SuperCell
        is ``[[10,11],[25],[12,25]]``, then same decoration can be represented
        with the ``decoration`` parameter as ``[10,25,25]`` and with the ``sigmas``
        parameter as ``[0,0,1]``. If not ``None``, ``sigmas`` overrides ``decoration`` and ``decoration_symbols``.

    .. todo::
        * check input. If a structure is initialized with a non-allowed substitutional species, and error should be raised.

    **Methods:**
    """
    def __init__(self, super_cell, decoration = None, decoration_symbols=None, sigmas = None, mc = False):
        self.scell = super_cell
        self.sites = super_cell.get_sites()

        if sigmas is None:
            if decoration_symbols is None:
                self.decor = decoration
            else:
                self.decor = []
                for s in decoration_symbols:
                    self.decor.append(an[s])
                decoration = self.decor

            self.sigmas = np.zeros(len(decoration),dtype=np.int8)
            self.ems = np.zeros(len(decoration),dtype=np.int8)

            for idx, species in enumerate(decoration):
                if species not in self.sites[idx]:
                    raise AttributeError("CELL: decoration not compatible with parent lattice definition.")

            for idx, species in enumerate(decoration):
                self.sigmas[idx] = np.argwhere(self.sites[idx] == species)
                self.ems[idx] = len(self.sites[idx])
        else:
            self.decor = np.zeros(len(sigmas),dtype=np.int8)
            self.ems = np.zeros(len(sigmas),dtype=np.int8)
            self.sigmas = sigmas
            for idx, sigma in enumerate(sigmas):
                self.decor[idx] = self.sites[idx][sigma]
                self.ems[idx] = len(self.sites[idx])


        self.atoms = Atoms(numbers = self.decor, positions = super_cell.get_positions(), tags = super_cell.get_tags(), cell = super_cell.get_cell(),pbc = super_cell.get_pbc())
        super(Structure,self).__init__(super_cell.get_parent_lattice(),super_cell.get_transformation())
        #self.set_atomic_numbers(self.decor)

        self._mc = mc

        if self._mc:
            self._idxs = {}
            self._comps = {}

            tags=self.get_tags()
            sublats = self.get_idx_subs()
            for key in sublats.keys():
                idxs=[]
                lens=[]
                for i,el in enumerate(sublats[key]):
                    idx =  [index for index in range(len(self.decor)) if self.sigmas[index] == i and tags[index] == key]
                    l = len(idx)
                    idxs.append(idx)
                    lens.append(l)
                self._idxs.update({key:idxs})
                self._comps.update({key:lens})

    def set_calculator(self, calculator):
        #super(Structure,self).set_calculator(calculator)
        self._calc = calculator
        self.atoms.set_calculator(calculator)

    def get_sigmas(self):
        """Return decoration array in terms of sigma variables.
        """
        return self.sigmas

    def get_supercell(self):
        """Return SuperCell member of the Structure
        """
        return self.scell

    def get_atoms(self):
        """Get Atoms object corresponding to the Structure object
        """
        return self.atoms

    def get_atomic_numbers(self):
        """Get decoration array
        """
        return self.atoms.get_atomic_numbers()

    def get_chemical_symbols(self):
        """Get decoration array
        """
        return self.atoms.get_chemical_symbols()

    """
    def get_potential_energy(self):
        from clusterx.utils import remove_vacancies
        _atoms = remove_vacancies(self.get_atoms())
        return _atoms.get_potential_energy()
    """

    def serialize(self, fmt="json", fname="structure.json"):
        """Serialize structure object

        Wrapper for ASEs write method.

        **Parameters:**

        ``fmt``: string (default: ``json``)
            Indicate the file format of the output file. All the formats
            accepted by ``ase.io.write()`` method are valid (see corresponding
            documentation in https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.write).

        ``fname``: string (default: ``structure.json``)
            file name (may includ absolute or relative path).
        """
        from ase.io import write

        write(fname,images=self.atoms,format=fmt)

        self._fname = fname

    def swap_random_binary(self, site_type, sigma_swap = [0,1]):

        if self._mc == True:
            rind1 = np.random.choice(range(self._comps[site_type][sigma_swap[0]]))
            rind2 = np.random.choice(range(self._comps[site_type][sigma_swap[1]]))
            ridx1 = self._idxs[site_type][sigma_swap[0]][rind1]
            ridx2 = self._idxs[site_type][sigma_swap[1]][rind2]
            rindices = [sigma_swap,[rind1,rind2]]
            self.swap(ridx1, ridx2, site_type = site_type,rindices = rindices)
            #self.swap(ridx1, ridx2, site_type = site_type,rindices = rindices)

            return ridx1,ridx2,site_type,rindices
        else:
            tags=self.get_tags()
            idx1 = [index for index in range(len(self.decor)) if self.sigmas[index] == sigma_swap[0] and tags[index] == site_type]
            idx2 = [index for index in range(len(self.decor)) if self.sigmas[index] == sigma_swap[1] and tags[index] == site_type]
            ridx1 = np.random.choice(idx1)
            ridx2 = np.random.choice(idx2)
            self.swap(ridx1,ridx2)
            
            return ridx1,ridx2

    def swap_random(self, site_types):
        if len(site_types) == 1:
            site_type = site_types[0]
        else:
            site_type = np.random.choice(site_types)

        len_subs = len(self.idx_subs[site_type])
        if len_subs > 2:
            sigma_swap = np.sort(np.random.choice(np.arange(len_subs), 2, replace=False))
        else:
            sigma_swap = np.arange(len_subs)

        return self.swap_random_binary(site_type, sigma_swap = sigma_swap)

    def swap(self, ridx1, ridx2, site_type = None, rindices = None):
        sigma1=self.sigmas[ridx1]
        sigma2=self.sigmas[ridx2]

        self.sigmas[ridx1] = sigma2
        self.sigmas[ridx2] = sigma1

        self.decor[ridx1] = self.sites[ridx1][sigma2]
        self.decor[ridx2] = self.sites[ridx2][sigma1]
        self.atoms.set_atomic_numbers(self.decor)

        if site_type is not None:
            self._idxs[site_type][rindices[0][0]][rindices[1][0]] = ridx2
            self._idxs[site_type][rindices[0][1]][rindices[1][1]] = ridx1
            
            #self._idxs[site_type][rindices[0][0]].sort()
            #self._idxs[site_type][rindices[0][1]].sort()


    def update_decoration(self, decoration):
        """Update decoration of the structure object
        """
        self.decor = decoration
        self.sigmas = np.zeros(len(decoration),dtype=np.int8)
        for idx, species in enumerate(decoration):
            self.sigmas[idx] = np.argwhere(self.sites[idx] == species)

        self.atoms.set_atomic_numbers(self.decor)

    def get_fractional_concentrations(self):
        """Get fractional concentration of each species on each sublattice

        This function returns a dictionary. The keys of the dictionary, denoted
        with :math:`t`, are the site types that admit substitution (i.e., those returned
        by ``Structure.get_substitutional_tags()``). The values of the dictionary
        are arrays of float, denoted with vectors :math:`\mathbf{c}_t`.
        The coordinates of :math:`\mathbf{c}_t` (:math:`c_{\sigma t}`) are equal to
        :math:`n_{\sigma t}/n_t`, where :math:`\sigma \in [ 0,m-1]`,
        :math:`n_{\sigma t}` is the number of atoms of species :math:`\sigma`
        occupying sites of type :math:`t`, and :math:`n_t` is the total number
        of sites of type :math:`t`. The coordinates of :math:`\mathbf{c}_t` sum
        up to :math:`1`.
        """

        #scell = self.get_supercell()
        #plat = scell.get_parent_lattice()

        concentration = {}

        substutitional_site_types = self.get_substitutional_tags()
        idx_subs = self.get_idx_subs()
        nsites_per_type = self.get_nsites_per_type()
        #sigmas = self.get_sigmas()
        numbers = self.get_atomic_numbers()

        for site_type in substutitional_site_types:
            concentration[site_type] = []
            nsites = nsites_per_type[site_type]
            atom_indices = self.get_atom_indices_for_site_type(site_type)
            for spnr in idx_subs[site_type]:
                n = np.array(numbers[atom_indices]).tolist().count(spnr)
                concentration[site_type].append(n/nsites)

        return concentration
