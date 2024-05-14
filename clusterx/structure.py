# Copyright (c) 2015-2021, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from clusterx.super_cell import SuperCell
from ase import Atoms
import numpy as np
from ase.data import atomic_numbers as an

class Structure(SuperCell):
    """Structure class

    A ``Structure`` object is a :class:`SuperCell <clusterx.super_cell.SuperCell>` object augmented by an array of species numbers (or symbols). This array
    indicates a particular configuration of the :class:`SuperCell <clusterx.super_cell.SuperCell>` (a *"decoration"*), in a way which is compatible with
    the sublattices defined in the :class:`SuperCell <clusterx.super_cell.SuperCell>` object. 

    The ``Structure`` class inherits from the :class:`SuperCell <clusterx.super_cell.SuperCell>` class. Therefore, all methods available to the
    :class:`SuperCell <clusterx.super_cell.SuperCell>` class (and to the :class:`ParentLattice <clusterx.parent_lattice.ParentLattice>` 
    class, from which :class:`SuperCell <clusterx.super_cell.SuperCell>` inherits) are available
    to the ``Structure`` class. Therefore, look at the documentation of the classes :class:`SuperCell <clusterx.super_cell.SuperCell>` 
    and :class:`ParentLattice <clusterx.parent_lattice.ParentLattice>` for additional methods for the ``Structure`` class.
   
    **Parameters:**

    ``super_cell``: SuperCell object
         :class:`SuperCell <clusterx.super_cell.SuperCell>` object.
    ``decoration``: list of int (default: ``None``)
        Atomic numbers of the structure. Overriden by ``sigmas`` and ``atomic_symbols`` if not ``None``.
    ``decoration_symbols``: list of strings (default: ``None``)
        Atomic symbols of the structure. Overriden by ``sigmas`` if not ``None``.
    ``sigmas``: list of int (default: ``None``)
        Every site in a supercell is represented by an array of the species that
        can occupy the site. Thus, *taking as reference these arrays*, a possible
        representation of a *decoration* is by indicating the ordinal number of the
        corresponding species. For instance, if the "sites"-based representation of the SuperCell
        is ``{0: [10,11], 1: [25], 2: [12,25]}`` (*), then equivalent structures are obtained with
        ``decoration = [10,25,25]`` or ``decoration_symbols = ["Ne","Mn","Mn"]`` or ``sigmas = [0,0,1]``. 
        If not ``None``, ``sigmas`` overrides ``decoration`` and ``decoration_symbols``.
    ``mc``: Boolean (default: ``False``)
        whether the initialization of the ``Structure`` object is in the context of a Monte Carlo (MC) run. Setting it to ``True``
        affects the behavior of the method :py:meth:`Structure.swap_random_binary() <clusterx.structure.Structure.swap_random_binary>`

        (*) This means, crystal site ``0`` can host any of the species ``10`` (Neon) or ``11`` (Sodium), etc. 
        See, e.g., :py:meth:`ParentLattice.get_sites() <clusterx.parent_lattice.ParentLattice.get_sites()>`.

    .. todo::
        * check input. If a structure is initialized with a non-allowed substitutional species, and error should be raised.

    **Methods:**
    """
    def __init__(self, super_cell, decoration = None, decoration_symbols=None, sigmas = None, mc = False):
        self.scell = super_cell
        self.sites = super_cell.get_sites()
        self._pbc = super_cell.get_pbc()

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
                
        self.precision_positions = 5

    def _get_roundpos(self):
        return np.around(self.get_positions(),decimals=self.precision_positions)

    def _get_roundcell(self):
        return np.around(self.get_cell(),decimals=self.precision_positions)

    def __repr__(self):
        tokens = []

        tokens.append('pbc={0}'.format(self._pbc))
        
        cell = self._get_roundcell()
        tokens.append('cell={0}'.format(cell.tolist()))

        pos = self._get_roundpos()
        tokens.append('pos_car={0}'.format(pos.tolist()))

        decor = self.decor
        tokens.append('at_nrs={0}'.format(decor))
        
        return '{0}({1})'.format(self.__class__.__name__, ', '.join(tokens))
        
    def __hash__(self):
        #sp = self._get_roundpos()
        #fingerprint = str(list(zip(sp,self.decor)))
        return hash(self.__repr__())
    
    def __eq__(self, other):
        spbc = self.pbc
        opbc = other.pbc
        scell = self._get_roundcell()
        ocell = other._get_roundcell()
        spos = self._get_roundpos()
        opos = other._get_roundpos()
        
        return (self.__class__ == other.__class__ and
                np.array_equal(spbc,opbc) and
                np.array_equal(scell,ocell) and
                np.array_equal(spos,opos) and
                np.array_equal(self.decor,other.decor))

    def set_calculator(self, calculator):
        """Set Calculator object for structure.
        """
        #super(Structure,self).set_calculator(calculator)
        self._calc = calculator
        self.atoms.set_calculator(calculator)

    def get_positions(self, wrap=False, **wrap_kw):
        return super(Structure, self).get_positions(wrap, **wrap_kw)

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

    def serialize(self, fmt="json", filepath="structure.json", fname=None):
        """Serialize structure object

        Wrapper for ASEs write method.

        **Parameters:**

        ``fmt``: string (default: ``json``)
            Indicate the file format of the output file. All the formats
            accepted by ``ase.io.write()`` method are valid (see corresponding
            documentation in https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.write).

        ``filepath``: string (default: ``structure.json``)
            DEPRECATED, use filepath instead file name (may includ absolute or relative path).

        ``fname``: string (default: ``None``)
            DEPRECATED, use filepath instead. File name (may includ absolute or relative path).
        """
        from ase.io import write

        if fname is not None:
            filepath = fname
        
        write(filepath,images=self.atoms,format=fmt)

        self._fname = filepath

    def swap_random_binary(self, site_type, sigma_swap = [0,1]):
        """Swap two randomly selected atoms in given sub-lattice.

        **Parameters:**

        ``site_type``: integer (required)
            Indicate index of sub-lattice where atoms are to be swapped.

        ``sigma_swap``: two-component integer array (default: ``[0,1]``)
            Indicate which atomic species (represented by sigma variables) in the sublattice
            are swapped. E.g., in the case of a binary, this can only be ``[0,1]``, while 
            for a ternary, this can be ``[0,1]``, ``[0,2]``, ``[1,2]`` (and, obviously, the exchanged ones, e.g. ``[1,0]``).

        **Return:**
        """
        if self._mc == True:
            rind1 = np.random.choice(range(self._comps[site_type][sigma_swap[0]]))
            rind2 = np.random.choice(range(self._comps[site_type][sigma_swap[1]]))
            ridx1 = self._idxs[site_type][sigma_swap[0]][rind1]
            ridx2 = self._idxs[site_type][sigma_swap[1]][rind2]
            rindices = [sigma_swap,[rind1,rind2]]
            self.swap(ridx1, ridx2, site_type = site_type,rindices = rindices)

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
        """Swap two randomly selected atoms in randomly selected sub-lattice.

        First, a sublattice from the site_types array is picked at random. Second,
        a pair of species from the selected sublattice are swapped.

        Structure object is modified by the swap. The swapped Structure object is also returned.

        See also :py:meth:`Structure.swap_random_binary() <clusterx.structure.Structure.swap_random_binary>`

        **Parameters:**

        ``site_types``: integer array (required)
            Indicate indices of sub-lattices to be considered in the random sub-lattice selection.
        """
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

            #a = sorted(self._idxs[site_type][rindices[0][0]])
            #b = sorted(self._idxs[site_type][rindices[0][1]])


    def update_decoration(self, decoration):
        """Update decoration of the structure object

        **Parameters:**

        ``decoration``:
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
