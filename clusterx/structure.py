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

    **Methods:**
    """
    def __init__(self, super_cell, decoration = None, decoration_symbols=None, sigmas = None):
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
                self.sigmas[idx] = np.argwhere(self.sites[idx] == species)
                self.ems[idx] = len(self.sites[idx])
        else:
            self.decor = np.zeros(len(sigmas),dtype=np.int8)
            self.ems = np.zeros(len(sigmas),dtype=np.int8)
            self.sigmas = sigmas
            for idx, sigma in enumerate(sigmas):
                self.decor[idx] = self.sites[idx][sigma]
                self.ems[idx] = len(self.sites[idx])

        super(Structure,self).__init__(super_cell.get_parent_lattice(),super_cell.get_transformation())
        self.atoms = Atoms(numbers = self.decor, positions = super_cell.get_positions(), tags = super_cell.get_tags(), cell = super_cell.get_cell(),pbc = super_cell.get_pbc())
        #self.set_atomic_numbers(self.decor)

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

    def serialize(self, fmt="json", tmp=False, fname=None):
        from ase.io import write

        if fname is None:
            fname = "structure.json"

        write(fname,images=self.atoms,format=fmt) 

        self._fname = fname

    def swap_random_binary(self, site_type, sigma_swap = [0,1]):
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
    
    def swap(self, ridx1, ridx2):
        sigma1=self.sigmas[ridx1]
        sigma2=self.sigmas[ridx2]

        self.sigmas[ridx1] = sigma2
        self.sigmas[ridx2] = sigma1
        self.decor[ridx1] = self.sites[ridx1][sigma2]
        self.decor[ridx2] = self.sites[ridx2][sigma1]

    def update_decoration(self, decoration):
        """Update decoration of the structure object
        """
        self.decor = decoration
        self.sigmas = np.zeros(len(decoration),dtype=np.int8)
        for idx, species in enumerate(decoration):
            self.sigmas[idx] = np.argwhere(self.sites[idx] == species)

        self.atoms.set_atomic_numbers(self.decor)

    def get_concentration(self):
        """Get concentration of each species on each sublattice
        """

        #scell = self.get_supercell()
        #plat = scell.get_parent_lattice()

        substutitional_site_types = self.get_substitutional_tags()
        nsites_per_type = self.get_nsites_per_type()
        print(nsites_per_type)
        #for site_type in substutitional_site_types:
