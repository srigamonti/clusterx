from clusterx.super_cell import SuperCell
from ase import Atoms
import numpy as np
import random

class Structure(SuperCell):
    """Structure class

    The structure class inherits from the SuperCell class. A structure is
    a super cell with a unique decoration on the sublattices that can be
    substituted.

    **Parameters:**

    ``super_cell``: SuperCell object
        Super cell.
    ``decoration``: list of int
        Atomic numbers of the structure. Overriden by ``sigma``.
    ``sigmas``: list of int
        Every site in a supercell is represented by an array of the species that
        can occupy the site. Thus, *taking as reference these arrays*, a possible
        representation of a decoration is by indicating the ordinal number of the
        corresponding species. For instance, if the "sites" representation of the SuperCell
        is ``[[10,11],[25],[12,25]]``, then same decoration can be represented
        with the ``decoration`` parameter as ``[10,25,25]`` and with the ``sigmas``
        parameter as ``[0,0,1]``. If not ``None``, ``sigmas`` overrides ``decoration``

    **Methods:**
    """
    def __init__(self, super_cell, decoration = None, sigmas = None):
        self.scell = super_cell
        self.sites = super_cell.get_sites()
        if sigmas is None:
            self.decor = decoration
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

    def serialize(self, fmt="json", tmp=False, fname=None):
        from ase.io import write

        if fname is None:
            fname = "structure.json"

        write(fname,images=self.atoms,format=fmt) 

        self._fname = fname

    def swap_random_binary(self, site_type):
        tags=self.get_tags()
        idx1 = [index for index in range(len(self.decor)) if self.sigmas[index] == 0 and tags[index] == site_type]
        idx2 = [index for index in range(len(self.decor)) if self.sigmas[index] == 1 and tags[index] == site_type]
        ridx1 = random.choice(idx1)
        ridx2 = random.choice(idx2)

        self.swap(site_type,ridx1,ridx2)

        return ridx1,ridx2

    def swap(self, site_type, ridx1, ridx2):
        tags=self.get_tags()
        sigma1=self.sigmas[ridx1]
        sigma2=self.sigmas[ridx2]

        self.sigmas[ridx1] = sigma2
        self.sigmas[ridx2] = sigma1
        self.decor[ridx1] = self.sites[ridx1][sigma2]
        self.decor[ridx2] = self.sites[ridx2][sigma1]
