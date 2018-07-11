from clusterx.super_cell import SuperCell
from ase import Atoms
import numpy as np

class Structure(SuperCell):
    """Structure class

    The structure class inherits from the SuperCell class. A structure is
    a super cell with a unique decoration on the sublattices that can be
    substituted.

    Parameters:

    super_cell: SuperCell object
        Super cell.
    decoration: list of int
        Atomic numbers of the structure.
    sigmas: list of int
        

        Since a structure is represented by a decorated supercell,
    """
    def __init__(self, super_cell, decoration = None, sigmas = None):
        self.scell = super_cell
        if sigmas is None:
            sites = super_cell.get_sites()
            self.decor = decoration
            self.sigmas = np.zeros(len(decoration),dtype=np.int8)
            self.ems = np.zeros(len(decoration),dtype=np.int8)
            for idx, species in enumerate(decoration):
                self.sigmas[idx] = np.argwhere(sites[idx] == species)
                self.ems[idx] = len(sites[idx])
        else:
            self.decor = np.zeros(len(sigmas),dtype=np.int8)
            self.ems = np.zeros(len(sigmas),dtype=np.int8)
            sites = super_cell.get_sites()
            self.sigmas = sigmas
            for idx, sigma in enumerate(sigmas):
                self.decor[idx] = sites[idx][sigma]
                self.ems[idx] = len(sites[idx])

        super(Structure,self).__init__(super_cell.get_parent_lattice(),super_cell.get_transformation())
        self.atoms = Atoms(numbers = self.decor, positions = super_cell.get_positions(), tags = super_cell.get_tags(), cell = super_cell.get_cell(),pbc = super_cell.get_pbc())
        self.set_atomic_numbers(self.decor)

    def get_supercell(self):
        return self.scell

    def get_atoms(self):
        return self.atoms

    def serialize(self, fmt="json", tmp=False, fname=None):
        from ase.io import write

        if fname is None:
            fname = "structure.json"

        write(fname,images=self,format=fmt)

        self._fname = fname
