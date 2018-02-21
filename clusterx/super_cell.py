from ase import Atoms
import clusterx as c
import numpy as np
import sys
from ase.visualize import view
from ase.build import make_supercell
from clusterx.parent_lattice import ParentLattice

class SuperCell(ParentLattice):
    """
    Builds a super cell

    A SuperCell object acts as a blue-print for the creation of structures
    with arbitrary decorations.
    """
    
    def __init__(self, parent_lattice, p):
        self._plat = parent_lattice
        self._p = p
        #super(SuperCell, self).__init__()
        if (parent_lattice.get_pbc()==(1,1,0)).all():
            self._p[0,2] = 0
            self._p[1,2] = 0
            self._p[2,0] = 0
            self._p[2,1] = 0
            self._p[2,2] = 1
            self.pbc[2] = 0

        prist = make_supercell(parent_lattice.get_atoms(),p)
        subs = [make_supercell(atoms,p) for atoms in parent_lattice.get_substitutions()]
        #ParentLattice.__init__(self, atoms = prist, substitutions = subs )
        super(SuperCell,self).__init__(atoms = prist, substitutions = subs )
        self._natoms = len(self)

        self.set_pbc(self.pbc)

    def get_parent_lattice(self):
        """
        Return the parent lattice object which defines the supercell.
        """
        return self._plat

    def get_transormation(self):
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
    

    def gen_random(self,nsubs={}):
        """
        Generate a random decoration of the super cell with given number of substitutions.

        The nsubs argument must have the format:
        ...
        """
        idx_subs = self.get_idx_subs()
        tags = self.get_tags()

        rndstr = SuperCell(self._plat,self._p)
        
        for tag, nsub in nsubs.items():
            #list all atom indices with the given tag
            sub_idxs = np.where(tags==tag)[0]

            #select nsub such indices for each substitutional
            #atom of given site tag
            sub_list = np.asarray([])
            for i,n in enumerate(nsub):
                sub_idxs = np.setdiff1d(sub_idxs,sub_list)
                sub_list = np.random.choice(sub_idxs,n,replace=False)

                for atom_index in sub_list:
                    rndstr[atom_index].number = idx_subs[tag][i+1]

        return rndstr