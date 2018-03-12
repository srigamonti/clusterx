from collections import Counter
from clusterx.symmetry import get_scaled_positions
import numpy as np

class Cluster():
    def __new__(cls, atom_indexes, atom_numbers, super_cell=None):
        for iai, ai in enumerate(atom_indexes):
            for _iai, _ai in enumerate(atom_indexes):
                if ai == _ai and atom_numbers[iai] != atom_numbers[_iai]:
                    raise ValueError("Cluster may not have different species on the same site.")

        if len(atom_indexes) != len(atom_numbers):
            raise ValueError("Initialization error, number of sites in cluster different from number of species.")
                    
        cl = super(Cluster,cls).__new__(cls)
        cl.__init__(atom_indexes, atom_numbers, super_cell)
        return cl
    
    def __init__(self, atom_indexes, atom_numbers, super_cell=None):
        self.ais = atom_indexes
        self.ans = atom_numbers
        self.npoints = len(atom_numbers)
        self.positions_cartesian = None
        if super_cell is not None:
            self.positions_cartesian = np.zeros((self.npoints,3)) 
            for ip, idx in enumerate(atom_indexes):
                self.positions_cartesian[ip] = super_cell.get_positions(wrap=True)[idx]
            

        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            ais = self.ais
            ans = self.ans
            oais = other.ais
            oans = other.ans
            npoints = self.npoints
            
            if len(ais) != len(oais):
                return False

            if Counter(ais) != Counter(oais):
                return False
            
            for i in range(npoints):
                for j in range(npoints):
                    if ais[i] == oais[j] and ans[i] != oans[j]:
                        return False

            return True
        
        else:
            return False

    def __len__(self):
        return len(self.ais)
        
    def get_idxs(self):
        return self.ais
    
    def get_nrs(self):
        return self.ans
    
    def set_idxs(self, atom_indexes):
        self.ais = atom_indexes
    
    def set_nrs(self, atom_numbers):
        self.ans = atom_numbers

        
    def get_positions(self):
        return self.positions_cartesian
