from collections import Counter

class Cluster():
    def __new__(cls, atom_indexes, atom_numbers):
        for iai, ai in enumerate(atom_indexes):
            for _iai, _ai in enumerate(atom_indexes):
                if ai == _ai and atom_numbers[iai] != atom_numbers[_iai]:
                    raise ValueError("Cluster may not have different species on the same site.")

        if len(atom_indexes) != len(atom_numbers):
            raise ValueError("Initialization error, number of sites in cluster different from number of species.")
                    
        cl = super(Cluster,cls).__new__(cls)
        cl.__init__(atom_indexes, atom_numbers)
        return cl
    
    def __init__(self, atom_indexes, atom_numbers):
        self.ais = atom_indexes
        self.ans = atom_numbers
        self.npoints = len(self.ais)
        
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

    
