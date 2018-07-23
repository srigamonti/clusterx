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
        self.ais = np.array(atom_indexes)
        self.ans = np.array(atom_numbers)
        self.npoints = len(atom_numbers)
        self.positions_cartesian = None
        self.radius = None
        if super_cell is not None:
            # Set alphas, site_type, and positions
            self.alphas = np.zeros(len(atom_indexes))
            self.site_type = np.zeros(len(atom_indexes))
            sites = super_cell.get_sites()
            idx_subs = super_cell.get_idx_subs()
            tags = super_cell.get_tags()

            self.positions_cartesian = np.zeros((self.npoints,3))
            for ip, idx in enumerate(atom_indexes):
                self.positions_cartesian[ip] = super_cell.get_positions(wrap=True)[idx]
                self.site_type[ip] = tags[idx]
                self.alphas[ip] = np.argwhere(sites[idx] == self.ans[ip])

            # Set radius
            r = 0.0
            if self.npoints > 1:
                for i1, idx1 in enumerate(self.ais):
                    for idx2 in self.ais[i1+1:]:
                        d = super_cell.get_distance(idx1,idx2,mic=False,vector=False)
                        if r < d:
                            r = d
            self.radius = r


    def __lt__(self,other):
        if self.npoints == other.npoints:
            return self.radius < other.radius
        else:
            return self.npoints < other.npoints

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
