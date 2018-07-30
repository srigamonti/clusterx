from collections import Counter
from clusterx.symmetry import get_scaled_positions
import numpy as np

class Cluster():
    """Cluster class

    Objects of this class represent clusters. The intialization of these objects
    relies on the definition of a supercell. Once initialized, some of the
    attributes still refer to the supercell (e.g. atom indexes), while other
    (e.g. atomic positions) are independent of the supercell definition.

    **Parameters:**

    ``atom_indexes``: array of integer
        The atomic indices of the base super cell used for the initialization of
        the cluster.
    ``atom_numbers``: array of integer
        In order to represent the selection of single-point basis functions for
        the cluster, atomic numbers are used in the initialization. The possible
        values to select from, correspond to the possible substitutional species
        that can occupy the site in the corresponding atom index in the array
        ``atom_indexes``.
    ``super_cell``: SuperCell object (default: ``None``)
        If not given on initialization, only basic attributes (e.g. atom indices
        and numbers) are set. Otherwise, a complete definition is set up, containing
        atom coordinates, basis function indices, site types (which indicate which
        sublattices a cluster point belongs to), etc.
    ``distances``: matrix of float (default: ``None``)
        to speed up the calculation of the cluster radius on initialization, the
        distances between all pairs of atoms in the supercell may be passed in
        this argument.


    **Methods:**
    """
    def __new__(cls, atom_indexes, atom_numbers, super_cell=None, distances=None):
        for iai, ai in enumerate(atom_indexes):
            for _iai, _ai in enumerate(atom_indexes):
                if ai == _ai and atom_numbers[iai] != atom_numbers[_iai]:
                    raise ValueError("Cluster may not have different species on the same site.")

        if len(atom_indexes) != len(atom_numbers):
            raise ValueError("Initialization error, number of sites in cluster different from number of species.")

        cl = super(Cluster,cls).__new__(cls)
        cl.__init__(atom_indexes, atom_numbers, super_cell, distances)
        return cl

    def __init__(self, atom_indexes, atom_numbers, super_cell=None, distances=None):
        #self.ais = np.array(atom_indexes)
        #self.ans = np.array(atom_numbers)
        if len(atom_indexes)!=0:
            self.ais, self.ans = (list(t) for t in zip(*sorted(zip(atom_indexes, atom_numbers))))
        else:
            self.ais = np.array(atom_indexes)
            self.ans = np.array(atom_numbers)
        self.npoints = len(atom_numbers)
        self.positions_cartesian = None
        #self.positions_scaled = None
        self.radius = None
        if super_cell is not None:
            # Set alphas, site_type, and positions
            self.alphas = np.zeros(len(atom_indexes))
            self.site_type = np.zeros(len(atom_indexes))
            sites = super_cell.get_sites()
            idx_subs = super_cell.get_idx_subs()
            tags = super_cell.get_tags()

            self.positions_cartesian = np.zeros((self.npoints,3))
            #self.positions_scaled = np.zeros((self.npoints,3))
            for ip, idx in enumerate(atom_indexes):
                self.positions_cartesian[ip] = super_cell.get_positions(wrap=True)[idx]
                #self.positions_scaled[ip] = super_cell.get_scaled_positions(wrap=True)[idx]
                self.site_type[ip] = tags[idx]
                self.alphas[ip] = np.argwhere(sites[idx] == self.ans[ip])

            # Set radius
            r = 0.0
            if self.npoints > 1:
                for i1, idx1 in enumerate(self.ais):
                    for idx2 in self.ais[i1+1:]:
                        if distances is not None:
                            d = distances[idx1,idx2]
                        else:
                            d = super_cell.get_distance(idx1,idx2,mic=False,vector=False)
                        if r < d:
                            r = d
            self.radius = r

    """
    def __lt__(self,other):
        if self.npoints == other.npoints:
            return self.radius < other.radius
        else:
            return self.npoints < other.npoints
    """

    def set_radius(self,distances):
        r = 0.0
        if self.npoints > 1:
            for i1, idx1 in enumerate(self.ais):
                for idx2 in self.ais[i1+1:]:
                    d = distances[idx1,idx2]
                    if r < d:
                        r = d
        self.radius = r

    def get_radius(self,distances=None):
        if self.radius is not None:
            return self.radius
        elif distances is not None:
            self.set_radius(distances)
            return self.radius

    def _get_idxs_norm(self):
        return np.linalg.norm(self.ais)

    def __lt__(self,other):
        if self.npoints == other.npoints and abs(self.radius-other.radius)<1e-5:
            ns = self._get_idxs_norm()
            no = other._get_idxs_norm()
            return ns < no
        elif self.npoints == other.npoints:
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

            ns = len(ais)
            no = len(oais)
            if ns != no:
                return False

            #if Counter(ais) != Counter(oais):
            #    return False

            #if sorted(ais) != sorted(oais):
            #    return False

            for i in range(ns):
                if ais[i] != oais[i] or ans[i] != oans[i]:
                    return False

            #if (ais != oais).any():
            #    return False

            #if (ans != oans).any():
            #    return False

            """
            for i in range(npoints):
                for j in range(npoints):
                    if ais[i] == oais[j] and ans[i] != oans[j]:
                        return False
            """
            return True

        else:
            return False

    def __len__(self):
        return len(self.ais)

    def get_idxs(self):
        """Return array of atom indices referred to the defining supercell.
        """
        return self.ais

    def get_nrs(self):
        """Return array of atom numbers.

        In the context of the clusters as mathematical objects, the atomic numbers
        here must be understood as an alias (i.e. an index), for the single-site
        basis function on each point of the cluster.
        """
        return self.ans

    def get_multiplicity(self, rr, tt, cell, pbc):
        """Calculate cluster multiplicity (Experimental. To get the cluster
        multiplicities, see documentation for ClustersPool class)

        Uses symmetry operations as returned by spglib to find cluster multiplicity.

        **Parameters:**

        ``rr``: array of 3x3 matrices
            rotations of the symmetry operations
        ``tt``: array of 3x1 vectors
            translations of the symmetry operations.
            From spglib doc: The orders of the rotation matrices and the translation vectors
            correspond with each other, e.g. , the second symmetry operation is
            organized by the set of the second rotation matrix and second
            translation vector in the respective arrays.
        ``cell``: 3x3 matrix
            The symmetry operations ``rr`` and ``tt`` refer to scaled coordinates.
            The parameter ``cell`` contains row-wise the corresponding cartesian
            coordinates of the cell vectors.

        **Example:**
        ::
            from clusterx.symmetry import get_spacegroup
            sc_sg, sc_sym = get_spacegroup(parent_lattice) # Scaled to parent_lattice
            m = []
            for cl in clusters_pool.get_cpool():
                m.append(cl.get_multiplicity(sc_sym["rotations"],sc_sym["translations"]))

        """
        from clusterx.symmetry import get_spacegroup, get_scaled_positions, get_internal_translations, wrap_scaled_positions

        orbit = []
        for r,t in zip(rr,tt):
            ts = np.tile(t,(self.npoints,1)).T
            #sp0 = np.dot(self.positions_cartesian,np.linalg.inv(cell))
            sp0 = get_scaled_positions(self.positions_cartesian, cell, pbc=pbc, wrap=True)
            sp1 = np.add(np.dot(r,sp0.T),ts).T
            orbit.append(sp1)

        rorbit = np.around(orbit,5)
        #print("rorbit",rorbit)
        all_pos = []
        for cl in rorbit:
            for sp in cl:
                all_pos.append(sp)

        #print("all_pos",all_pos)
        unique_pos = np.unique(all_pos, axis=0)

        #print("unique_pos",unique_pos)
        key_orbit = []
        for icl,cl in enumerate(rorbit):
            key_vec = []
            for ipos, pos in enumerate(cl):
                for iupos ,upos in enumerate(unique_pos):
                    if (pos == upos).all():
                        key_vec.append(iupos)

            key_orbit.append(sorted(key_vec))

        #print("key_orbit",key_orbit)
        #print("unique clusters",np.unique(key_orbit,axis=0))
        #print(key_orbit)
        #print("ALL POS",all_pos)
        #print("UNIQUE", unique_pos)
        #print(np.around(orbit,5))
        #print(np.unique(np.around(orbit,5),axis=0))
        #return len(np.unique(np.around(orbit,5),axis=0))
        return len(np.unique(key_orbit,axis=0))

    def get_positions(self):
        """Return cartesian coordinates of the cluster points.
        """
        return self.positions_cartesian
