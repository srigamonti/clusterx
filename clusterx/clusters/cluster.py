# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

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

    def __init__(self, atom_indexes, atom_numbers, super_cell=None, distances=None):

        if len(atom_indexes) != len(atom_numbers):
            raise ValueError("Initialization error, number of sites in cluster different from number of species.")

        if len(atom_indexes)!=0:
            try:
                self.ais,self.ans = list(zip(*sorted(zip(np.array(atom_indexes),np.array(atom_numbers)))))
            except:
                raise ValueError("Cluster initialization failed")

        else:
            self.ais = np.array(atom_indexes, dtype=int)
            self.ans = np.array(atom_numbers)

        self.npoints = len(atom_numbers)
        self.positions_cartesian = None
        self.alphas = None
        #self.positions_scaled = None
        self.radius = None
        if super_cell is not None:
            # Set alphas, site_type, and positions
            self.alphas = np.zeros(len(atom_indexes),dtype=int)
            self.site_type = np.zeros(len(atom_indexes),dtype=int)
            sites = super_cell.get_sites()
            idx_subs = super_cell.get_idx_subs()
            tags = super_cell.get_tags()

            self.positions_cartesian = np.zeros((self.npoints,3))
            #self.positions_scaled = np.zeros((self.npoints,3))
            for ip, idx in enumerate(atom_indexes):
                #self.positions_cartesian[ip] = super_cell.get_positions(wrap=True)[idx]
                self.positions_cartesian[ip] = super_cell.get_positions()[idx]
                #self.positions_scaled[ip] = super_cell.get_scaled_positions(wrap=True)[idx]
                self.site_type[ip] = tags[idx]
                self.alphas[ip] = np.argwhere(sites[idx] == self.ans[ip])

            """
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
            
            # Set radius
            # FIX this! Radius cannot be based on distance in supercell, as it will fail for small supercells and large clusters wrapped into it.
            r = 0.0
            if self.npoints > 1:
                if distances is not None:
                    for i1, idx1 in enumerate(self.ais):
                        for idx2 in self.ais[i1+1:]:
                            d = distances[idx1,idx2]
                            if r < d:
                                r = d
                else:
                    for i1 in range(self.npoints-1):
                        for i2 in range(i1+1, self.npoints):
                            d = np.linalg.norm(self.positions_cartesian[i1]-self.positions_cartesian[i2])
                            if r < d:
                                r = d
            self.radius = r

        self.myhash = self.__hash__()

    """
    def __lt__(self,other):
        if self.npoints == other.npoints:
            return self.radius < other.radius
        else:
            return self.npoints < other.npoints
    """

    def get_alphas(self):
        """Return labels of point basis-functions of cluster
        """
        return self.alphas

    def _compute_radius(self,distances):
        r = 0.0
        if self.npoints > 1:
            for i1, idx1 in enumerate(self.ais):
                for idx2 in self.ais[i1+1:]:
                    d = distances[idx1,idx2]
                    if r < d:
                        r = d
        self.radius = r

    def get_radius(self,distances=None):
        """Return cluster radius
        The radius of a cluster is the maximum distance between any pair of its points.
        """
        if self.radius is not None:
            return self.radius
        elif distances is not None:
            self._compute_radius(distances)
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

    def __repr__(self):
        return "Cluster["+str(list(zip(self.ais,self.ans)))+"]"
        
    def __hash__(self):
        return hash(str(list(zip(self.ais,self.ans))))
        
    def __eq__(self, other):
        return self.myhash == other.myhash

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
        """
        from clusterx.symmetry import get_scaled_positions, get_internal_translations, wrap_scaled_positions

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
