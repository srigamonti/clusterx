# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from ase.db.jsondb import JSONDatabase
from ase.atoms import Atoms
import clusterx as c
from clusterx.clusters.cluster import Cluster
from clusterx.super_cell import SuperCell
from clusterx.parent_lattice import ParentLattice
from clusterx.symmetry import get_scaled_positions, get_internal_translations, wrap_scaled_positions
import os
import sys
import subprocess
import numpy as np
import json
import time
from ase.db import connect
from clusterx.utils import get_cl_idx_sc

class ClustersPool():
    """
    **Clusters pool class**

    **Parameters:**

    ``parent_lattice``: ParentLattice object
        Parent lattice of the system.
    ``npoints``: array of integers
        The number of points of the clusters in the pool. Must be of the same
        dimension as ``radii``.
    ``radii``: array of float
        The maximum radii of the clusters in the pool, for each corresponding
        number of points in ``npoints``. If ``super_cell`` is given (see below),
        any element of the ``radii`` array can be a negative number. In such a
        case, all clusters fitting in the given ``super_cell`` (for the
        corresponding number of points in npoints) are generated.
    ``super_cell``: SuperCell object
        If present, ``radii`` is overriden and the pool will consist of all
        possible simmetrically distinct clusters in the given super cell.
        Periodic boundary conditions of the parent lattice are taken into
        account (thus, e.g., if a supercell of side :math:`a` of a square
        lattice is given, then in the directions of periodicity the clusters
        will not be longer than :math:`a/2`).
        (Note: Experimental feature.  The generated multiplicities may not be
        valid for clusters larger than half the size of the supercell. Use
        ``radii`` to get accurate multiplicities.)
    ``method``: integer (1 or 2)
        If 1, an elimination method is used to build the clusters pool. If 2, an
        incremental method is used instead. Use method 2 only if ``super_cell`` is
        ``None``.
    ``filepath``: string (default: None)
        Overrides all the above. Used to initialize from file. Path of a json
        file containing a serialized ClustersPool object, as generated
        by the ``ClustersPool.serialize()`` method.
    ``json_db_filepath``: string (default: None)
        *DEPRECATED*, use ``filepath`` instead. Overrides all the above. Used to initialize from file. Path of a json
        file containing a serialized ClustersPool object, as generated
        by the ``ClustersPool.serialize()`` method.

    **Examples:**

    The example below, will generate all possible clusters up to 2 points in the
    super cell ``scell``, while more compact clusters up to a radius of 4.1 and
    2.9 for 3 and 4 points, respectively, are generated::

        from ase import Atoms
        from clusterx.parent_lattice import ParentLattice
        from clusterx.super_cell import SuperCell
        from clusterx.clusters.clusters_pool import ClustersPool

        plat = ParentLattice(
            Atoms(cell=np.diag([2,2,5]),positions=[[0,0,0]]),
            site_symbols=[["Cu","Al"]],
            pbc=(1,1,0)
            )

        scell = SuperCell(plat,np.array([(6,0,0),(0,6,0),(0,0,1)]))
        cp = ClustersPool(plat, npoints=[0,1,2,3,4], radii=[0,0,-1,4.1,2.9], super_cell=scell)

        cp.write_clusters_db(db_name="cpool.json")

    The example

    **Methods:**
    """
    def __init__(self, parent_lattice=None, npoints=[], radii=[], super_cell=None, method=0, filepath=None, json_db_filepath=None, db = None):

        if json_db_filepath is not None:
            filepath = json_db_filepath
            
        if filepath is not None:
            db = connect(filepath)

        if db is not None:            
            self.nclusters = db.metadata.get("nclusters",0)
            plat_dict = db.metadata.get("parent_lattice",{})
            self._plat = ParentLattice.plat_from_dict(plat_dict)
            self._multiplicities = np.zeros(self.nclusters, dtype=int)
            self._cpool = []
            self._cpool_dict = {}
            self._npoints =  db.metadata.get("_npoints",[])
            self._radii =  db.metadata.get("_radii",[])
            scell_dict = db.metadata.get("super_cell",{})
            self._cpool_scell = SuperCell.scell_from_dict(scell_dict)
            self._distances = self._cpool_scell.get_all_distances(mic=False)
            self._sdistances = self._cpool_scell.get_substitutional_atoms().get_all_distances(mic=True)

            cl_nrs = db.metadata.get("atom_numbers",[])
            cl_positions = db.metadata.get("atom_positions",[])
            sc_positions = self._cpool_scell.get_positions(wrap=True)
            for i in range(self.nclusters):
                if len(cl_positions[i]) == 0:
                    idxs = []
                else:
                    idxs = get_cl_idx_sc(cl_positions[i],sc_positions, method=1, tol=1e-3)

                self._cpool.append(Cluster(idxs,cl_nrs[i],self._cpool_scell,self._distances))

            self._multiplicities = db.metadata.get("multiplicities",[])

        else:
            self._npoints = np.array(npoints)
            self._radii = np.array(radii,dtype=float)
            self._plat = parent_lattice

            self._multiplicities = []
            self._cpool = []
            self._cpool_dict = {}

            # Determine which supercell will be used to build the clusters pool
            if isinstance(super_cell, SuperCell) or isinstance(super_cell, ParentLattice):
                self._cpool_scell = super_cell
            elif (self._npoints < 2).all():
                self._cpool_scell = SuperCell(parent_lattice,np.diag([1,1,1]))
            elif super_cell is None and len(radii) != 0:
                self._cpool_scell = self.get_containing_supercell()
            else:
                self._cpool_scell = SuperCell(parent_lattice,np.diag([1,1,1]))

            self._distances = self._cpool_scell.get_all_distances(mic=False)
            self._distances_mic_true = self._cpool_scell.get_all_distances(mic=True)
            self._sdistances = self._cpool_scell.get_substitutional_atoms().get_all_distances(mic=True)
            self.set_radii(npoints=npoints,radii=radii)

            if self._npoints.size != 0:
                self.gen_clusters(method=method)

            self.nclusters = len(self._cpool)

        self.sc_sg, self.sc_sym = self._plat.get_sym()
        self.current = 0
        self.high = self.nclusters - 1
        self._cpool_atoms = []

    def set_radii(self,npoints=[],radii=[]):
        eps = 1.0e-8
        scell = self._cpool_scell
        self._radii = np.array(radii,dtype=float)


        dmax = np.around(np.amax(self._sdistances),decimals=3)

        try:
            sd = np.unique(np.around(np.sort(self._distances.flatten()),decimals=3))
            idmax2 = np.argwhere(np.abs(sd - dmax) < eps)
            dmax2 = sd[idmax2+1][0,0]
        except:
            dmax2 = dmax

        # Check if supercell is large enough
        if len(radii) > 0:
            for i in range(len(radii)):
                if radii[i]<0:
                    self._radii[i] = (dmax + dmax2)/2.0
                    #self._radii[i] = dmax + eps

        radii = self._radii

        if len(radii) > 0:
            if np.amax(radii) > dmax2:
                sys.exit("Containing supercell is too small to find clusters pool. Read documentation for ClustersPool class.")
        else:
            radii = np.zeros(len(npoints),dtype=float)
            for i, npts in enumerate(npoints):
                if npts == 0 or npts == 1:
                    radii[i] = 0.0
                else:
                    radii[i] = (dmax + dmax2)/2.0
                    #radii[i] = dmax + eps
            self._radii = radii

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current > self.high:
            raise StopIteration
        else:
            self.current+=1
            return self._cpool[self.current-1]

    def __getitem__(self, index):
        return self._cpool[index]

    def __len__(self):
        return len(self._cpool)

    def sort(self):
        """Sort clusters pool
        """
        self._cpool.sort()

    def add(self, other):
        """Add other ClustersPool object to this one

        - The actual ClustersPool object gets modified
        - self._cpool_scell need be the same for both summands (at the moment not checked by the function)
        """

        m = []
        for icl, cl in enumerate(self):
            m.append(self._multiplicities[icl])
            
            
        for icl, cl in enumerate(other):
           self._cpool.append(cl) 
           m.append(other._multiplicities[icl])

        self._multiplicities = np.array(m,dtype=int)
        self.nclusters = len(self._cpool)
        self.get_cpool_dict()
           
    def get_plat(self):
        """Return parent lattice for the clusters pool
        """
        return self._plat

    def get_multiplicities(self):
        """Return cluster multiplicities

        The multiplicity of a cluster is equal to the number of its
        symmetrically equivalent realizations of it, or equivalently, the size
        of its orbit (without considering parent lattice translations).
        """
        return np.array(self._multiplicities)

    def add_cluster(self, cluster):
        self._cpool.append(cluster)
        self.nclusters = len(self._cpool)

    def get_npoints(self):
        """Return array of number of points of the clusters in the pool.

        The returned integer array has the same length and its elements
        correspond with those of the array ``self.get_radii()``. Thus, the
        pool is defined by having all clusters of ``self.get_npoints()[i]``
        number of points up to a radius of ``self.get_max_radii()[i]``
        """
        return self._npoints

    def get_all_npoints(self):
        """Return array containing the number of points of each cluster in the pool
        """
        npoints = []
        for cl in self._cpool:
            npoints.append(cl.npoints)

        return np.array(npoints)

    def get_max_radii(self):
        """Return array of maximum radii of the clusters in the pool.

        The returned float array has the same length and its elements
        correspond with those of the array ``self.get_npoints()``. Thus, the
        pool is defined by having all clusters of ``self.get_npoints()[i]``
        number of points up to a radius of ``self.get__max_radii()[i]``
        """
        return np.array(self._radii)

    def get_unique_radii(self):
        """
        Return sorted array of all possible unique radii of clusters in the pool.
        """
        unique_radii = []
        for cl in self._cpool:
            unique_radii.append(cl.radius)

        unique_radii = np.unique(np.around(np.array(unique_radii), decimals=5))

        return unique_radii

    def get_all_radii(self):
        """Return array containing the radius of each cluster in the pool
        """
        radii = np.zeros(len(self))
        for icl, cl in enumerate(self._cpool):
            radii[icl] = cl.radius

        return radii

    def get_subpool(self, cluster_indexes):
        """Return a ClustersPool object formed by a subset of the clusters pool

        **Parameters:**

        cluster_indexes: array of integers
            The indexes of the clusters to build the subpool from.
        """
        subpool = ClustersPool(self._plat, super_cell = self._cpool_scell)
        add_mults = (len(self._multiplicities) == len(self))
        for i,cl in enumerate(self._cpool):
            if i in cluster_indexes:
                subpool.add_cluster(cl)
                if add_mults:
                    subpool._multiplicities.append(self._multiplicities[i])
        return subpool

    def get_clusters_sets(self, grouping_strategy = "size", nclmax = 0, set0=[0,0]):
        """Return cluster sets for cluster selection based on CV

        **Parameters:**

        grouping_strategy: string
            Can take the values "size" and "combinations". If "size", a set in the clusters set
            is determined by two parameters, the maximum number of points and the maximum radius.
            Thus, a set is formed by all clusters in the pool whose radius and npoints are smaller
            or equal than the respective maxima for the set.
            When "combinations" is used, all possible sets up to nclmax number of clusters
            are returned.

        nclmax: integer
            The maximum clusters-set size when strategy="combinations" is used.

        set0: [points, radius]
            The clusters-set, which is defined by the two parameters - maximum number of points (points)
            and the maximum radius, [points, radius] -, is included in all combinations,
            when strategy="size+combinations" is used.

        """
        if grouping_strategy == "size":
            from collections import Counter
            clsets = []
            nsets = 0
            unique_radii = self.get_unique_radii()
            for np in self._npoints:
                for r in unique_radii:
                    _clset = []
                    for icl, cl in enumerate(self._cpool):
                        if cl.npoints <= np and cl.radius <= r + 1e-4:
                            _clset.append(icl)

                    # Check whether in _clset there's at least one cluster with np number of points
                    include = False
                    for icl in _clset:
                        if self._cpool[icl].npoints == np:
                            include = True
                            break

                    # Check whether _clset is already in clsets
                    if include:
                        for clset in clsets:
                            if Counter(clset) == Counter(_clset):
                                include = False
                                break

                    if include:
                        clsets.append(_clset)
                        nsets += 1

        if grouping_strategy == "combinations":
            import itertools
            clsets = []

            cl_list=[x for x in range(len(self._cpool))]

            if nclmax<len(cl_list):
                ncl=nclmax
            else:
                ncl=len(cl_list)

            for icl in range(1,ncl+1):
                for x in itertools.combinations(cl_list,icl):
                    clsets.append([el for el in x])

        if grouping_strategy == "size+combinations":
            import itertools

            clsets = []
            nsets = 0
            np = int(set0[0])
            r = float(set0[1])

            _clset0 = []
            _clset1 = []
            for icl, cl in enumerate(self._cpool):
                if cl.npoints <= np and cl.radius <= r + 1e-4:
                    _clset0.append(icl)
                else:
                    _clset1.append(icl)

            if nclmax<len(_clset1):
                ncl=nclmax
            else:
                ncl=len(_clset1)

            for icl1 in range(0,ncl+1):
                for x in itertools.combinations(_clset1,icl1):
                    clset=_clset0+[el for el in x]
                    clset.sort()
                    clsets.append(clset)

        return clsets

    def gen_clusters(self,method=0):
        """Generate pool of clusters

        **Parameters:**

        ``method``: int
        0: Fast, default method. 1: Slow, use only for benchmarking purposes.

        """
        if method == 0:
            self.gen_clusters0()
        if method == 1:
            self.gen_clusters1()

    def gen_clusters0(self, verbosity=0):
        from clusterx.super_cell import SuperCell
        from itertools import product, combinations
        #from tqdm import tqdm
        from tqdm.notebook import tqdm
        import scipy
        import time

        disable_tqdm = True
        print_info = False

        if verbosity == 0:
            disable_tqdm = True
            print_info = False

        if verbosity == 1:
            disable_tqdm = False
            print_info = True

        if print_info: print("INFO(clusters_pool): Initialization started")
        
        npoints = self._npoints
        scell = self._cpool_scell
        natoms = scell.get_natoms()
        sites = scell.get_sites()
        satoms = scell.get_substitutional_sites()
        nsatoms = len(satoms)
        distances = self._distances
        radii = self._radii

        symper = scell.get_sym_perm()
        
        if print_info: print("INFO(clusters_pool): Initialization finished")

        # Find Wyckoff sites
        wyck_idxs = set()
        full_list = set()

        for idx in tqdm(satoms, total=nsatoms, desc="INFO(clusters_pool): Finding wyckoff sites", disable=disable_tqdm):
            sigmas = np.zeros(natoms, dtype = "int")
            np.put(sigmas, idx, [1])
                    
            if idx not in full_list:
                wyck_idxs.add(idx)
                
            for per in symper:
                full_list.add(sigmas[np.ix_(per)].nonzero()[0][0])

        # Determine set of indices compatible with given radii
        idxs_sets = []
        for irad, radius in enumerate(radii):
            idxs_sets.append([])
            idxs = []
            for widx in wyck_idxs:
                idxs.append(widx)
                for satidx in satoms:
                    if self._distances_mic_true[widx,satidx] <= radius:
                        idxs.append(satidx)
            idxs_sets[irad] = np.unique(np.array(idxs))
            
        # Find out clusters
        cpool_idx_ss = []
        cpool_maxradii = []
        full_list = set()
        
        for i, (npts,radius) in enumerate(zip(npoints,radii)):
            
            n_max = int(scipy.special.binom(len(idxs_sets[i])-1, npts-1))*len(wyck_idxs)
            
            for widx in tqdm(
                    wyck_idxs,
                    total=len(wyck_idxs),
                    desc=f"Loop over Wyckoff sites",
                    position=0,
                    disable=disable_tqdm):
                for idxs_ in tqdm(
                        combinations(set(idxs_sets[i]).difference(set([widx])),npts-1),
                        total=n_max,
                        desc=f"INFO(clusters_pool): Finding unique clusters of {npts} points",
                        position=1,
                        leave=False,
                        disable=disable_tqdm):
                
                    idxs = [widx]
                    for idx in idxs_:
                        idxs.append(idx)

                    _radius = 0
                    for idxs2 in combinations(idxs,2):
                        d = self._distances_mic_true[idxs2]
                        if _radius < d:
                            _radius = d

                    if _radius <= radius or radius < 0:
                        sites_arrays = []
                        for idx in idxs:
                            sites_arrays.append(np.arange(1,len(sites[idx])))

                        for ss in product(*sites_arrays):

                            sigmas = np.zeros(natoms, dtype = "int")
                            np.put(sigmas, idxs, ss)

                            if tuple(sigmas.nonzero()[0].tolist()) not in full_list:
                                cpool_idx_ss.append([idxs,list(ss)])
                                cpool_maxradii.append(radius)

                            for per in symper:
                                newidxs = sigmas[np.ix_(per)].nonzero()[0]
                                if set(newidxs).intersection(wyck_idxs):
                                    full_list.add(
                                        tuple(
                                            newidxs.tolist()
                                        )
                                    )

        for i, idx_ss in tqdm(enumerate(cpool_idx_ss), total=len(cpool_idx_ss), desc="INFO(clusters_pool): Creating clusters objects", disable=disable_tqdm):
            sps = []
            for idx,isp in zip(idx_ss[0],idx_ss[1]):
                sps.append(sites[idx][isp])

            _cl = Cluster(idx_ss[0], sps)
            
            _orbit = self.get_cluster_orbit(scell, _cl.get_idxs(), _cl.get_nrs(),distances=distances)

            r = []
            for __cl in _orbit:
                r.append(__cl.get_radius())

            r = np.around(r, 5).tolist()
            rmin = min(r)
            if rmin > cpool_maxradii[i]:
                continue
            else:
                _cl = _orbit[r.index(min(r))]

            mult = _orbit.get_multiplicity_in_parent_lattice()
            
            self._cpool.append(Cluster(_cl.get_idxs(),_cl.get_nrs(),self._cpool_scell,self._distances))
            
            self._multiplicities.append(int(mult))
            
        if len(self._cpool) == 0:
            return [],0
        else:
            self._cpool, self._multiplicities = (list(t) for t in zip(*sorted(zip(self._cpool, self._multiplicities))))

            
    def gen_clusters1(self):
        from itertools import product, combinations

        npoints = self._npoints
        scell = self._cpool_scell
        sites = scell.get_sites()
        satoms = scell.get_substitutional_sites()
        distances = self._distances
        radii = self._radii
        
        for npts,radius in zip(npoints,radii):
            clrs_full = []
            for idxs in combinations(satoms,npts):
                sites_arrays = []
                for idx in idxs:
                    sites_arrays.append(sites[idx][1:])
                for ss in product(*sites_arrays):
                    _cl = Cluster(idxs, ss)
                    if _cl.get_radius(distances) <= radius:
                        clrs_full.append(_cl)

            clrs_full.sort()

            while len(clrs_full) != 0:
                _cl=clrs_full[0]
                _orbit = self.get_cluster_orbit(scell, _cl.get_idxs(), _cl.get_nrs(),distances=distances)
                mult = _orbit.get_multiplicity_in_parent_lattice()
                orbit = _orbit.as_array()
                orbit.sort()

                delids = []
                for __cl1 in orbit[:]:
                    for i,__cl2 in enumerate(clrs_full):
                        if __cl1 == __cl2:
                            delids.append(i)
                clrs_full = [c for i,c in enumerate(clrs_full) if i not in delids]

                self._cpool.append(Cluster(_cl.get_idxs(),_cl.get_nrs(),self._cpool_scell,self._distances))
                self._multiplicities.append(mult)

        if len(self._cpool) == 0:
            return [],0
        else:
            self._cpool, self._multiplicities = (list(t) for t in zip(*sorted(zip(self._cpool, self._multiplicities))))

        
    def get_cpool_scell(self): # Deprecated. Use get_supercell instead
        return self._cpool_scell
    
    def get_supercell(self):
        """Return SuperCell instance used for the construction of the clusters pool
        """
        return self._cpool_scell

    def get_cpool(self): # Deprecated. Use get_cpool_list instead
        return self.get_cpool_list()
    
    def get_cpool_list(self):
        """ Return python list containing the clusters in the pool
        """
        return self._cpool

    def get_cpool_clusters(self): # Deprecated. Use get_cpool_list instead
        return self.get_cpool_list()

    def get_cpool_arrays(self):
        """Get arrays of atom indices and atom numbers of the clusters in the pool
        """
        atom_idxs = []
        atom_nrs = []

        for cl in self._cpool:
            atom_idxs.append(cl.get_idxs())
            atom_nrs.append(cl.get_nrs())

        return np.array(atom_idxs, dtype=object), np.array(atom_nrs, dtype=object)

    def serialize(self, filepath="cpool.json", db_name=None):
        """Serialize clusters pool object to json database file

        The generated json file is compatible with ASE's GUI, so you can
        visualize the clusters with it. You can also initialize a new
        ClustersPool object from this file, for this read the documentation
        for the ``json_db_filepath`` attribute of ClustersPool class.

        **Parameters:**

        ``filepath``: string (default: "cpool.json")
            Name of the file to save the ClustersPool object. 

        ``db_name``: string (default: "cpool.json")
            *DEPRECATED*, use ``filepath`` instead. Name of the json database file.

        """
        if db_name is not None:
            filepath = db_name 

        self.write_clusters_db(db_name=filepath)

    def as_dict(self):
        """Return a python-dictionary representation of the clusters pool
        """
        return self.get_cpool_dict()

    # Deprecated: use as_dict() instead.
    def get_cpool_dict(self):
        nrs = []
        idxs = []
        pos = []
        alphas = []

        for cl in self._cpool:
            nrs.append(cl.get_nrs())
            idxs.append(cl.get_idxs())
            pos.append(cl.get_positions())
            alphas.append(cl.get_alphas())

        self._cpool_dict.update({"atom_numbers" : nrs})
        self._cpool_dict.update({"atom_indexes" : idxs})
        self._cpool_dict.update({"atom_positions" : pos})
        self._cpool_dict.update({"multiplicities" : self.get_multiplicities()})
        self._cpool_dict.update({"npoints" : self.get_all_npoints()})
        self._cpool_dict.update({"radii" : self.get_all_radii()})
        self._cpool_dict.update({"_npoints" : self._npoints})
        self._cpool_dict.update({"_radii" : self._radii})
        self._cpool_dict.update({"alphas" : alphas})
        self._cpool_dict.update({"nclusters" : len(self)})
        self._cpool_dict.update({"super_cell" : self._cpool_scell.as_dict()})
        self._cpool_dict.update({"parent_lattice" : self._plat.as_dict()})
        #self._cpool_dict.update({"parent_lattice_pbc" : self._plat.get_pbc()})
        #self._cpool_dict.update({"parent_lattice_pristine_unit_cell" : self._plat.get_cell()})
        #self._cpool_dict.update({"parent_lattice_pristine_positions" : self._plat.get_positions()})
        #self._cpool_dict.update({"parent_lattice_pristine_numbers" : self._plat.get_atomic_numbers()})
        #self._cpool_dict.update({"parent_lattice_tags" : self._plat.get_tags()})
        #self._cpool_dict.update({"parent_lattice_idx_subs" : self._plat.get_idx_subs()})

        return self._cpool_dict

    def dump_cpool_dict(self):
        print(json.dumps(self._cpool_dict,indent=4))

    def get_cluster(self, cln):
        return self._cpool_dict[cln]

    def write_clusters_db(self, orbit=None, super_cell=None, db_name="cpool.json"):
        """Write cluster orbit to Atoms JSON database

        **Parameters:**

        ``orbit``: python list (default: ``None``)
            list of ``Cluster`` objects. If ``None``, just the clusters in the ``ClustersPool`` object are written.

        ``super_cell``: ``SuperCell`` object (default: ``None``)
            The supercell in which the clusters are supported. 
            If ``None``, the ``SuperCell`` object of ``self`` (output of method ``get_cpool_scell()``) is used.

        ``db_name``: string
            Name of the json file containing the database
        """
        from ase.db.jsondb import JSONDatabase
        from subprocess import call
        
        call(["rm","-f",db_name])
        atoms_db = JSONDatabase(filename=db_name)

        cpool_atoms  = self.get_cpool_atoms(orbit=orbit, super_cell=super_cell)

        for atoms in cpool_atoms:
            atoms_db.write(atoms)
        #atoms_db.write(Atoms(symbols=None))

        atoms_db.metadata = self.get_cpool_dict()

    def get_cpool_atoms(self, orbit=None, super_cell=None):
        from clusterx.structure import Structure

        if orbit is None:
            orbit = self.get_cpool()
        elif isinstance(orbit,self.__class__):
            orbit = orbit.get_cpool()
        if super_cell is None:
            super_cell = self.get_cpool_scell()

        orbit_nrs = []
        orbit_idxs = []
        for cluster in orbit:
            orbit_nrs.append(cluster.get_nrs())
            orbit_idxs.append(cluster.get_idxs())

        atnums = super_cell.get_atomic_numbers()
        sites = super_cell.get_sites()
        idx_subs = super_cell.get_idx_subs()
        tags = super_cell.get_tags()

        self._cpool_atoms = []
        for icl,cl in enumerate(orbit):
            atoms = super_cell.copy()
            ans = atnums.copy()
            for i,atom_idx in enumerate(cl.get_idxs()):
                ans[atom_idx] = orbit_nrs[icl][i]
            atoms.set_atomic_numbers(ans)
            atoms0 = Structure(atoms,decoration=ans).get_atoms()
            positions = []
            numbers = []
            indices = []
            for i,atom in enumerate(atoms0):
                nr = atom.get('number')
                if nr != 0:
                    indices.append(i)
                    positions.append(atom.get('position'))
                    numbers.append(nr)
            self._cpool_atoms.append(Atoms(cell=atoms0.get_cell(), pbc=atoms0.get_pbc(),numbers=numbers,positions=positions))

        return self._cpool_atoms

    def get_cpool_structures(self):
        """Get array of structure objects representing the clusters in the clusters pool
        """
        from clusterx.structure import Structure

        strs = []
        nrs0 = self.get_cpool_scell().get_atomic_numbers()
        for cluster in self._cpool:
            cl_nrs = cluster.get_nrs()
            cl_idxs = cluster.get_idxs()
            nrs = nrs0.copy()
            for i,idx in enumerate(cl_idxs):
                nrs[idx] = cl_nrs[i]

            strs.append(Structure(super_cell=self.get_cpool_scell(), decoration = nrs))

        return strs
    

    def get_equiv_clusters(self, cluster):
        ids = cluster.get_idxs()
        nrs = cluster.get_nrs()

    def get_cluster_orbit(self, super_cell=None, cluster_sites=None, cluster_species=None, tol = 1e-3, distances=None, no_trans = False, cluster_index=None, cluster_positions = None):
        """
        Get cluster orbit inside a supercell.

        Returns a pool of clusters corresponding to a cluster orbit in the given super cell, the weights of each cluster
        in the orbit (accounting for the periodic boundary conditions) and the multiplicity of the cluster.

        An example is as follows::
        
            orbit = clusters_pool.get_cluster_orbit(scell, cluster_index = 3)

        Here, the orbit of the fourth cluster in the pool 
        (*i.e.*, the cluster ``clusters_pool[3]``) is computed.
        
        | The output ``orbit`` is a clusters pool object, containing the orbit of the input 
          cluster in the supercell ``scell`` (given as input parameter). 
        | The output ``weigths=[w1, w2, ... , wNc]`` is an integer numpy array with the same 
          length as the ``orbit``, with ``wi`` being the weight of cluster ``i`` in the orbit. 
        | The output ``mult`` is an integer number equal to the multiplicity of the cluster, i.e., 
          the number of distinct realizations of the cluster which appear by the application of all
          symmetries of the space group of the parent lattice. 
        | The output ``rmult`` is an integer number equal to the reduced multiplicity of the cluster, i.e., 
          the number of distinct realizations of the cluster which appear by the application of all
          symmetries of the space group of the parent lattice, but only those which correspond to 
          configurations that can be relized in the supercell. 

        The output verifies the relation :math:`n_{SC} m_r = \sum_i w_i`, 
        with :math:`n_{SC}` the "index" of the supercell (*i.e.* :math:`n_{SC}=N_{SC}/N_{pl}`, with 
        :math:`N_{SC}` the number of atoms in the supercell and :math:`N_{pl}` the number of atoms in the
        parent lattice), :math:`m_r` the reduced 
        multiplicity of the cluster and :math:`w_i` the weight of cluster *i* in the orbit. 
        That is, the statement::

            scell.get_index() * rmult == np.sum(weights)

        evaluates to ``True``.
        

        **Parameters:**

        ``super_cell``: SuperCell object
            The super cell in which the orbit is calculated.
        ``cluster_sites``: Array of integer
            the atom indices of the cluster as referred to the SuperCell object
            given in ``super_cell``, or the ``ClustersPool.get_cpool_scell()`` superCell
            object (see ``cluster_index``).
        ``cluster_species``: array of integer
            Decoration (with species numbers) of the cluster for which the orbit is calculated. The species
            numbers serve as index for the site cluster basis functions. Thus, for  instance
            if in a given site, say, ``i=12``, the possible species to put are ``14``,
            ``15`` and ``16`` (``14`` for the pristine), then ``15`` represents the site
            basis function with label ``1`` and ``16`` the basis function with label ``2``.
        ``tol``: float
            tolerance to determine whether cluster and atom positions are the same.
        ``distances``: 2D array of floats
             distances of all of the atoms with all of the atoms. Can be used to achieve larger efficiency.
        ``no_trans``: Boolean
            set to True to ignore translations of the parent_lattice inside the SuperCell. Thus
            a reduced orbit is obtained which only contains the symmetry operations of the parent lattice.
        ``cluster_index``: integer
            Index of a cluster in the pool. Overrides ``super_cell``, and the
            orbit is calculated on the supercell of the ``ClustersPool.get_cpool_scell()`` object.
        ``cluster_positions``: list of vectors 
            the atoms positions in cartesian coordinates. cluster_positions[i] = [ix, iy, iz], 
            where ix,iy and iz are float numbers representing the x, y and z cartesian coordinate, respectively,
            of atom i.
        """
        from clusterx.clusters.clusters_pool import ClusterOrbit

        if super_cell is None:
            super_cell = self.get_cpool_scell()
            
        if cluster_index is not None:
            super_cell = self.get_cpool_scell()
            atom_idxs, atom_nrs = self.get_cpool_arrays()
            cluster_sites = atom_idxs[cluster_index]
            cluster_species = atom_nrs[cluster_index]
            return ClusterOrbit(super_cell, cluster_sites, cluster_species, tol, distances, no_trans)

        if cluster_sites is not None and cluster_positions is not None:
            print("ERROR (clusterx.clusters.clusters_pool.gen_cluster_orbit): One of cluster_sites or cluster_positions must be None")

        if cluster_sites is not None:
            return ClusterOrbit(super_cell, cluster_sites, cluster_species, tol, distances, no_trans)
        
        if cluster_positions is not None:
            return ClusterOrbit(super_cell, cluster_positions = cluster_positions, cluster_species=cluster_species, tol=tol, distances=distances, no_trans=no_trans)

    def get_containing_supercell(self,tight=False):
        """
        Return a supercell able to contain all clusters in the pool defined by
        the arrays ``self.get_npoints()`` and ``self.get_radii()``.
        Returns the supercell which circumscribes a sphere of diameter at least
        as large as the largest cluster radius.
        """
        from clusterx.super_cell import SuperCell
        from numpy import linalg as LA

        rmax = np.amax(self._radii)
        #l = LA.norm(self._plat.get_cell(), axis=1) # Lengths of the cell vectors

        cell = self._plat.get_cell()

        c = np.zeros((3,3))
        h = np.zeros(3)
        # Get distances h between parallel planes of the unit cell
        for i in range(3):
            c[i] = np.cross(cell[(i+1)%3],cell[(i+2)%3])
            h[i] = np.dot(cell[i],c[i]/LA.norm(c[i]))

        if rmax == 0:
            m = np.diag([1,1,1])
        else:
            if tight:
                m = np.diag([int(n) for n in np.ceil(rmax/h)]) # number of repetitions of unit cell along each lattice vector to contain largest cluster
            else:
                m = np.diag([int(n) for n in np.ceil(2*rmax/h)]) # number of repetitions of unit cell along each lattice vector to contain largest cluster

        for i, p in enumerate(self._plat.get_pbc()): # Apply pbc's
            if not p:
                m[i,i] = 1

        sc =  SuperCell(self._plat, m)

        return sc

    def gen_atoms_database(self, fname="clusters.json"):
        """
        Builds an ASE's json database object (self._atoms_db). Atoms items in
        the built database are a representation of the clusters
        embedded in a supercell appropriate for visualization
        with ASE's gui.
        """
        from ase.data import chemical_symbols as cs
        from ase import Atoms
        from ase.db.jsondb import JSONDatabase
        from clusterx.utils import isclose
        from subprocess import call

        rtol = 1e-3
        cld = self.get_cpool_dict()
        prim_cell = self._plat.get_cell()
        #scell = self.get_containing_supercell()
        scell = self._cpool_scell

        call(["rm","-f",fname])
        atoms_db = JSONDatabase(filename=fname) # For visualization
        sites = scell.get_sites()
        for kcl,icl in cld.items():

            #wrap cluster positions
            chem = []
            for c in icl["site_basis"]:
                chem.append(cs[c[1]])

            atoms = Atoms(symbols=chem,positions=icl["positions_car"],cell=scell.get_cell(),pbc=scell.get_pbc())
            atoms.wrap(center=[0.5,0.5,0.5])
            wrapped_pos = atoms.get_positions()

            # Dummy species
            chem = []
            for i in range(scell.get_natoms()):
                chem.append("H")

            # Map cluster to supercell
            #for p,c in zip(icl["positions_car"],icl["site_basis"]):
            for p,c in zip(wrapped_pos,icl["site_basis"]):
                for ir,r in enumerate(scell.get_positions()):
                    if isclose(r,p,rtol=1e-2):
                        chem[ir] = cs[sites[ir][c[1]+1]]

            atoms = Atoms(symbols=chem,positions=scell.get_positions(),cell=scell.get_cell(),pbc=scell.get_pbc())
            atoms_db.write(atoms)

    def get_atoms_database(self):
        pass

    def display_info(self, ecis = None):
        """Display in screen information about the clusters pools

        DEPRECATED: Use `print_info()` instead.

        Displays on screen a table with information concerning the clusters pool.

        **Parameters**:

        ``ecis``: Array of float (optional)
            Effective cluster interactions. If not ``None``, the displayed
            table contains the ECI values in the last column.
        """
        import warnings
        warnings.warn("The name of this function has changed. Please use `print_info()`.", category=FutureWarning)
        self.print_info(ecis=ecis)

    def print_info(self, ecis = None):
        """Print information about the clusters pools

        Prints a table with information concerning the clusters pool.

        **Parameters**:

        ``ecis``: Array of float (optional)
            Effective cluster interactions. If not ``None``, the displayed
            table contains the ECI values in the last column.
        """
        info_str = ""
        if ecis is None:
            info_str += ("\n+-------------------------------------------------------------------------------+")
            info_str += ("\n|                             Clusters Pool Info                                |")
            info_str += ("\n+-------------------------------------------------------------------------------+")
            info_str += ("\n|{0:^19s}|{1:^19s}|{2:^19s}|{3:^19s}|".format("Index","Nr. of points","Radius","Multiplicity"))
            info_str += ("\n+-------------------------------------------------------------------------------+")
        else:
            info_str += ("\n+---------------------------------------------------------------------------------------------------+")
            info_str += ("\n|                                       Clusters Pool Info                                          |")
            info_str += ("\n+---------------------------------------------------------------------------------------------------+")
            info_str += ("\n|{0:^19s}|{1:^19s}|{2:^19s}|{3:^19s}|{4:^19s}|".format("Index","Nr. of points","Radius","Multiplicity","ECI"))
            info_str += ("\n+---------------------------------------------------------------------------------------------------+")
        for i, cl in enumerate(self._cpool):
            if ecis is None:
                info_str += ("\n|{0:^19d}|{1:^19d}|{2:^19.3f}|{3:^19d}|".format(i,cl.npoints,cl.radius,self._multiplicities[i]))
            else:
                info_str += ("\n|{0:^19d}|{1:^19d}|{2:^19.3f}|{3:^19d}|{4:^19.4f}|".format(i,cl.npoints,cl.radius,self._multiplicities[i],ecis[i]))

        if ecis is None:
            info_str += ("\n+-------------------------------------------------------------------------------+\n")
        else:
            info_str += ("\n+---------------------------------------------------------------------------------------------------+\n")

        print(info_str)
        return info_str

class ClusterOrbit(ClustersPool):
    """Cluster orbit class
    """
    def __init__(self, super_cell, cluster_sites=None, cluster_species=None, tol = 1e-3, distances=None, no_trans=False, json_db_filepath=None, cluster_positions=None):
        self.orbit_array = None
        
        if json_db_filepath is not None:
            super(ClusterOrbit,self).__init__(json_db_filepath=json_db_filepath)
            db = connect(json_db_filepath)
            self.weights = db.metadata.get("weights",np.array([],int))
            self.multiplicity = db.metadata.get("multiplicity",0)
            self.reduced_multiplicity = db.metadata.get("reduced_multiplicity",0)
        else:
            platt = super_cell.get_parent_lattice()
            super(ClusterOrbit,self).__init__(parent_lattice = platt, super_cell = super_cell)
            self._gen_orbit(super_cell, cluster_sites, cluster_species, tol, distances, no_trans, cluster_positions)
        # The iteration variable "high" was not set, because the clusters are not calculated by super().__init__().
        # Therefore we set it here explicitly.
        self.high = self.nclusters - 1


    def _gen_orbit(self, super_cell, cluster_sites=None, cluster_species=None, tol = 1e-3, distances=None, no_trans=False, cluster_positions=None):
        self.gen_orbit_slow_version(super_cell, cluster_sites=cluster_sites, cluster_species=cluster_species, tol = tol, distances=distances, no_trans=no_trans, cluster_positions=cluster_positions)
        

    def gen_orbit(self, super_cell, cluster_sites=None, cluster_species=None, tol = 1e-3, distances=None, no_trans=False, cluster_positions=None):
        """
        Generate cluster orbit inside a supercell.

        Returns a pool of clusters corresponding to a cluster orbit in the given super cell, the weights of each cluster
        in the orbit (accounting for the periodic boundary conditions) and the multiplicity of the cluster.

        An example is as follows::
        
            orbit = clusters_pool.get_cluster_orbit(scell, cluster_index = 3)

        Here, the orbit of the fourth cluster in the pool 
        (*i.e.*, the cluster ``clusters_pool[3]``) is computed.
        
        | The output ``orbit`` is a clusters pool object, containing the orbit of the input 
          cluster in the supercell ``scell`` (given as input parameter). 
        | The output ``weigths=[w1, w2, ... , wNc]`` is an integer numpy array with the same 
          length as the ``orbit``, with ``wi`` being the weight of cluster ``i`` in the orbit. 
        | The output ``mult`` is an integer number equal to the multiplicity of the cluster, i.e., 
          the number of distinct realizations of the cluster which appear by the application of all
          symmetries of the space group of the parent lattice. 
        | The output ``rmult`` is an integer number equal to the reduced multiplicity of the cluster, i.e., 
          the number of distinct realizations of the cluster which appear by the application of all
          symmetries of the space group of the parent lattice, but only those which correspond to 
          configurations that can be relized in the supercell. 

        The output verifies the relation :math:`n_{SC} m_r = \sum_i w_i`, 
        with :math:`n_{SC}` the "index" of the supercell (*i.e.* :math:`n_{SC}=N_{SC}/N_{pl}`, with 
        :math:`N_{SC}` the number of atoms in the supercell and :math:`N_{pl}` the number of atoms in the
        parent lattice), :math:`m_r` the reduced 
        multiplicity of the cluster and :math:`w_i` the weight of cluster *i* in the orbit. 
        That is, the statement::

            scell.get_index() * rmult == np.sum(weights)

        evaluates to ``True``.
        

        **Parameters:**

        ``super_cell``: SuperCell object
            The super cell in which the orbit is calculated.
        ``cluster_sites``: Array of integer
            the atom indices of the cluster as referred to the SuperCell object
            given in ``super_cell``, or the ``ClustersPool.get_cpool_scell()`` superCell
            object (see ``cluster_index``).
        ``cluster_species``: array of integer
            Decoration (with species numbers) of the cluster for which the orbit is calculated. The species
            numbers serve as index for the site cluster basis functions. Thus, for  instance
            if in a given site, say, ``i=12``, the possible species to put are ``14``,
            ``15`` and ``16`` (``14`` for the pristine), then ``15`` represents the site
            basis function with label ``1`` and ``16`` the basis function with label ``2``.
        ``tol``: float
            tolerance to determine whether cluster and atom positions are the same.
        ``distances``: 2D array of floats
             distances of all of the atoms with all of the atoms. Can be used to achieve larger efficiency.
        ``no_trans``: Boolean
            set to True to ignore translations of the parent_lattice inside the SuperCell. Thus
            a reduced orbit is obtained which only contains the symmetry operations of the parent lattice.
        ``cluster_index``: integer
            Index of a cluster in the pool. Overrides ``super_cell``, and the
            orbit is calculated on the supercell of the ``ClustersPool.get_cpool_scell()`` object.
        """
        from scipy.spatial.distance import cdist
        from sympy.utilities.iterables import multiset_permutations
        import sys
        from collections import Counter
        from clusterx.utils import isclose

        round_decimals = 8

        natoms = super_cell.get_natoms()
        sites = super_cell.get_sites()
        
        # empty cluster
        empty_cluster = None
        if (cluster_sites is None and cluster_positions is None):
            return
        if cluster_sites is not None:
            if len(cluster_sites) == 0:
                empty_cluster = True
        if cluster_positions is not None:
            if len(cluster_positions) == 0:
                empty_cluster = True
                
        if empty_cluster:
            self.add_cluster(Cluster([],[],super_cell))
            self.weights = np.array([1], int)
            self.multiplicity = 1
            self.reduced_multiplicity = 1
            return

        if cluster_sites is None:
            spos1 = super_cell.get_scaled_positions(wrap=True) # Super-cell scaled positions
            sps_sc = np.around(spos1,round_decimals) # Wrapping in ASE doesn't always work. Here a hard external fix by rounding and then applying again ASE's style wrapping.

            for i, periodic in enumerate(super_cell.get_pbc()):
                if periodic:
                    sps_sc[:, i] %= 1.0
                    sps_sc[:, i] %= 1.0

            #sp0 = get_scaled_positions(p0, self._plat.get_cell(), pbc = super_cell.get_pbc(), wrap = False) # sp0: scaled cluster positions with respect to parent lattice
            sps_cl = np.around(get_scaled_positions(cluster_positions, super_cell.get_cell(), pbc = super_cell.get_pbc(), wrap = True), round_decimals)

            cluster_sites = []

            option_2 = 2

            if option_2 == 1: 
                for sp_cl in sps_cl:
                    for idxsc,sp_sc in enumerate(sps_sc):
                        if isclose(sp_sc,sp_cl):
                            cluster_sites.append(idxsc)
            if option_2 == 2:
                for sp_cl in sps_cl:
                    cluster_sites.append(np.where(np.all(sps_sc==sp_cl,axis=1))[0][0]) # THIS MAY NEED TO BE REPLACED WITH utils.isclose() !!!
                

        cluster_tags = []
        for idx, sp in zip(cluster_sites, cluster_species):
            cluster_tags.append(np.where(sites[idx] == sp)[0][0])

        small_orbit = set()
        for per in super_cell.get_sym_perm(include_sc_trans = False):
            csper = []
            for idx in cluster_sites:
                csper.append(per[idx])
                
            small_orbit.add(
                tuple(
                    csper
                )
            )

        self.reduced_multiplicity = len(small_orbit)

        orbit_set = set()
        orbit_dict = {}
        for per in super_cell.get_sym_perm(include_sc_trans = not no_trans):
            idxs = []
            for idx in cluster_sites:
                idxs.append(per[idx])
            idxs_tuple = tuple(idxs)

            if idxs_tuple not in orbit_set:
                orbit_set.add(idxs_tuple)
                orbit_dict[idxs_tuple] = 1
            else:
                orbit_dict[idxs_tuple] += 1

        orbit_list = orbit_dict.keys()
        self.weights = np.array(list(orbit_dict.values()))
        self.multiplicity = self.reduced_multiplicity
            
        orbit = []
        
        for cl_tuple in orbit_list:
            orbit.append(Cluster(list(cl_tuple),cluster_species,super_cell,distances))
            
        for cl in orbit:
            self.add_cluster(cl)

    def gen_orbit_slow_version(self, super_cell, cluster_sites=None, cluster_species=None, tol = 1e-3, distances=None, no_trans=False, cluster_positions=None):
        """
        Generate cluster orbit inside a supercell.

        Returns a pool of clusters corresponding to a cluster orbit in the given super cell, the weights of each cluster
        in the orbit (accounting for the periodic boundary conditions) and the multiplicity of the cluster.

        An example is as follows::
        
            orbit = clusters_pool.get_cluster_orbit(scell, cluster_index = 3)

        Here, the orbit of the fourth cluster in the pool 
        (*i.e.*, the cluster ``clusters_pool[3]``) is computed.
        
        | The output ``orbit`` is a clusters pool object, containing the orbit of the input 
          cluster in the supercell ``scell`` (given as input parameter). 
        | The output ``weigths=[w1, w2, ... , wNc]`` is an integer numpy array with the same 
          length as the ``orbit``, with ``wi`` being the weight of cluster ``i`` in the orbit. 
        | The output ``mult`` is an integer number equal to the multiplicity of the cluster, i.e., 
          the number of distinct realizations of the cluster which appear by the application of all
          symmetries of the space group of the parent lattice. 
        | The output ``rmult`` is an integer number equal to the reduced multiplicity of the cluster, i.e., 
          the number of distinct realizations of the cluster which appear by the application of all
          symmetries of the space group of the parent lattice, but only those which correspond to 
          configurations that can be relized in the supercell. 

        The output verifies the relation :math:`n_{SC} m_r = \sum_i w_i`, 
        with :math:`n_{SC}` the "index" of the supercell (*i.e.* :math:`n_{SC}=N_{SC}/N_{pl}`, with 
        :math:`N_{SC}` the number of atoms in the supercell and :math:`N_{pl}` the number of atoms in the
        parent lattice), :math:`m_r` the reduced 
        multiplicity of the cluster and :math:`w_i` the weight of cluster *i* in the orbit. 
        That is, the statement::

            scell.get_index() * rmult == np.sum(weights)

        evaluates to ``True``.
        

        **Parameters:**

        ``super_cell``: SuperCell object
            The super cell in which the orbit is calculated.
        ``cluster_sites``: Array of integer
            the atom indices of the cluster as referred to the SuperCell object
            given in ``super_cell``, or the ``ClustersPool.get_cpool_scell()`` superCell
            object (see ``cluster_index``).
        ``cluster_species``: array of integer
            Decoration (with species numbers) of the cluster for which the orbit is calculated. The species
            numbers serve as index for the site cluster basis functions. Thus, for  instance
            if in a given site, say, ``i=12``, the possible species to put are ``14``,
            ``15`` and ``16`` (``14`` for the pristine), then ``15`` represents the site
            basis function with label ``1`` and ``16`` the basis function with label ``2``.
        ``tol``: float
            tolerance to determine whether cluster and atom positions are the same.
        ``distances``: 2D array of floats
             distances of all of the atoms with all of the atoms. Can be used to achieve larger efficiency.
        ``no_trans``: Boolean
            set to True to ignore translations of the parent_lattice inside the SuperCell. Thus
            a reduced orbit is obtained which only contains the symmetry operations of the parent lattice.
        ``cluster_index``: integer
            Index of a cluster in the pool. Overrides ``super_cell``, and the
            orbit is calculated on the supercell of the ``ClustersPool.get_cpool_scell()`` object.
        """
        from scipy.spatial.distance import cdist
        from sympy.utilities.iterables import multiset_permutations
        import sys
        from collections import Counter

        if cluster_sites is not None:
            # Get original cluster cartesian positions (p0)
            pos = super_cell.get_positions(wrap=True)
            p0 = np.array([pos[site] for site in cluster_sites])
        else:
            p0 = cluster_positions
            
        # empty cluster
        if len(p0) == 0:
            self.add_cluster(Cluster([],[],super_cell))
            self.weights = np.array([1], int)
            self.multiplicity = 1
            self.reduced_multiplicity = 1
            return

        cluster_species = np.array(cluster_species)
        
        # Get internal translation of the parent lattice in the super cell
        if no_trans:
            internal_trans = np.zeros((3,3))
        else:
            internal_trans = super_cell.get_internal_translations() # Scaled to super_cell

        if distances is None:
            distances = super_cell.get_all_distances(mic=False)
        

        spos1 = super_cell.get_scaled_positions(wrap=True) # Super-cell scaled positions
        spos = np.around(spos1,8) # Wrapping in ASE doesn't always work. Here a hard external fix by rounding and then applying again ASE's style wrapping.

        for i, periodic in enumerate(super_cell.get_pbc()):
            if periodic:
                spos[:, i] %= 1.0
                spos[:, i] %= 1.0
                

        # Get cluster orbit for the point symmetry operations of the parent lattice only (small cluster orbit)
        sp0 = get_scaled_positions(p0, self._plat.get_cell(), pbc = super_cell.get_pbc(), wrap = False) # sp0: scaled cluster positions with respect to parent lattice
        
        _orbit = []
        clset = set()

        orbit0 = self._get_small_cluster_orbit(sp0, cluster_species)
        
        mult = len(orbit0) # Multiplicity of the cluster with respect to the point symmetries of the parent lattice (not the supercell). This is the standard cluster multiplicity used in literature.

        reduced_orbit0 = []

        for _sp1 in orbit0:
            # Get cartesian, then scaled to supercell
            _p1 = np.dot(_sp1, self._plat.get_cell())
            __sp1 = get_scaled_positions(_p1, super_cell.get_cell(), pbc = super_cell.get_pbc(), wrap = True)
            reduced_orbit0.append(__sp1)
            
        reduced_mult = mult
        
        for _sp1 in reduced_orbit0:
            for itr,tr in enumerate(internal_trans): # Now apply the internal translations
                __sp1 = np.add(_sp1, tr)
                __sp1 = wrap_scaled_positions(__sp1,super_cell.get_pbc())
                _cl = get_cl_idx_sc(__sp1, spos, method=1, tol=tol)

                ocl = Cluster(_cl,cluster_species)
                _orbit.append(ocl)
                clset.add(ocl)

        weights = []
        orbit = []
        crossedout = set()
        ncl = len(_orbit)
        cnt = 0
        for i in range(ncl):
            if i not in crossedout:
                weights.append(1)
                orbit.append(Cluster(_orbit[i].get_idxs(),_orbit[i].get_nrs(),super_cell,distances))
                cnt += 1
            
                for j in range(i+1, ncl):
                    if j not in crossedout:
                        if _orbit[i].myhash == _orbit[j].myhash:
                            weights[cnt-1] += 1
                            crossedout.add(j)
                        

        for cl in orbit:
            self.add_cluster(cl)
        
        self.weights = np.array(weights, int)
        self.multiplicity = mult
        self.reduced_multiplicity = reduced_mult
        

    def as_array(self):
        if self.orbit_array is None:
            self.orbit_array = []
            for cl in self._cpool:
                self.orbit_array.append(cl)

        return self.orbit_array
            
    
    def _get_small_cluster_orbit(self, cluster_scaled_positions=None, cluster_species=None, tol = 1e-3):
        """ For a given cluster, return the set of distinct clusters obtained by applying the point
        symmetries of the parent lattice.

        **Returns:**
            2D array of shape (nc, np), where nc is the number of clusters of the small cluster orbit and np is the number of points of the cluster.
            E.g. the element [i,:] will be a vector of length np contaning the scaled poisitions of the points of cluster i.

        **Note:**
           This operation does not rely on a supercell. It just uses information from the parent lattice and the cluster. 
        """

        sp0 = cluster_scaled_positions

        orbit0 = []
        for r,t in zip(self.sc_sym['rotations'], self.sc_sym['translations']):
            ts = np.tile(t,(len(sp0),1)).T # Every column represents the same translation for every cluster site
            orbit0.append(np.add(np.dot(r,sp0.T),ts).T) # Apply rotation, then translation

        orbit1 = []
        n = len(orbit0)
        crossedout = []
        
        for i in range(n):
            
            if i in crossedout:
                continue

            sp_i = orbit0[i]

            orbit1.append(sp_i)
            crossedout.append(i)
            
            for j in range(i+1,n):
                if j not in crossedout:
                    sp_j = orbit0[j]
                    if self._are_equivalent_clusters(sp_i, sp_j, cluster_species, cluster_species, tol):
                        crossedout.append(j)

        return orbit1

    def _are_equivalent_clusters(self, scaled_positions1, scaled_positions2, species1, species2, tol = 1e-3):
        """Check for symmetric equivalence with respect to unit cell translations. Takes into account positions and species.
        """
        from scipy.spatial import distance_matrix
        
        sp1 = scaled_positions1
        sp2 = scaled_positions2
        s1 = species1
        s2 = species2

        centroid1 = np.mean(sp1, axis=0)
        centroid2 = np.mean(sp2, axis=0)

        c12 = np.subtract(centroid1, centroid2)

        for c in np.around(c12,decimals=2):
            if not c.is_integer():
                return False

        _sp2 = np.add(sp2,c12)
        sp2 = _sp2

        d = distance_matrix(sp1, sp2)

        n = len(sp1)
        jout = []
        
        for i in range(n):
            for j in range(n):
                if j not in jout and d[i,j] < tol and s1[i] == s2[j]:
                    jout.append(j)
                    break

            if len(jout) != i+1:
                return False

        return True

    def get_multiplicity_in_parent_lattice(self):
        return self.multiplicity
    
    def get_reduced_multiplicity(self):
        return self.reduced_multiplicity

    def get_weights(self):
        return self.weights

    
    def serialize(self, json_db_filepath="cpool.json"):
        """Write cluster orbit to Atoms JSON database

        **Parameters:**

        ``orbit``: python list (default: ``None``)
            list of ``Cluster`` objects. If ``None``, just the clusters in the ``ClustersPool`` object are written.

        ``super_cell``: ``SuperCell`` object (default: ``None``)
            The supercell in which the clusters are supported. 
            If ``None``, the ``SuperCell`` object of ``self`` (output of method ``get_cpool_scell()``) is used.

        ``db_name``: string
            Name of the json file containing the database
        """
        from ase.db.jsondb import JSONDatabase
        from subprocess import call
        
        call(["rm","-f",json_db_filepath])
        atoms_db = JSONDatabase(filename=json_db_filepath)

        cpool_atoms  = self.get_cpool_atoms()

        for atoms in cpool_atoms:
            atoms_db.write(atoms)
        #atoms_db.write(Atoms(symbols=None))

        atoms_db.metadata = self.get_cpool_dict()
