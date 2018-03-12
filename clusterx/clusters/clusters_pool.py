from ase.db.jsondb import JSONDatabase
from ase.atoms import Atoms
import clusterx as c
from clusterx.clusters.cluster import Cluster
import os
import sys
import subprocess
import numpy as np
import json

class ClustersPool():
    """
    Clusters pool class
    """
    def __init__(self, parent_lattice, npoints=np.array([[0]]), radii=np.array([[0]]), name="_clusters_pool", filename=None):
        self._npoints = npoints
        self._radii = radii
        self._name = name
        self._filename = name+".json"
        self._plat = parent_lattice
        self._cpool = []
        self._cpool_scell = self.get_containing_supercell(use="radii")
        self._cpool_dict = {}
        if (npoints != np.array([[0]])).any():
            self.gen_clusters()

    def __len__(self):
        return len(self._cpool)
    
    def gen_clusters(self):
        from clusterx.super_cell import SuperCell
        from itertools import product, combinations
        npoints = self._npoints
        radii = self._radii

        scell = self._cpool_scell
        natoms = scell.get_natoms()
        sites = scell.get_sites()
        satoms = scell.get_substitutional_sites()
        nsatoms = len(satoms)
        idx_subs = scell.get_idx_subs()
        tags = scell.get_tags()
        distances = scell.get_all_distances(mic=False)
        # Check if supercell is large enough
        if np.amax(radii)>np.amax(distances):
            sys.exit("Containing supercell is too small to find clusters pool.")
        
        for npts,radius in zip(npoints,radii):
            clrs_full = []
            for idxs in combinations(satoms,npts):
                sites_arrays = []
                for idx in idxs:
                    sites_arrays.append(sites[idx][1:])
                for ss in product(*sites_arrays):
                    cl = Cluster(idxs,ss,scell)
                    if self.get_cluster_radius(distances,cl) <= radius:
                        if cl not in clrs_full:
                            self._cpool.append(cl)
                            clrs_full.append(cl)
                            orbit = self.get_cluster_orbit(scell, cl.get_idxs(), cluster_species=cl.get_nrs(),tight=True,distances=distances)
                            #for cl_idxs in orbit:
                            #    clrs_full.append(Cluster(cl_idxs,cl.get_nrs(),scell))
                            for _cl in orbit:
                                clrs_full.append(_cl)

            
    def get_cpool_scell(self):
        return self._cpool_scell
    
    def get_cpool(self):
        return self._cpool
    
    def get_cpool_arrays(self):
        atom_idxs = []
        atom_nrs = []

        for cl in self._cpool:
            atom_idxs.append(cl.get_idxs())
            atom_nrs.append(cl.get_nrs())

        return np.array(atom_idxs), np.array(atom_nrs)
        
    def get_cluster_radius(self, distances, cluster):
        npoints = len(cluster)
        atom_idxs = cluster.get_idxs()
        r = 0
        if npoints == 0 or npoints == 1:
            return 0
        if npoints > 1:
            for i1, idx1 in enumerate(atom_idxs):
                for idx2 in atom_idxs[i1+1:]:
                    d = distances[idx1,idx2]
                    if r < d:
                        r = d
        return r
    
    def serialize(self, fmt, fname=None):
        if fmt == "json":
            self.gen_atoms_database(fname)

    def get_cpool_dict(self):
        return self._cpool_dict

    def dump_cpool_dict(self):
        print(json.dumps(self._cpool_dict,indent=4))
            
    def get_clusters_array(self):
        cld = self.get_cpool_dict()
        cla = []
        for key,item in cld.items():
            cla.append([])
            cla[key].append(item["multiplicity"])
            cla[key].append(item["radius"])
            cla[key].append(item["npoints"])
            if item["npoints"] > 0:
                cla[key].append(item["positions_lat"])
                cla[key].append(item["site_basis"])
        return cla

    def get_cluster(self, cln):
        return self._cpool_dict[cln]

    def write_orbit_db(self, orbit, super_cell, db_name):
        """Write cluster orbit to Atoms database
        """
        from ase.db.jsondb import JSONDatabase
        from subprocess import call
        orbit_nrs = []
        orbit_idxs = []
        for cluster in orbit:
            orbit_nrs.append(cluster.get_nrs())
            orbit_idxs.append(cluster.get_idxs())

        atnums = super_cell.get_atomic_numbers()
        sites = super_cell.get_sites()
        idx_subs = super_cell.get_idx_subs()
        tags = super_cell.get_tags()

        call(["rm","-f",db_name])
        atoms_db = JSONDatabase(filename=db_name) # For visualization
        for icl,cl in enumerate(orbit):
            atoms = super_cell.copy()
            ans = atnums.copy()
            for i,atom_idx in enumerate(cl.get_idxs()):
                ans[atom_idx] = orbit_nrs[icl][i]
            atoms.set_atomic_numbers(ans)
            atoms_db.write(atoms)


    def get_cluster_orbit(self, super_cell, cluster_sites, cluster_species, tol = 1e-3, tight=False, distances=None):
        """
        Get cluster orbit inside a supercell.
        cluster_sites: array of atom indices of the cluster, referred to the supercell.
        """
        from clusterx.symmetry import get_spacegroup, get_scaled_positions, get_internal_translations, wrap_scaled_positions
        from scipy.spatial.distance import cdist
        from sympy.utilities.iterables import multiset_permutations
        import sys
        from collections import Counter
        
        substitutional_sites = super_cell.get_substitutional_sites()
        for _icl in cluster_sites:
            if _icl not in substitutional_sites:
                return None

        radius = None
        if tight:
            radius = self.get_cluster_radius(distances,Cluster(cluster_sites,cluster_species,super_cell))
            
        shash = None
        cluster_species = np.array(cluster_species)
        # Get symmetry operations of the parent lattice
        sc_sg, sc_sym = get_spacegroup(self._plat) # Scaled to parent_lattice
        internal_trans = get_internal_translations(self._plat, super_cell) # Scaled to super_cell
        # Get original cluster cartesian positions (p0)
        pos = super_cell.get_positions(wrap=True)
        p0 = np.array([pos[site] for site in cluster_sites])
        
        spos = super_cell.get_scaled_positions(wrap=True) # Super-cell scaled positions
        # sp0: scaled cluster positions with respect to parent lattice
        sp0 = get_scaled_positions(p0, self._plat.get_cell(), pbc = super_cell.get_pbc(), wrap = False)
        orbit = []
        for r,t in zip(sc_sym['rotations'], sc_sym['translations']):
            ts = np.tile(t,(len(sp0),1)).T # Every column represents the same translation for every cluster site 
            _sp1 = np.add(np.dot(r,sp0.T),ts).T # Apply rotation, then translation
            # Get cartesian, then scaled to supercell
            _p1 = np.dot(_sp1, self._plat.get_cell())
            _sp1 = get_scaled_positions(_p1, super_cell.get_cell(), pbc = super_cell.get_pbc(), wrap = True)

            for tr in internal_trans: # Now apply the internal translations
                __sp1 = np.add(_sp1, tr)
                __sp1 = wrap_scaled_positions(__sp1,super_cell.get_pbc())
                sdistances = cdist(__sp1, spos, metric='euclidean') # Evaluate all (scaled) distances between cluster points to scell sites
                _cl = np.argwhere(np.abs(sdistances) < tol)[:,1] # Atom indexes of the transformed cluster

                include = True
                shash = np.arange(len(_cl))
                for _icl in _cl:
                    if _icl not in substitutional_sites:
                        include = False
                        break
                    
                if len(_cl)>1:
                    for i in range(len(_cl)):
                        for j in range(i+1,len(_cl)):
                            if _cl[i] == _cl[j] and cluster_species[i] != cluster_species[j]:
                                include = False
                                
                if tight:
                    _radius = self.get_cluster_radius(distances,Cluster(_cl,cluster_species,super_cell))
                    if _radius > radius:
                        include = False
                    
                if include:
                    for cl_obj in orbit:
                        cl = cl_obj.get_idxs()
                        if Counter(cl) == Counter(_cl):
                            shash = self._find_hash(cl,_cl)

                            csh = cluster_species[shash[:]]
                            if (cluster_species == csh).all():
                                include = False
                                break
                                
                if include:
                    orbit.append(Cluster(_cl,cluster_species,super_cell))

        return np.array(orbit)

    def _find_hash(self,a,b):
        table = np.zeros(len(a),dtype=int)
        for i,ia in enumerate(a):
            for j, jb in enumerate(b):
                if ia == jb and j not in table:
                    table[i] = j
        return table

    def get_containing_supercell(self, use = "pool"):
        """
        Return a supercell able to contain all clusters in the pool
        use: "pool" or "radii"
        If "pool", it returns the supercell twice as large as the largest cluster in the pool.
        If "radii", it returns the supercell which contains a sphere of diameter at least as large as the largest cluster radius.
        """
        from clusterx.super_cell import SuperCell

        if use == "pool":
            posl = []
            for cln, cl in self._cpool_dict.items():
                for pos in cl["positions_lat"]:
                    posl.append(pos)
            posl = np.array(posl)

            dn = 2*(np.ceil(np.max(posl,axis=0)).astype(int) - np.floor(np.min(posl,axis=0)).astype(int)) 
            sc = SuperCell(self._plat,np.diag(dn))

        if use == "radii":
            from numpy import linalg as LA
            rmax = np.amax(self._radii)
            if rmax == 0:
                rmax = 1e-2
            l = LA.norm(self._plat.get_cell(), axis=1) # Lengths of the cell vectors
            n = [int(n) for n in np.ceil(rmax/l)] # number of repetitions of unit cell along each lattice vector to contain largest cluster
            for i, p in enumerate(self._plat.get_pbc()): # Apply pbc's
                if not p:
                    n[i] = 1
            sc =  SuperCell(self._plat, np.diag(n))

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
        scell = self.get_containing_supercell()
        
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

        

