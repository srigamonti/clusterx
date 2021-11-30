# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from collections import Counter
import numpy as np
import sys
from clusterx.symmetry import get_scaled_positions
from clusterx.utils import PolynomialBasis
from functools import lru_cache

class CorrelationsCalculator():
    """
    Correlations calculator object.

    **Parameters:**

    ``basis``: string
        |  cluster basis to be used. Possible values are: ``binary-linear``, ``trigonometric``, ``polynomial``, and ``chebyshev``.
        |  ``binary-linear``: highly interpretable, non-orthogonal basis functions for binary compounds
        |  ``trigonomentric``: orthonormal basis; constructed from sine and cosine functions; based on: Axel van de Walle, CALPHAD 33, 266 (2009)
        |  ``polynomial``: orthonormal basis; uses orthogonalized polynomials
        |  ``chebyshev``: orthonormsl basis; chebyshev polynomials for symmetrized sigmas (sigma in {-m/2, ..., 0, ..., m/2 }); based on: J.M. Sanchez, Physica 128A, 334-350 (1984)
    ``parent_lattice``: ParentLattice object
        the parent lattice of the cluster expansion.
    ``clusters_pool``: ClustersPool object
        The clusters pool to be used in the calculator.
    ``lookup``: boolean
        Switches if a lookup table for the single-site basis functions should be used. Default is ``True``. Reduces performance in case of 'binary-linear' basis.

    ``json_db_filepath``: string (default: None)
        Overrides all the above. Used to initialize from file. Path of a json
        file containing a serialized CorrelationsCalculator object, as generated by the ``CorrelationsCalculator.serialize()`` method.

    """
    def __init__(self, basis = None, parent_lattice = None, clusters_pool = None, json_db_filepath = None, db = None, lookup = True):
        if json_db_filepath is not None:
            from ase.db import connect
            db = connect(json_db_filepath)
        
        if db is not None:
            corr_dict = db.metadata.get('correlations_calculator',None)
            self.basis = corr_dict.get('basis','trigonometric')

            from clusterx.clusters.clusters_pool import ClustersPool
            self._cpool = ClustersPool(db = db)
            self._plat = self._cpool._plat            
            lookup = corr_dict.get('lookup',lookup)

        else:
            self.basis = basis
            self._plat = parent_lattice
            self._cpool = clusters_pool
            
        # For each supercell (with corresponding transformation matrix) a set of cluster orbit set is created
        self._scells = []
        self._cluster_orbits_set = []
        ####
        self._2pi = 2*np.pi
        if self.basis == 'polynomial':
            self.basis_set = PolynomialBasis()
        elif self.basis == 'chebyshev':
            self.basis_set = PolynomialBasis(symmetric = True)

        if lookup:
            self._lookup_table = self._get_lookup_table()
        else:
            self._lookup_table = None

        self._mc = False
        self._num_mc_calls = 0
        self._cluster_orbits_mc = None

        
    def serialize(self, db_name = 'corrcalc.json'):
        """Write correlations calculator to Atoms Json database

        **Parameter**:

        ``db_name``: string (default: 'corrcalc.josn')
            Name of the json file containing the database

        """
        from ase.db.jsondb import JSONDatabase
        from subprocess import call
        call(["rm","-f",db_name])
        atoms_db = JSONDatabase(filename=db_name)

        cpool_atoms  = self._cpool.get_cpool_atoms()

        for atoms in cpool_atoms:
            atoms_db.write(atoms)
        #atoms_db.write(Atoms(symbols=None))
        cpooldict =self._cpool.get_cpool_dict()

        corr_dict = {}
        corr_dict.update({'basis':self.basis})
        
        if self._lookup_table is not None:
            corr_dict.update({'lookup':True})
        else:
            corr_dict.update({'lookup':False})
            
        cpool.update({'correlations_calculator':corr_dict})
        atoms_db.metadata = cpooldict

    def _get_lookup_table(self):
        idx_subs = self._plat.get_idx_subs()
        max_m = max([len(idx_subs[x]) for x in idx_subs])
        if max_m < 2:
            raise ValueError("No substitutional sites.")
        lookup_table = np.empty((max_m,max_m,max_m))
        for m in range(max_m):
            for alpha in range(m+1):
                for sigma in range(m+1):
                    lookup_table[m][alpha][sigma] = self.site_basis_function(alpha, sigma, m+1)
        return lookup_table

    @lru_cache(maxsize=None)
    def _trigo_basis_function(self,alpha,sigma,m):
        # Axel van de Walle, CALPHAD 33, 266 (2009)
        
        if alpha == 0:
            return 1

        elif alpha%2 != 0:
            return -np.cos(self._2pi*np.ceil(alpha/2.0)*sigma/m)

        else:
            return -np.sin(self._2pi*np.ceil(alpha/2.0)*sigma/m)

        
    #@profile
    def site_basis_function(self, alpha, sigma, m):
        """
        Calculates the site basis function.

        Evaluation of the single site basis functions using different basis sets.

        **Parameters:**

        ``alpha``: integer
            integer number between 0 and ``m`` - 1; represents the index of the basis function
        ``sigma``: integer
            integer number between 0 and ``m`` - 1; represents the occupation variable
        ``m``: integer
            number of components of the sublattice

        """

        #if hasattr(self, "_lookup_table") and self.basis != "binary-linear":
        #if hasattr(self, "_lookup_table"):
        #    return self._lookup_table[m-1][alpha][sigma]

        if self.basis == "trigonometric":
        
            """
            # Axel van de Walle, CALPHAD 33, 266 (2009)
            if alpha == 0:
                return 1

            elif alpha%2 != 0:
                return -np.cos(self._2pi*np.ceil(alpha/2.0)*sigma/m)

            else:
                return -np.sin(self._2pi*np.ceil(alpha/2.0)*sigma/m)
            """
            return self._trigo_basis_function(alpha,sigma,m)
        

        if self.basis == "binary-linear":
            # Only for binary alloys. Allows for simple interpretation of cluster interactions.
            return sigma

        if self.basis == "polynomial":

            return self.basis_set.evaluate(alpha, sigma, m)

        if self.basis == "chebyshev":
            # Method proposed by J.M. Sanchez, Physica 128A, 334-350 (1984).
            # Equivalent to polynomial basis.

            def _map_sigma(sigma, m):
                # Maps sigma = 0, 1, 2, ..., M-1 to -M/2 <= sigma <= M/2.
                shifted_sigma = int(sigma - int(m / 2))
                if (m % 2) == 0:
                    if shifted_sigma >= 0:
                        shifted_sigma += 1
                return shifted_sigma

            sigma = _map_sigma(sigma, m)

            return self.basis_set.evaluate(alpha, sigma, m)


    def cluster_function(self, cluster, structure_sigmas, ems):
        cluster_atomic_idxs = np.array(cluster.get_idxs())
        cluster_alphas = cluster.alphas
        cf = 1.0
        for cl_alpha, cl_idx in zip(cluster_alphas,cluster_atomic_idxs):
            cf *= self.site_basis_function(cl_alpha, structure_sigmas[cl_idx], ems[cl_idx])
        return cf

    def get_binary_random_structure_correlations(self,concentration):
        """Return cluster correlations for binary quasirandom structure

        .. todo::
            extend for other bases. Write method for n-aries.
        """
        correlations = np.zeros(len(self._cpool))
        if self.basis == "binary-linear":
            for icl,cl in enumerate(self._cpool.get_cpool()):
                correlations[icl]=np.power(concentration,cl.npoints)
        else:
            return None

        return np.around(correlations,decimals=12)

    def get_orbit_lengths(self,structure):
        """Return integer array of orbit lenghts

        ``structure``: ParentLattice, SuperCell, or Structure object
            Object containing the lattice definition to determine the orbit
            of the clusters in the CorrelationsCalculator.

        """
        cluster_orbits = self.get_cluster_orbits_for_scell(structure)
        lengths = np.zeros(len(cluster_orbits),dtype=int)
        for i,orbit in enumerate(cluster_orbits):
            lengths[i] = len(orbit)
        return lengths

    def get_cluster_orbits_for_scell(self,scell):
        """Return array of cluster orbits for a given supercell

        **Parameters**

        ``scell``: ParentLattice, SuperCell, or Structure object
            Object containing the lattice definition to determine the orbit
            of the clusters in the CorrelationsCalculator.
        """
        #if isinstance(scell,Structure):
        from clusterx.utils import get_cl_idx_sc
        from clusterx.clusters.clusters_pool import ClustersPool
                
        cluster_orbits = None

        for i, _scell in enumerate(self._scells):
            if cluster_orbits is None:
                if len(scell.get_positions()) == len(_scell.get_positions()):
                    if np.allclose(scell.get_positions(),_scell.get_positions(),atol=1e-3):
                        cluster_orbits = self._cluster_orbits_set[i]
                        break

        if cluster_orbits is None:
            from clusterx.structure import Structure
            from clusterx.symmetry import wrap_scaled_positions
            # Add new super cell and calculate cluster orbits for it.
            cluster_orbits = []
            #scell = structure.get_supercell()
            from clusterx.super_cell import SuperCell
            if isinstance(scell,Structure):
                scell = scell.get_supercell()
            elif isinstance(scell,SuperCell):
                pass
            elif isinstance(scell,ParentLattice):
                scell = SuperCell(scell,[1,1,1])

            cpool = ClustersPool(scell.get_parent_lattice(),super_cell=scell)

            for icl,cluster in enumerate(self._cpool.get_cpool()):
                positions = cluster.get_positions()

                cl_spos = wrap_scaled_positions(get_scaled_positions(positions, scell.get_cell(), pbc=scell.get_pbc(), wrap=True),scell.get_pbc())
                sc_spos = wrap_scaled_positions(scell.get_scaled_positions(wrap=True),scell.get_pbc())
                cl_idxs = get_cl_idx_sc(cl_spos,sc_spos,method=0)
                _cluster_orbit = cpool.get_cluster_orbit(scell, cl_idxs, cluster_species=cluster.get_nrs())
                cluster_orbit = _cluster_orbit.as_array()
                mult = _cluster_orbit.get_multiplicity_in_parent_lattice()
                cluster_orbits.append(cluster_orbit)

            self._scells.append(scell) # Add supercell to calculator
            self._cluster_orbits_set.append(cluster_orbits) # Add corresponding cluster orbits

        return cluster_orbits

    def get_cluster_orbit_pools_for_scell(self,scell):
        """Return array of cluster_pool objects, containing cluster orbits for a given supercell

        **Parameters**

        ``scell``: ParentLattice, SuperCell, or Structure object
            Object containing the lattice definition to determine the orbit
            of the clusters in the CorrelationsCalculator.
        """
        from clusterx.utils import get_cl_idx_sc
        from clusterx.super_cell import SuperCell
        from clusterx.structure import Structure
        from clusterx.symmetry import wrap_scaled_positions
        from clusterx.clusters.clusters_pool import ClustersPool
        
        cluster_orbit_pools = []

        if isinstance(scell,Structure):
            scell = scell.get_supercell()
        elif isinstance(scell,SuperCell):
            pass
        elif isinstance(scell,ParentLattice):
            scell = SuperCell(scell,[1,1,1])

        cpool = ClustersPool(scell.get_parent_lattice(),super_cell=scell)
        
        for icl,cluster in enumerate(self._cpool.get_cpool()):
            positions = cluster.get_positions()

            cl_spos = wrap_scaled_positions(get_scaled_positions(positions, scell.get_cell(), pbc=scell.get_pbc(), wrap=True),scell.get_pbc())
            sc_spos = wrap_scaled_positions(scell.get_scaled_positions(wrap=True),scell.get_pbc())
            cl_idxs = get_cl_idx_sc(cl_spos,sc_spos,method=0)

            _cluster_orbit_pool = cpool.get_cluster_orbit(scell, cl_idxs, cluster_species=cluster.get_nrs())
            cluster_orbit_pool = _cluster_orbit_pool.as_array()
            mult = _cluster_orbit_pool.get_multiplicity_in_parent_lattice()

            cluster_orbit_pools.append(cluster_orbit_pool)

        return cluster_orbit_pools

    
    def reset_mc(self, mc = False):
        #print("reset")
        self._mc = mc
        self._num_mc_calls = 0
        self._cluster_orbits_mc = None
        
    def get_cluster_correlations(self, structure, multiplicities=None):
        """Get cluster correlations for a structure
        **Parameters:**

        ``structure``: Structure object
            structure for which to calculate the correlations.
        ``mc``: Boolean
            Set to ``True`` when performing Monte-Carlo simulations, to use an
            optimized version of the method.
        ``multiplicities``: array of integers
            if None, the accumulated correlation functions are devided by the size
            of the cluster orbits, otherwise they are devided by the given
            multiplicities.

        .. todo::

            remove multiplicities option and always give intensive correlations.
        """
        #from clusterx.utils import get_cl_idx_sc
        cluster_orbits = None

        if self._mc and self._cluster_orbits_set != [] and self._num_mc_calls != 0:
            #cluster_orbits = self._cluster_orbits_set[0]
            cluster_orbits = self._cluster_orbits_mc
        else:
            cluster_orbits = self.get_cluster_orbits_for_scell(structure.get_supercell())
            if self._mc is True:
                self._num_mc_calls = 1
                self._cluster_orbits_mc = cluster_orbits

        correlations = np.zeros(len(self._cpool))
        for icl, cluster in enumerate(self._cpool.get_cpool()):
            cluster_orbit = cluster_orbits[icl]
            for cluster in cluster_orbit:
                cf = self.cluster_function(cluster, structure.sigmas, structure.ems)
                correlations[icl] += cf

            if multiplicities is None:
                correlations[icl] /= len(cluster_orbit)
            else:
                correlations[icl] /= multiplicities[icl]
        return np.around(correlations,decimals=12)

    def get_correlation_matrix(self, structures_set, outfile = None):
        """Return correlation matrix for a structures set.

        **Parameters:**

        ``structures_set``: StructuresSet object
            a 2D numpy matrix is returned. every row in the matrix corresponds to
            a structure in the ``StructuresSet`` object.
        """
        corrs = np.empty((len(structures_set),len(self._cpool)))
        for i,st in enumerate(structures_set):
            corrs[i] = self.get_cluster_correlations(st)

        if outfile is not None:
            f  = open(outfile,"w+")
            for covec in corrs:
                for co in covec:
                    f.write("%2.12f\t"%(co))
                f.write("\n")
            f.close()

        return corrs
