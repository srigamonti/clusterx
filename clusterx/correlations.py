from collections import Counter
import numpy as np
import sys
from clusterx.symmetry import get_scaled_positions
from clusterx.utils import PolynomialBasis

class CorrelationsCalculator():
    """
    Correlations calculator object.

    **Parameters:**

    ``basis``: string
        cluster basis to be used. Possible values are: ``binary-linear``, ``trigonometric``, ``polynomial``, and ``chebychev``.
    ``parent_lattice``: ParentLattice object
        the parent lattice of the cluster expansion.
    ``clusters_pool``: ClustersPool object
        The clusters pool to be used in the calculator
    """
    def __init__(self, basis, parent_lattice, clusters_pool):
        self.basis = basis
        self._plat = parent_lattice
        # For each supercell (with corresponding transformation matrix) a set of cluster orbit set is created
        self._scells = []
        self._cluster_orbits_set = []
        ####
        self._cpool = clusters_pool
        self._2pi = 2*np.pi
        if self.basis == 'polynomial':
            self.basis_set = PolynomialBasis()
        elif self.basis == 'chebychev':
            self.basis_set = PolynomialBasis(symmetric = True)


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
        if self.basis == "trigonometric":
            # Axel van de Walle, CALPHAD 33, 266 (2009)
            if alpha == 0:
                return 1

            elif alpha%2 != 0:
                return -np.cos(self._2pi*np.ceil(alpha/2.0)*sigma/m)

            else:
                return -np.sin(self._2pi*np.ceil(alpha/2.0)*sigma/m)

        if self.basis == "binary-linear":
            # Only for binary alloys. Allows for simple interpretation of cluster interactions.
            return sigma

        if self.basis == "polynomial":

            return self.basis_set.evaluate(alpha, sigma, m)

        if self.basis == "chebychev":
            # Method proposed by J.M. Sanchez, Physica 128A, 334-350 (1984).
            # Same results as for "trigonometric" in case of a binary.
            # WARNING: Forces that sigma = +-m, +-(m-1), ..., +- 1, (0), i.e. explicitly sigma = -1,0,1

            def _map_sigma(sigma, m):
                # Maps sigma = 0, 1, 2, ..., M-1 to -M/2 <= sigma <= M/2.
                shifted_sigma = int(sigma - int(m / 2))
                if (m % 2) == 0:
                    if shifted_sigma >= 0:
                        shifted_sigma += 1
                return shifted_sigma

            sigma = _map_sigma(sigma, m)

            return self.basis_set.evaluate(alpha, sigma, m)


    def cluster_function(self, cluster, structure_sigmas,ems):
        cluster_atomic_idxs = cluster.get_idxs()
        cluster_alphas = cluster.alphas
        cf = 1.0
        for cl_alpha, cl_idx in zip(cluster_alphas,cluster_atomic_idxs):
            cf *= self.site_basis_function(cl_alpha, structure_sigmas[cl_idx], ems[cl_idx])

        return cf

    def get_binary_random_structure_correlations(self,concentration):
        correlations = np.zeros(len(self._cpool))
        if self.basis == "binary-linear":
            for icl,cl in enumerate(self._cpool.get_cpool()):
                correlations[icl]=np.power(concentration,cl.npoints)

        return np.around(correlations,decimals=12)

    def get_cluster_correlations(self, structure, mc = False, multiplicities=None):
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
        from clusterx.utils import get_cl_idx_sc
        cluster_orbits = None
        if mc and self._cluster_orbits_set != []:
            cluster_orbits = self._cluster_orbits_set[0]
        elif not mc:
            for i, scell in enumerate(self._scells):
                if cluster_orbits is None:
                    if len(structure.get_positions()) == len(scell.get_positions()):
                        if np.allclose(structure.get_positions(),scell.get_positions(),atol=1e-3):
                            cluster_orbits = self._cluster_orbits_set[i]
                            break

        if cluster_orbits is None:
            # Add new super cell and calculate cluster orbits for it.
            cluster_orbits = []
            scell = structure.get_supercell()
            for icl,cluster in enumerate(self._cpool.get_cpool()):
                positions = cluster.get_positions()
                cl_spos = get_scaled_positions(positions, scell.get_cell(), pbc=scell.get_pbc(), wrap=True)
                sc_spos = structure.get_scaled_positions(wrap=True)
                cl_idxs = get_cl_idx_sc(cl_spos,sc_spos,method=1)

                cluster_orbit, mult = self._cpool.get_cluster_orbit(scell, cl_idxs, cluster_species=cluster.get_nrs())
                cluster_orbits.append(cluster_orbit)

            self._scells.append(scell) # Add supercell to calculator
            self._cluster_orbits_set.append(cluster_orbits) # Add corresponding cluster orbits

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
