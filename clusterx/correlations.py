from collections import Counter
import numpy as np
import sys
from clusterx.symmetry import get_scaled_positions

class CorrelationsCalculator():
    """
    Correlations calculator object.
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


    def site_basis_function(self, alpha, sigma, m):
        """
        Calculates the site basis function.
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
            # Only for binary alloys. Allows for simple interpretation of cluster interactions
            return sigma


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

    def get_cluster_correlations(self, structure, mc = False):
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
                cl_idxs = []
                for clp in cl_spos:
                    for idx, scp in enumerate(sc_spos):
                        if np.allclose(clp,scp,atol=1e-3):
                            cl_idxs.append(idx)

                cluster_orbit = self._cpool.get_cluster_orbit(scell, cl_idxs, cluster_species=cluster.get_nrs())
                cluster_orbits.append(cluster_orbit)

            self._scells.append(scell) # Add supercell to calculator
            self._cluster_orbits_set.append(cluster_orbits) # Add corresponding cluster orbits

        correlations = np.zeros(len(self._cpool))
        for icl, cluster in enumerate(self._cpool.get_cpool()):
            cluster_orbit = cluster_orbits[icl]
            for cluster in cluster_orbit:
                cf = self.cluster_function(cluster, structure.sigmas, structure.ems)
                correlations[icl] += cf
            correlations[icl] /= len(cluster_orbit)

        return np.around(correlations,decimals=12)


    def get_correlation_matrix(self, structrues_set):
        corrs = np.empty((len(structrues_set),len(self._cpool)))
        for i,st in enumerate(structrues_set):
            corrs[i] = self.get_cluster_correlations(st)

        return corrs
