# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import clusterx as c
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.structure import Structure
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.correlations import CorrelationsCalculator
from clusterx.utils import PolynomialBasis
from ase import Atoms
import numpy as np
import time

def test_cluster_correlations():
    """Test calculation of cluster correlations.

    After successful execution of the test, the generated structure and clusters pool may be visualized with the command::

        ase gui test_cluster_correlations_structure_#.json
        ase gui test_cluster_correlations_cpool.json

    """

    def scalar_product_basis_set(function1, function2, alpha1, alpha2, M = 3, symmetric = False, scaled = True):
        """
        Function to test basis single site basis functions regarding their orthogonality.
        Expects that sigma in {0,1,...,M-1}, where sigma is an ising type discrete spin varaible and M is the number of species in an alloy.
        """
        scaling = 1 / M if scaled else 1
        if symmetric:
            sigmas = [x for x in range(-int(M/2),int(M/2)+1)]
            if M%2 == 0:
                sigmas.remove(0)
        else:
            sigmas = [x for x in range(M)]
        scalar_product = 0
        for sigma in sigmas:
            scalar_product += function1(alpha1, sigma, M) * function2(alpha2, sigma, M)
        scalar_product = scalar_product * scaling
        return scalar_product

    def test_orthonormality(function, m=3, symmetric = True):
        print("\northonormality:\nj k <theta_j | theta_k>")
        for j in range(m):
            for k in range(m):
                print(j , k, round(scalar_product_basis_set(function, function, j, k , M = m, symmetric = symmetric),10))

    cell = [[3,0,0],
            [0,1,0],
            [0,0,5]]

    positions = [[0,0,0],
        [1,0,0],
        [2,0,0]]

    pbc = [True,True,False]

    pri = Atoms(['H','H','H'], positions=positions, cell=cell, pbc=pbc)
    su1 = Atoms(['C','H','H'], positions=positions, cell=cell, pbc=pbc)
    su2 = Atoms(['H','He','H'], positions=positions, cell=cell, pbc=pbc)
    su3 = Atoms(['H','N','H'], positions=positions, cell=cell, pbc=pbc)

    plat = ParentLattice(pri,substitutions=[su1,su2,su3],pbc=pbc)
    cpool = ClustersPool(plat, npoints=[1,2], radii=[0,1.2])
    corrcal_tri = CorrelationsCalculator("trigonometric", plat, cpool)
    corrcal_poly = CorrelationsCalculator("polynomial", plat, cpool)

    scell1 = SuperCell(plat,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    structure1 = Structure(scell1,[1,1,1,6,7,1,1,2,1])
    corrs1_tri = corrcal_tri.get_cluster_correlations(structure1)
    corrs1_poly = corrcal_poly.get_cluster_correlations(structure1)

    # Doubling of structure1. Correlations should not change.
    scell2 = SuperCell(plat,np.array([(1,0,0),(0,6,0),(0,0,1)]))
    structure2 = Structure(scell2,[1,1,1,6,7,1,1,2,1,1,1,1,6,7,1,1,2,1])
    corrs2_tri = corrcal_tri.get_cluster_correlations(structure2)
    corrs2_poly = corrcal_poly.get_cluster_correlations(structure2)

    t = time.time()
    test_orthonormality(corrcal_tri.site_basis_function, symmetric = False)
    print('Time for trigonometric basis', time.time() - t)
    t = time.time()
    test_orthonormality(corrcal_poly.site_basis_function, symmetric = False)
    print('Time for polynomial basis', time.time() - t)

    print('\nPolynomial basis functions (m=3):')
    corrcal_poly.basis_set.print_basis_functions(3)

    print("\nTest binary-linear basis")

    bin_plat = ParentLattice(pri, substitutions = [su1], pbc = pbc)
    bin_cpool = ClustersPool(bin_plat, npoints=[1,2], radii=[0,1.2])
    coorrcal_bin_lin = CorrelationsCalculator("binary-linear", bin_plat, bin_cpool)

    scell_bin = SuperCell(bin_plat,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    structure_bin = Structure(scell1,[1,1,1,6,1,1,6,1,1])
    corrs_bin = coorrcal_bin_lin.get_cluster_correlations(structure_bin)

    print ("\n\n========Test writes========")
    print (test_cluster_correlations.__doc__)
    scell = cpool.get_cpool_scell()
    cpool.write_clusters_db(cpool.get_cpool(),scell,"test_cluster_correlations_cpool.json")
    bin_cpool.write_clusters_db(bin_cpool.get_cpool(),scell_bin,"test_cluster_correlations_cpool_bin.json")
    structure1.serialize(fmt="json",fname="test_cluster_correlations_structure_1.json")
    structure2.serialize(fmt="json",fname="test_cluster_correlations_structure_2.json")
    structure_bin.serialize(fmt="json",fname="test_cluster_correlations_structure_bin.json")

    #print(corrs1_poly)
    #print(corrs1_tri)

    print ("========Asserts========")

    assert np.allclose([-0.33333333,0.,-0.,0.33333333,0.57735027,-0.33333333,-0.25,-0.,-0.25],corrs1_tri,atol=1e-5)
    assert np.allclose([-0.33333333,0.,-0.,0.33333333,0.57735027,-0.33333333,-0.25,-0.,-0.25],corrs2_tri,atol=1e-5)

    assert np.allclose([-0.33333333, 0., 0., 0.81649658, 0.47140452, -0.33333333, -0.5, -0., -0.5], corrs2_poly,atol=1e-5)
    assert np.allclose(corrs1_poly, corrs2_poly, atol=1e-5)

    assert np.allclose(corrs_bin, [0.6666666, 0.33333333], atol=1e-5)
