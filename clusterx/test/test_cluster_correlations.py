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

        ase gui test_clusters_correlations_structure_#.json
        ase gui test_clusters_correlations_cpool.json

    """

    cell = [[3,0,0],
            [0,1,0],
            [0,0,5]]
    positions = [
        [0,0,0],
        [1,0,0],
        [2,0,0]]
    pbc = [True,True,False]

    pri = Atoms(['H','H','H'], positions=positions, cell=cell, pbc=pbc)
    su1 = Atoms(['C','H','H'], positions=positions, cell=cell, pbc=pbc)
    su2 = Atoms(['H','He','H'], positions=positions, cell=cell, pbc=pbc)
    su3 = Atoms(['H','N','H'], positions=positions, cell=cell, pbc=pbc)

    plat = ParentLattice(pri,substitutions=[su1,su2,su3],pbc=pbc)
    cpool = ClustersPool(plat, npoints=[1,2], radii=[0,1.2])
    corrcal = CorrelationsCalculator("trigonometric", plat, cpool)

    scell1 = SuperCell(plat,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    structure1 = Structure(scell1,[1,1,1,6,7,1,1,2,1])
    corrs1 = corrcal.get_cluster_correlations(structure1)

    # Doubling of structure1. Correlations should not change.
    scell2 = SuperCell(plat,np.array([(1,0,0),(0,6,0),(0,0,1)]))
    structure2 = Structure(scell2,[1,1,1,6,7,1,1,2,1,1,1,1,6,7,1,1,2,1])
    corrs2 = corrcal.get_cluster_correlations(structure2)

    # Testing for Chebychev polinomials as single site basis functions.
    # Testing does not (yet) include multi - sublattices.

    su21 = Atoms(['C','C','C'], positions=positions, cell=cell, pbc=pbc)
    su22 = Atoms(['N','N','N'], positions=positions, cell=cell, pbc=pbc)

    plat2 = ParentLattice(pri,substitutions=[su21,su22],pbc=pbc)
    cpool2 = ClustersPool(plat2, npoints=[1,2], radii=[0,1.2])
    corrcal2 =  CorrelationsCalculator("chebyshev", plat2, cpool2)

    scell21 = SuperCell(plat2,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    structure21 = Structure(scell21,[1,1,1,6,7,1,6,7,1])
    corrs21 = corrcal2.get_cluster_correlations(structure21)

    # Doubling of structure21. Correlations should not change.
    scell22 = SuperCell(plat2,np.array([(1,0,0),(0,6,0),(0,0,1)]))
    structure22 = Structure(scell22,[1,1,1,6,7,1,6,7,1,1,1,1,6,7,1,6,7,1])
    corrs22 = corrcal2.get_cluster_correlations(structure22)

    # Binary array for Chebychev-polinomials.
    plat3 = ParentLattice(pri, substitutions = [su21], pbc = pbc)
    cpool3 = ClustersPool(plat3,  npoints=[1,2], radii=[0,1.2])
    corrcal3 =  CorrelationsCalculator("chebyshev", plat3, cpool3)


    corrcal4 = CorrelationsCalculator("chebyshev", plat2, cpool2)

    scell3 = SuperCell(plat3,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    structure3 = Structure(scell3,[1,1,1,6,6,1,6,6,1])
    corrs3 = corrcal3.get_cluster_correlations(structure3)

    print('\n',corrs21,'\n', corrs22)


    corrs4 = corrcal4.get_cluster_correlations(structure22)

    corrcal4.basis_set.print_basis_functions(2)

    #print('\n\nTime for evaluation of basis functions\nn', time() - t1)

    #t1 = time()
    """
    testbasis = PolynomialBasis(symmetric = True)
    for m in range(1,11):
        print("\n",m)
        testbasis.print_basis_functions(m)
    """

    corrcal_poly = CorrelationsCalculator("polynomial", plat, cpool)
    corrs_poly = corrcal_poly.get_cluster_correlations(structure1)

    def scalar_product_basis_set(function1, function2, alpha1, alpha2, M = 3, symmetric = False):
        """
        Function to test basis single site basis functions regarding their orthogonality.
        Expects that sigma in {0,1,...,M-1}, where sigma is an ising type discrete spin varaible and M is the number of species in an alloy.
        """
        scaling = 1 / M
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

    t = time.time()
    test_orthonormality(corrcal.site_basis_function, symmetric = False)
    print('Time for AfdW basis', time.time() - t)
    t = time.time()
    test_orthonormality(corrcal4.site_basis_function, symmetric = False)
    print('Time for Chebyshev basis', time.time() - t)

    print ("\n\n========Test writes========")
    print (test_cluster_correlations.__doc__)
    scell = cpool.get_cpool_scell()
    cpool.write_clusters_db(cpool.get_cpool(),scell,"test_cluster_correlations_cpool.json")
    structure1.serialize(fmt="json",fname="test_cluster_correlations_structure_1.json")
    structure2.serialize(fmt="json",fname="test_cluster_correlations_structure_2.json")

    scell = cpool2.get_cpool_scell()
    cpool2.write_clusters_db(cpool2.get_cpool(),scell,"test_cluster_correlations_cpool_chebyshev.json")
    structure21.serialize(fmt="json",fname="test_cluster_correlations_structure_1_chebyshev.json")
    structure22.serialize(fmt="json",fname="test_cluster_correlations_structure_2_chebyshev.json")

    print ("===========================\n")

    print ("========Asserts========")

    assert np.allclose([-0.33333333,0.,-0.,0.33333333,0.57735027,-0.33333333,-0.25,-0.,-0.25],corrs1,atol=1e-5)
    assert np.allclose([-0.33333333,0.,-0.,0.33333333,0.57735027,-0.33333333,-0.25,-0.,-0.25],corrs2,atol=1e-5)
    assert np.allclose([-0.40824829, 0.23570226, 0.16666667, -0.28867513, -0.16666667, 0.33333333, 0., 0.33333333],corrs21,atol=1e-5)
    assert np.allclose([-0.40824829, 0.23570226, 0.16666667, -0.28867513, -0.16666667, 0.33333333, 0., 0.33333333],corrs22,atol=1e-5)
    assert np.allclose([-0.11111111, 0.11111111, 0.11111111],corrs3,atol=1e-5)
