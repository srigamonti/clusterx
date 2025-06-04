# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import pytest
import numpy as np

from clusterx.clusters.cluster import Cluster
from clusterx.test.defaults import get_clathrate_supercell


cluster_list = [
    Cluster([0,1,2],[13,14,15]),
    Cluster([0],[13])
]


@pytest.mark.xfail(reason="incomplete order relation")
@pytest.mark.parametrize("a", cluster_list)
def test_order_reflexive(a):
    assert a <= a, "Reflexive property of order relation failed"


@pytest.mark.xfail(reason="incomplete order relation")
@pytest.mark.parametrize("a", cluster_list)
def test_order_strict_irreflexive(a):
    assert not a < a, "Irreflexive property of strict order relation failed"


@pytest.mark.xfail(reason="incomplete order relation")
@pytest.mark.parametrize("a", cluster_list)
@pytest.mark.parametrize("b", cluster_list)
@pytest.mark.parametrize("c", cluster_list)
def test_order_transitive(a, b, c):
    if a <= b and b <= c:
        assert a <= c, "Transitive property of order relation failed"


@pytest.mark.xfail(reason="incomplete order relation")
@pytest.mark.parametrize("a", cluster_list)
@pytest.mark.parametrize("b", cluster_list)
def test_order_strict_transitive(a, b, c):
    if a < b and b < c:
        assert a < c, "Transitive property of strict order relation failed"


@pytest.mark.xfail(reason="incomplete order relation")
@pytest.mark.parametrize("a", cluster_list)
@pytest.mark.parametrize("b", cluster_list)
def test_order_antisymmetric(a, b):
    if a <= b and b <= a:
        assert a == b, "Antisymmetric property of order relation failed"


@pytest.mark.xfail(reason="incomplete order relation")
@pytest.mark.parametrize("a", cluster_list)
@pytest.mark.parametrize("b", cluster_list)
def test_order_strict_asymmetric(a, b):
    if a < b:
        assert not b < a, "Asymmetric property of strict order relation failed"


@pytest.mark.xfail(reason="incomplete order relation")
@pytest.mark.parametrize("a", cluster_list)
@pytest.mark.parametrize("b", cluster_list)
def test_order_totality(a, b):
    assert a <= b or b <= a, "Totality property of order relation failed"


@pytest.mark.xfail(reason="incomplete order relation")
@pytest.mark.parametrize("a", cluster_list)
@pytest.mark.parametrize("b", cluster_list)
def test_order_strict_totality(a, b):
    if a != b:
        assert a < b or b < a, "Totality property of strict order relation failed"


def test_init_supercell():
    """Test initialization of Cluster with SuperCell"""
    sc = get_clathrate_supercell()
    c = Cluster([0,1],[13,14], super_cell=sc)
    np.testing.assert_array_equal(c.get_alphas(), [1, 0])
    assert c.get_radius() > 0.0, "Cluster radius should be greater than zero"
    c.radius = None  # Reset radius to test if it is recalculated
    distances = np.array([[0.0, 2.0], [2.0, 0.0]])
    assert c.get_radius(distances) == 2.0, "Cluster radius should 2.0 when distances is set manually"


def test_init_index():
    """Test initialization of Cluster without SuperCell"""
    c = Cluster([0,1,2,3],[13,14,15,16])
    np.testing.assert_array_equal(c.ais, np.array([0,1,2,3]))
    np.testing.assert_array_equal(c.ans, np.array([13,14,15,16]))
    assert c.npoints == 4
    assert c.positions_cartesian is None
    assert c.alphas is None
    assert c.radius is None
    assert c.myhash == hash(str(list(zip(c.ais,c.ans)))), "Hash not equal to expected value"


def test_equal():
    c1 = Cluster([0,1,2,3],[13,14,15,16])
    c2 = Cluster([0,2,1,3],[13,15,14,16])
    assert c1 == c2, "Permutation invariance with index and number not given"

    c1 = Cluster([0,1,2,3],[13,14,14,16])
    c2 = Cluster([0,2,1,3],[13,14,14,16])
    assert c1 == c2, "Permutation invariance with index and same number not given"

    c1 = Cluster([0,1,2,1],[13,14,15,14])
    c2 = Cluster([1,1,0,2],[14,14,13,15])
    assert c1 == c2, "Permutation invariance with index and number not given"


def test_not_equal():
    c1 = Cluster([0,1,2,3],[13,14,15,16])
    c2 = Cluster([0,2,1,3],[13,14,15,16])
    assert c1 != c2, "False permutation invariance wrt index"

    c1 = Cluster([],[])
    c2 = Cluster([0,2,1,3],[13,14,14,16])
    assert c1 != c2, "False equivalence with empty cluster"
    
    c1 = Cluster([0,1,2],[13,14,15])
    c2 = Cluster([0,1,2,3],[13,14,15,16])
    assert c1 != c2, "False equivalence with different number of sites"

    c1 = Cluster([0,1,2,1],[13,14,15,14])
    c2 = Cluster([1,0,2],[14,13,15])
    assert c1 != c2, "False equivalence with different number of sites and number of species"


def test_exceptions():
    with pytest.raises(ValueError, match="Initialization error, number of sites in cluster different from number of species."):
        Cluster([0],[13,14])
    Cluster([0,1,2,1],[13,14,15,13])


def test_empty():
    Cluster([],[])


def test_inclusion():
    """
    Test inclusion of clusters in a list
    """
    c1 = Cluster([0,1,2,1],[13,14,15,14])
    c2 = Cluster([1,0,2],[14,13,15])
    c3 = Cluster([0,1,2],[13,14,15]) 
    c4 = Cluster([0,1,3],[13,14,15]) 
    clusters = [c1,c2]
    assert c3 in clusters, "Equivalent cluster should be found in cluster list"
    assert c4 not in clusters, "Non-equivalent cluster should not be found in cluster list"


def test_repeated_atom_index():
    Cluster([0,1,2,1],[13,14,15,13]) # TODO: is this supposed to raise an error?
