# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import pytest
from clusterx.clusters.cluster import Cluster


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
