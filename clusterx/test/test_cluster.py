# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from clusterx.clusters.cluster import Cluster

def test_cluster():
    """
    Test cluster class
    """
    try:
        c1 = Cluster([0,1,2,1],[13,14,15,13])
        repeated_atom_index_detected = False
    except:
        repeated_atom_index_detected = True

    try:
        c1 = Cluster([0,1,2,1],[13,14,15])
        is_consistent_length = False
    except:
        is_consistent_length = True

    try:
        c1 = Cluster([],[])
        can_create_empty = True
    except:
        can_create_empty = False

        
    c1 = Cluster([0,1,2,3],[13,14,15,16])
    c2 = Cluster([0,2,1,3],[13,15,14,16])
    eq1 = c1==c2

    c1 = Cluster([0,1,2,3],[13,14,15,16])
    c2 = Cluster([0,2,1,3],[13,14,15,16])
    eq2 = c1!=c2
    
    c1 = Cluster([0,1,2,3],[13,14,14,16])
    c2 = Cluster([0,2,1,3],[13,14,14,16])
    eq3 = c1==c2
    
    c1 = Cluster([],[])
    c2 = Cluster([0,2,1,3],[13,14,14,16])
    eq4 = c1!=c2
    
    c1 = Cluster([0,1,2,1],[13,14,15,14])
    c2 = Cluster([1,1,0,2],[14,14,13,15])
    eq5 = c1==c2

    c1 = Cluster([0,1,2],[13,14,15])
    c2 = Cluster([0,1,2,3],[13,14,15,16])
    eq6 = c1!=c2

    c1 = Cluster([0,1,2,1],[13,14,15,14])
    c2 = Cluster([1,0,2],[14,13,15])
    eq7 = c1!=c2
    
    c3 = Cluster([0,1,2],[13,14,15]) 
    c4 = Cluster([0,1,3],[13,14,15]) 
    clusters = [c1,c2]
    in1 = c3 in clusters
    in2 = c4 not in clusters
    print("end test inclusion")
    
    print ("\n\n========Test writes========")
    print (test_cluster.__doc__)
    print ("===========================\n")

    print ("========Asserts========")

    print("test is_multiple_species_in_site: ",repeated_atom_index_detected)
    assert repeated_atom_index_detected
    print("test is_consistent_length: ",is_consistent_length)
    assert is_consistent_length
    print("test can_create_empty: ",can_create_empty)
    assert can_create_empty
    print("test equalities:", eq1,eq2,eq3,eq4,eq5,eq6)
    assert eq1
    assert eq2
    assert eq3
    assert eq4
    assert eq5
    assert eq6
    assert eq7
    print("test array inclusion:", in1,in2)
    assert in1
    assert in2
