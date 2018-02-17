import numpy as np
from ase.build import bulk
from clusterx.parent_lattice import ParentLattice

def test_parent_lattice_creation():
    """Test creation of a parent lattice for a fictitious quaternary zincblende crystal.
    """
    atol = 1e-8

    cual = bulk('CuAl','zincblende',a=6.1)
    agal = bulk('AgAl','zincblende',a=6.1)
    sral = bulk('SrAl','zincblende',a=6.1)
    cuc = bulk('CuC','zincblende',a=6.1)

    pl = ParentLattice(atoms=cual,substitutions=[agal,sral,cuc])

    correct_pos = np.array([[ 0.  ,  0.  ,  0.  ], [ 0.25  ,  0.25 ,  0.25 ]])
    

    sites = pl.get_sites()
    idx_subs = pl.get_idx_subs()

    print ("\n\n========Test writes========")
    print (test_parent_lattice_creation.__doc__)
    print (sites)
    print (idx_subs)
    print ("===========================\n")

    print ("========Asserts========")
    
    assert np.allclose(pl.get_scaled_positions(), correct_pos, atol=atol)
