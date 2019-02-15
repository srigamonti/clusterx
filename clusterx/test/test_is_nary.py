# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from ase.build import bulk

def test_is_nary():
    """Test ParentLattice.is_nary() method
    """
    cu = bulk("Cu","fcc")
    au = bulk("Au","fcc")
    ag = bulk("Ag","fcc")

    plat = ParentLattice(atoms=cu, substitutions=[au,ag])

    scell = SuperCell(plat, [[2,2,-2],[2,-2,2],[-2,2,2]])

    s = scell.gen_random(nsubs={0:[10,3]})

    isok1 = (s.is_nary(3) is True) and (s.is_nary(4) is False)

    plat2 = ParentLattice(atoms=cu, site_symbols=[["Cu","Au","Ag","X"]])

    isok2 = (plat2.is_nary(2) is False) and (plat2.is_nary(4) is True)

    cuau = bulk("CuAu","zincblende",a=3.0)
    cuag = bulk("CuAg","zincblende",a=3.0)

    plat3 = ParentLattice(atoms=cuau, substitutions=[cuag])
    isok3 = (plat3.is_nary(2) is True) and (plat3.is_nary(4) is False)

    cuau = bulk("CuAu","zincblende",a=3.0)
    cuag = bulk("CuAg","zincblende",a=3.0)
    siau = bulk("SiAu","zincblende",a=3.0)

    plat4 = ParentLattice(atoms=cuau, substitutions=[cuag,siau])
    isok4 = (plat4.is_nary(2) is False) and (plat4.is_nary(3) is False) and (plat4.is_nary(4) is False)

    assert (isok1 and isok2 and isok3 and isok4)
