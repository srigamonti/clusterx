from tempfile import NamedTemporaryFile
import subprocess
from subprocess import call
from clusterx.io.formats import write
import numpy as np
import clusterx as c

def get_spacegroup(atoms, tool):
    """
    Get space symmetry of an atoms object.

    tool: ase, atat or spglib
    """

    if tool == "atat":
        call(["rm","-f","sym.out"])
        tmpf = NamedTemporaryFile(mode="w",dir=".",delete=False)
        write(tmpf.name, atoms, fmt="atat")
        process = subprocess.Popen(["corrdump","-sym","-l="+tmpf.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        corr, err = process.communicate()
        tmpf.close()
        #call(["rm","-f",tmpf.name])
        #corrfile.write(corr)
        if err:
            print("Error (gen_correlations): corrdump ended with following error:\n")
            print(err)

        #call(["rm","-f","corrdump.log","sym.out"])

        #return np.array([ float(x) for x in corr.split() ])

    if tool == "ase":
        import ase
        sg = ase.spacegroup.get_spacegroup(atoms)
        print(sg)


    if tool == "spglib":
        import spglib
        sg = spglib.get_spacegroup(atoms)
        #print(sg)
        
        sym = spglib.get_symmetry(atoms)
        #print (sym)
        #print(sym['equivalent_atoms'])
        #print("Number of symmetry operations: ",len(sym['rotations']))
        #sym2 = [(r, t) for r, t in zip(sym['rotations'], sym['translations'])]
        #for s in sym2:
        #    print(s[0],s[1])

        return sg, sym


def get_scaled_positions(positions, cell, pbc=(True,True,True), wrap=True):
    """Get scaled positions.
    """
    from numpy.linalg import solve
    s = solve(cell.T,positions.T).T

    if wrap:
        s = wrap_scaled_positions(s, pbc)

    return s

def wrap_scaled_positions(s, pbc):
    s = np.around(s, decimals=8)
    for i, pbc in enumerate(pbc):
        if pbc:
            s[:, i] %= 1.0

    return s


def get_internal_translations(parent_lattice, super_cell):
    """
    Return the internal translations of a parent lattice with respect to a super cell.
    Translations are expressed in scaled coordinates with respect to the super cell.
    """
    from ase import Atoms
    from clusterx.super_cell import SuperCell
    from clusterx.parent_lattice import ParentLattice
    
    atoms0 = Atoms(symbols=['H'], positions=[(0,0,0)], cell=parent_lattice.get_cell(), pbc=parent_lattice.get_pbc())
    atoms1 = ParentLattice(atoms=atoms0,pbc=parent_lattice.get_pbc())
    atoms2 = SuperCell(atoms1, super_cell.get_transformation())
    return atoms2.get_scaled_positions(wrap=True)
