import numpy as np
import spglib
    
def get_spacegroup(parent_lattice):
    """
    Get space symmetry of a ParentLattice object. Sites allowing different substitutional species are treated as symetrically distinct.
    """

    atoms = parent_lattice.get_pristine().copy()

    # Create a fictitious atoms object with species set to site type.
    # This way, two sites which are symetrically equivalent in the pristine lattice
    # become inequivalent if those sites can be occupied by different species.
    atoms.set_atomic_numbers(parent_lattice.get_tags() + 1)
    
    sg = spglib.get_spacegroup(atoms)
    sym = spglib.get_symmetry(atoms)

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
    """
    Wrap scaled coordinates to the unit cell
    """
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
