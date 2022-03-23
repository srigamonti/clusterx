# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import numpy as np
import spglib

def get_spacegroup(parent_lattice):
    """
    Get space symmetry of a ParentLattice object. 
    
    Sites allowing different substitutional species are treated as symetrically distinct.

    Uses ``spglib`` library.
    """

    atoms = parent_lattice.get_pristine().copy()
    
    # Break possible symmetries along non-periodic directions to get correct
    # symmetries with spglib.
    new_cell = []
    cell = atoms.get_cell()
    for i, bc in enumerate(atoms.get_pbc()):
        if not bc:
            new_cell.append(cell[i]*np.pi)
        else:
            new_cell.append(cell[i])

    atoms.set_cell(new_cell,scale_atoms=False)
    # Create a fictitious atoms object with species set to site type.
    # This way, two sites which are symetrically equivalent in the pristine lattice
    # become inequivalent if those sites can be occupied by different species.
    atoms.set_atomic_numbers(parent_lattice.get_tags() + 1)

    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    spglib_input = (lattice, positions, numbers)
    
    sg = spglib.get_spacegroup(spglib_input)
    sym = spglib.get_symmetry(spglib_input)

    # After finding symmetries, undo the previous re-scaling done to break
    # symmetries
    for it in range(len(sym['translations'])):
        for i, bc in enumerate(atoms.get_pbc()):
            if not bc:
                sym['translations'][it][i] = sym['translations'][it][i]*np.pi

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

    The function returns a numpy array containing three-dimensional vectors representing the 
    internal translation symmetries of the super cell.

    **Parameters:**
    ``parent_lattice``: ParentLattice instance
        The ParentLattice object
    
    ``super_cell``: SuperCell instance
        The SuperCell object. Must be a supercell of the parent lattice given as first argument. 
    """
    from ase import Atoms
    from clusterx.super_cell import SuperCell
    from clusterx.parent_lattice import ParentLattice

    atoms0 = Atoms(symbols=['H'], positions=[(0,0,0)], cell=parent_lattice.get_cell(), pbc=parent_lattice.get_pbc())
    atoms1 = ParentLattice(atoms=atoms0,pbc=parent_lattice.get_pbc())
    atoms2 = SuperCell(atoms1, super_cell.get_transformation())

    tra0 = atoms2.get_scaled_positions(wrap=True)
    tra1 = []
    for tr in tra0:
        include_tr = True
        for id_, bd in enumerate(parent_lattice.get_pbc()):
            if tr[id_] != 0 and not bd: # exclude translation vectors along non-periodic directions
                include_tr = False
                break

        if include_tr:
            tra1.append(tr)

    return np.array(tra1)
