

def get_correlations(atoms, clusters, basis):
    cell = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()

