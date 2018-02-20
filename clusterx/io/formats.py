import ase.io.formats as aseio

def write(filename, atoms, fmt=None, parallel=True, **kwargs):
    if fmt == "atat":
        from clusterx.io.atat import write_atat
        write_atat(filename, atoms)
    else:
        aseio.write(filename, atoms, format, parallel, **kwargs)
