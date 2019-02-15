# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import ase.io.formats as aseio

def write(filename, atoms, fmt=None, parallel=True, **kwargs):
    if fmt == "atat":
        from clusterx.io.atat import write_atat
        write_atat(filename, atoms)
    else:
        aseio.write(filename, atoms, format, parallel, **kwargs)
