# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.



def write_atat_tmp(fileobj, atoms):
    cell = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()

    for cellv in cell:
        f.write("%2.12f\t%2.12f\t%2.12f\n"%(cellv[0],cellv[1],cellv[2]))

    f.write("1.000000000000\t0.000000000000\t0.000000000000\n")
    f.write("0.000000000000\t1.000000000000\t0.000000000000\n")
    f.write("0.000000000000\t0.000000000000\t1.000000000000\n")

    for pos, s in zip(positions, symbols):
        f.write("%2.12f\t%2.12f\t%2.12f\t%s\n"%(pos[0],pos[1],pos[2],s))

def write_atat(filename, atoms):
    cell = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()

    with open(filename,"w") as f:
        for cellv in cell:
            f.write("%2.12f\t%2.12f\t%2.12f\n"%(cellv[0],cellv[1],cellv[2]))

        f.write("1.000000000000\t0.000000000000\t0.000000000000\n")
        f.write("0.000000000000\t1.000000000000\t0.000000000000\n")
        f.write("0.000000000000\t0.000000000000\t1.000000000000\n")

        for pos, s in zip(positions, symbols):
            f.write("%2.12f\t%2.12f\t%2.12f\t%s\n"%(pos[0],pos[1],pos[2],s))
    
