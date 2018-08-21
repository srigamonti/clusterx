import os, sys
import numpy as np
import math
import os.path
import json
import math
import sys
from operator import itemgetter
from copy import deepcopy
import random
from random import randrange
import multiprocessing
from multiprocessing import Pool
import numpy as np
from random import randint
#
from ase.data import atomic_numbers as cn
from ase import Atoms
from ase.io import read, write
from ase.spacegroup import crystal
#
import subprocess
import clusterx as c
import clusterx.nested_sampling as ns

from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
import clusterx.clusters
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.clusters.cluster import Cluster
from clusterx.correlations import CorrelationsCalculator

from clusterx.structures_set import StructuresSet
import clusterx.parent_lattice
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
from clusterx.correlations import CorrelationsCalculator

import clusterx.clusters.clusters_pool
import clusterx.clusters.cluster
from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.clusters.cluster import Cluster

from clusterx.utils import isclose

def main():

    plat, sc_lat, energy_dict, corcE = build_lattice_and_get_corr()

    diagnostics=True
    write_files=False
    write_log=True

    nsub1=16
    nsub2=0.0
    nw=1
    niter=5
    nsteps=2
    nprocs=1

    ns_settings = {}
    ns_settings["walkers"] = nw
    ns_settings["iters"] = niter
    ns_settings["steps"] = nsteps
    # Generate output
    print ("\n\n========Nested Sampling Test========")
    print("\nNow running single component nested_sampling with these settings:")
    print("Number of substitutional atoms: %s" % nsub1)
    print("Number of walkers: %s" % nw)
    print("Number of nested sampling iterations: %s" % niter)
    print("Number of Monte Carlo steps: %s" % nsteps)

    logfile = ("NS_Nw-%s_Niter-%s_Nsteps-%s.log" %(nw, niter, nsteps))
    outfile = ("NS_thermo_summary_Nw-%s_Niter-%s_Nsteps-%s.log" %(nw, niter, nsteps))

    found_lowest = False
    lowest_e = 10.0
    # now run the code
    xhistory, outer_e, xs, E_xs, total_ewalk, total_xwalk =  ns.sc_nested_sampling(ns_settings, energy_dict, corcE, nprocs, Nsub1=nsub1, Nsub2=nsub2, lat=sc_lat, alwaysclone=True, diagnostics=diagnostics)
    min_index, lowest_E = min(enumerate(total_ewalk), key=itemgetter(1)) # find new lowest-energy sample
    
    #plot energy history vs. iterations
    ns.plot_sc_outer_vs_interation(outer_e, outfile_name="outer_vs_Iter.pdf")
   
    print("Nested sampling finished!")
    print("Energies walked", len(total_ewalk))
    print("number of iterations", len(outer_e))
    print("\nNested sampling finished and found this energy to be the lowest:")
    print("Total:", lowest_E)

    if write_log == True:
         ns.write_summary(logfile, nsub1, nsub2, total_ewalk, outer_e, xhistory, lowest_E)

    ncount_new, lowest_new, outer_new = get_info_from_file( logfile )

    print ("========Asserts========")
    assert isclose(float(ncount_new), float(nsteps)*float(nsub1)*float(niter))
    assert isclose(len(outer_new), float(niter))
    assert isclose(len(lowest_new), 1)

    if ( float(ncount_new) - float(nsteps)*float(nsub1)*float(niter)) != 0.0:
        print("Should be Nsteps*Nsubs*Niters random energies walked")
        print("Error with the number of random energies walked")
    elif ( float(len(outer_new)) - float(niter)) != 0.0:
        print("Should be Niters number of outer energies")
        print("error with the number of Niter (outer) energies")
    elif float(len(lowest_new)) != 1.0:
        print("Should be one lowest energy found")
    else:
        print("\n Test of Nested-Sampling method in CELL was successful.\n\n")

def get_info_from_file(logfile):

    rf=open( (logfile), 'r' )
    rf.seek(0)
    outer = []
    lowest = []
    ncount = 0
    for line in rf:
        if "walked_random_structure" in line:
            ncount += 1
        elif "outer" in line:
            outer.append( line.strip().split()[2])
        elif "lowest" in line:
            lowest.append( line.strip().split()[2])

    return ncount, lowest, outer

def define_initial_lattice():

    a = 10.5148
    x = 0.185; y = 0.304; z = 0.116
    wyckoff = [
        (0, y, z), #24k
        (x, x, x), #16i
        (1/4., 0, 1/2.), #6c
        (1/4., 1/2., 0), #6d
        (0, 0 , 0) #2a
    ]

    # Build the parent lattice
    pri = crystal(['Si','Si','Si','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])
    sub = crystal(['Al','Al','Al','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])
    plat = ParentLattice(atoms=pri,substitutions=[sub])
    
    scellE = SuperCell(plat,[(1,0,0),(0,1,0),(0,0,1)])
        
    return plat, scellE


def get_model(plat):

    # Build clusters pool
    cpool = ClustersPool(plat)
    cpsc = cpool.get_cpool_scell()
    s = cn["Al"]
    cpool.add_cluster(Cluster([],[],cpsc))
    cpool.add_cluster(Cluster([0],[s],cpsc))
    cpool.add_cluster(Cluster([24],[s],cpsc))
    cpool.add_cluster(Cluster([40],[s],cpsc))
    cpool.add_cluster(Cluster([6,4],[s,s],cpsc))
    cpool.add_cluster(Cluster([37,32],[s,s],cpsc))
    cpool.add_cluster(Cluster([39,12],[s,s],cpsc))
    cpool.add_cluster(Cluster([16,43],[s,s],cpsc))
    cpool.add_cluster(Cluster([35,11],[s,s],cpsc))
    cpool.add_cluster(Cluster([39,30],[s,s],cpsc))
    cpool.add_cluster(Cluster([22,17],[s,s],cpsc))
    cpool.add_cluster(Cluster([35,42],[s,s],cpsc))
    cpool.add_cluster(Cluster([32,14],[s,s],cpsc))
    cpool.add_cluster(Cluster([11,10],[s,s],cpsc))
    cpool.add_cluster(Cluster([18,9],[s,s],cpsc))
    cpool.add_cluster(Cluster([18,43],[s,s],cpsc))

    # Energy
    cpoolE = cpool.get_subpool([0,1,2,3,4,5,6,7,9,15])
    ecisE = [
        -78407.3247588,
        47.164484875,
        47.1673476881,
        47.1569012692,
        0.00851281608144,
        0.0139835351147,
        0.0108175321899,
        0.0101521144776,
        0.00121744613474,
        0.000413664306204
    ]
    multE = [1,24,16,6,12,8,48,24,24,24]
    corcE = CorrelationsCalculator("binary-linear",plat,cpoolE)

    return multE, ecisE, corcE

def build_lattice_and_get_corr():
    a = 10.5148
    x = 0.185; y = 0.304; z = 0.116
    wyckoff = [
        (0, y, z), #24k
        (x, x, x), #16i
        (1/4., 0, 1/2.), #6c
        (1/4., 1/2., 0), #6d
        (0, 0 , 0) #2a
    ]

    # Build the parent lattice
    pri = crystal(['Si','Si','Si','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])
    sub = crystal(['Al','Al','Al','Ba','Ba'], wyckoff, spacegroup=223, cellpar=[a, a, a, 90, 90, 90])
    plat = ParentLattice(atoms=pri,substitutions=[sub])

    scellE = SuperCell(plat,[(1,0,0),(0,1,0),(0,0,1)])
    
    # Build clusters pool
    cpool = ClustersPool(plat)
    cpsc = cpool.get_cpool_scell()
    s = cn["Al"]
    cpool.add_cluster(Cluster([],[],cpsc))
    cpool.add_cluster(Cluster([0],[s],cpsc))
    cpool.add_cluster(Cluster([24],[s],cpsc))
    cpool.add_cluster(Cluster([40],[s],cpsc))
    cpool.add_cluster(Cluster([6,4],[s,s],cpsc))
    cpool.add_cluster(Cluster([37,32],[s,s],cpsc))
    cpool.add_cluster(Cluster([39,12],[s,s],cpsc))
    cpool.add_cluster(Cluster([16,43],[s,s],cpsc))
    cpool.add_cluster(Cluster([35,11],[s,s],cpsc))
    cpool.add_cluster(Cluster([39,30],[s,s],cpsc))
    cpool.add_cluster(Cluster([22,17],[s,s],cpsc))
    cpool.add_cluster(Cluster([35,42],[s,s],cpsc))
    cpool.add_cluster(Cluster([32,14],[s,s],cpsc))
    cpool.add_cluster(Cluster([11,10],[s,s],cpsc))
    cpool.add_cluster(Cluster([18,9],[s,s],cpsc))
    cpool.add_cluster(Cluster([18,43],[s,s],cpsc))
    #for cl in cpool._cpool:
    #    print("cl_idxs: ",cl.get_idxs())

    clarray=cpool.get_cpool_arrays()

    # Build clusters pool
    cpool = ClustersPool(plat)
    cpsc = cpool.get_cpool_scell()
    s = cn["Al"]
    cpool.add_cluster(Cluster([],[],cpsc))
    cpool.add_cluster(Cluster([0],[s],cpsc))
    cpool.add_cluster(Cluster([24],[s],cpsc))
    cpool.add_cluster(Cluster([40],[s],cpsc))
    cpool.add_cluster(Cluster([6,4],[s,s],cpsc))
    cpool.add_cluster(Cluster([37,32],[s,s],cpsc))
    cpool.add_cluster(Cluster([39,12],[s,s],cpsc))
    cpool.add_cluster(Cluster([16,43],[s,s],cpsc))
    cpool.add_cluster(Cluster([35,11],[s,s],cpsc))
    cpool.add_cluster(Cluster([39,30],[s,s],cpsc))
    cpool.add_cluster(Cluster([22,17],[s,s],cpsc))
    cpool.add_cluster(Cluster([35,42],[s,s],cpsc))
    cpool.add_cluster(Cluster([32,14],[s,s],cpsc))
    cpool.add_cluster(Cluster([11,10],[s,s],cpsc))
    cpool.add_cluster(Cluster([18,9],[s,s],cpsc))
    cpool.add_cluster(Cluster([18,43],[s,s],cpsc))

    # Energy
    cpoolE = cpool.get_subpool([0,1,2,3,4,5,6,7,9,15])
    ecisE = [
        -78407.3247588,
        47.164484875,
        47.1673476881,
        47.1569012692,
        0.00851281608144,
        0.0139835351147,
        0.0108175321899,
        0.0101521144776,
        0.00121744613474,
        0.000413664306204
    ]
    multE = [1,24,16,6,12,8,48,24,24,24]
    
    corcE = CorrelationsCalculator("binary-linear",plat,cpoolE)

    energy_dict = {}
    energy_dict["mult"] = deepcopy(multE)
    energy_dict["ecis"] = deepcopy(ecisE)

    print(multE)
    print(energy_dict["mult"])

    return plat, scellE, energy_dict, corcE

if __name__ == '__main__':
    main()



