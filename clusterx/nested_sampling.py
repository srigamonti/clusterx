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
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell
import clusterx.clusters
import clusterx.clusters.clusters_pool
from clusterx.clusters.clusters_pool import ClustersPool
import clusterx.clusters.cluster
from clusterx.clusters.cluster import Cluster
from clusterx.correlations import CorrelationsCalculator
from clusterx.structures_set import StructuresSet
import clusterx.parent_lattice
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell

def nested_sampling(nsub1=None, nsub2=None, nw=None,niter=None, nsteps=None, diagnostics=True, write_files=False, write_log=True, plot_outer_v_iter=False):
    """
    Description:
        Perform nested sampling

    Parameters:

    Parameters for nested sampling are number of substitutions for atom type 1 (nsub1),  atom type 2 (nsub2), number of walkers or active points (nw) 
    number of iterations (niter), and number of steps (nsteps)

    energy model needed for acceptance and rejection. This occurs in the the function ``eval_energy``.
    ``eval_energy`` function takes the energy dictonary and configuration and outputs the energy
     "corcE", "mult", and "ecisE" are all contained in the energy_dict

    ``sc_lat``: SuperCell object
        Supercell in which the sampling is performed.

    ``nsub1``: integer
        number of substitutional atoms for species 1 for which the compositional space is examined
    ``nsub2``: integer
        number of substitutional atoms for species 2 (for the binary phase, this is assumed to be zero) for which the compositional space is examined
    ``ns_settings``: dictonary
        The ns_settings dictonary contains three settings:
        ``nw``: integer
            number of walkers for sampling configuration space 
        ``niter``: integer
            number of iterations, where the outer energy is evaluated at each iteration 
        ``nsteps``: integer
            number of stochastic steps or walk length
    
    to-do: 
        make parallel, for now nprocs will be hard coded
    """

    # if len(argv) < 1:
    #     print("No settings defined, will instead use the default settings used in test/test_nested_sampling.py")
    #     sys.exit()

    ns_settings = {}
    ns_settings["walkers"] = nw
    ns_settings["iters"] = niter
    ns_settings["steps"] = nsteps
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
    xhistory, outer_e, xs, E_xs, total_ewalk, total_xwalk = sc_nested_sampling(ns_settings, energy_dict, Nsub1=nsub1, Nsub2=nsub2, lat=sc_lat, nprocs=1, alwaysclone=True, diagnostics=False)
    min_index, lowest_E = min(enumerate(total_ewalk), key=itemgetter(1)) # find new lowest-energy sample
    
    #plot energy history vs. iterations
    if plot_outer_v_iter==True:
        plot_sc_outer_vs_interation(outer_e, outfile_name="outer_vs_Iter.pdf")
   
    print("Nested sampling finished!")
    print("Energies walked", len(total_ewalk))
    print("number of iterations", len(outer_e))
    print("\nNested sampling finished and found this energy to be the lowest:", lowest_E)

    if write_log == True:
         write_summary(logfile, nsub1, nsub2, total_ewalk, outer_e, xhistory, lowest_E)


def sc_nested_sampling(ns_settings, energy_dict, Nsub1=None , Nsub2=None, lat=None, nprocs, alwaysclone=True, diagnostics=False):

    # todo: determine total number of subsititonal sites from the structure 
    Nsubs = Nsub1 
    Nw = ns_settings["walkers"]
    Niter = ns_settings["iters"]
    Nsteps = ns_settings["steps"]
    pool = Pool(processes=Nprocs)
    # zero out most variables
    E_xs = np.zeros(Nw, dtype=float)                            # dynamically updated list of total energies of walksers 
    Ehistory = np.zeros( (Niter, 1), dtype=float)               # history of "outermost" energies - this is the main output of NS with total energies 
    accept_ratio_history = np.zeros((Niter), dtype=float)       # history of MC acceptance ratios
    stepsize_history = np.zeros((Niter), dtype=float)           # history of MC step sizes
    age = np.random.rand(Nw,1)                                  # the 'age' variable is a dynamically updated list that stores the number of iterations since a configuration was cloned
    clone = np.zeros((Niter), dtype=bool)                       # history of decisions whether a new configuration was cloned randomly or just continued from the outermost one
    emove = np.zeros((Niter), dtype=float)                      # 
    Elim = float(Nsub1)*500.0                                   # set initial energy limit. needs to be high enough that everything is accessible, but too high a value leads to waste
    total_ewalk = [] 
    
    xs = StructuresSet(lat)                                                             # configurations walkers
    total_xwalk = StructuresSet(lat,filename="test_all_walked_NS_structures.json")      # total walked states during the number of walkers per niters 
    xhistory = StructuresSet(lat,filename="test_all_outer_NS_structures.json")          # history of "outermost" configurations 
    scell = SuperCell(lat,np.array([(1,0,0),(0,3,0),(0,0,1)]))
    for i in range(Nw):
        struc2 = lat.gen_random({0:[Nsub1]})
        # check if energy less than limit, if so store configuration                
        if  eval_energy(energy_dict, struc2) < Elim:
            xs.add_structure(struc2)
    E_xs[:] = [ eval_energy(energy_dict, xs[i]) for i in range(0, Nw) ]
    
    # outer NS loop
    # this is the main NS loop which descends through decreasing energy levels and collects the density of states.
    for i in range(0, Niter):
        ws = []
        outer, Elim = max(enumerate(E_xs[:]), key=itemgetter(1)) # find new outermost sample, i.e. the one with the highest energy
        if diagnostics == True:
            print("On iter number %s out of %s Niters" %(i, Niter))
            print("NS iteration %s energy of outermost walker %s is %s" %(i, outer, Elim ))
        
        Ehistory[i] = E_xs[outer] # store the just found highest energy in our history that we will return
        xhistory.add_structure(xs[outer])
        #
        # Decide  whether to clone a new sample to replace outer or continue the outermost sample with the new energy limit starting from oldest or outer (highest energy)
        # find the sample which has walked the most (called "age")
        oldest = max(enumerate(age), key=itemgetter(1))[0]
        if ( alwaysclone ) or ( outer == oldest ): # clone if we are either always cloning OR the outermost sample is also the oldest. 
            if Nw == 1:
                A = outer+1
            else:
                A = outer+random.randint(1, Nw-1)     
            source = (A%Nw)
            new_xs = StructuresSet(lat)
            for idx, strc in enumerate(xs):
                if idx == outer:
                    new_xs.add_structure(xs[source])
                else:
                    new_xs.add_structure(xs[idx])
            xs = new_xs
            E_xs[:] = [ eval_energy(energy_dict, xs[i]) for i in range(0, Nw) ]
            age[outer] = math.floor(age[source])+random.random() # the cloned sample copies the age of its source plus a random number (so that we can distinguish the two copies)
            clone[i] = True
        else:
            clone[i] = False
        #
        # select samples to move and store them in the list called ws
        ws.append( outer ) # the outermost sample 
        if( Nw >1) and (Nw > Nprocs):
            while len(ws) < Nprocs:
                tmp = random.randrange(len(E_xs))
                if tmp in ws: continue
                ws.append( tmp )
        
        if diagnostics == True:
            print("%s walkers (corresponding to the maximum number of procs) will be computed and the walk length will be divided by this number" %( Nprocs ))
            print("That is, we will do a parallel walk of %s over %s procs" %(float(Nsteps*Nsubs), Nprocs))
            print("So each walk will output %s energies" %(float(Nsteps*Nsubs)/float(Nprocs)))
            print("walk samples in walk-list %s" %(ws)) # walk samples in walk-list ws        
            print("Elist before walk:")
            for idx in range(len(E_xs)):
                print(E_xs[idx], end='')
            print("")
            print("Looking for something lower than: %s" %( Elim ))

        if (Nw > Nprocs):
            steps = int(float(Nsteps*Nsubs)/float(Nprocs))
        else:
            steps = int(float(Nsteps*Nsubs))

        accept_ratio = []
        # now walk the outer most sample to randomize 
        for n in ws:
            x, n_accepted, ewalk, xwalk = sc_walk(xs[n], lat, Elim, steps, energy_dict)
            accept_ratio.append( n_accepted )
            E_xs[n] = eval_energy(energy_dict, x) 
            new_xs = StructuresSet(lat)
            for idx, strc in enumerate(xs):
                if idx == n:
                    new_xs.add_structure( x )
                else:
                    new_xs.add_structure(xs[idx])
            xs = new_xs
            # save all walked energies and structures to find the lowest energy post factum 
            total_ewalk.extend(ewalk)
            for it, new_x in enumerate(xwalk):
                total_xwalk.add_structure(new_x)

            age[n] += 1

        if diagnostics == True:
            print("Walk ended with energy:", E_xs[n])
            print("number of steps accepted / number of steps taken (Nsubs*Nsteps)")
            for ratio in accept_ratio:
                print(ratio, end='')
        # save the acceptance rates of the walk for analysis of exploration
        accept_ratio_history[i] = np.mean(accept_ratio)

    if diagnostics == True:
        print(len(total_xwalk))
        print(len(total_ewalk))
        
    return xhistory, Ehistory, xs, E_xs, np.array(total_ewalk), total_xwalk

def sc_walk(x, lat, Elimit, steps, energy_dict):
    
    ewalk = np.zeros( (steps), dtype=float) 
    # initialise structure trajectory
    xwalk = StructuresSet(lat)
    xwalk.add_structure(x)
    ewalk[0] = eval_energy(energy_dict, x) 
    naccepted = 0
    for i in range(1, steps):
        ind1,ind2=x.swap_random_binary(0)
        Enew = eval_energy(energy_dict, x)
        xwalk.add_structure(x)
        ewalk[i] = Enew
        if Elimit < Enew:
            x.swap(0,ind1,ind2)
        else: 
            naccepted += 1

    return x, naccepted/float(steps), ewalk, xwalk


def eval_energy(energy_dict, x):

    corrs = energy_dict["corcE"].get_cluster_correlations(x,mc=True)
    #corrs = corcE.get_cluster_correlations(x,mc=True)

    multE = deepcopy(energy_dict["mult"])
    ecisE = deepcopy(energy_dict["ecis"])
    
    erg = 0
    for j in range(len(ecisE)):
        erg += multE[j] * ecisE[j] * corrs[j]
    
    return erg

def write_summary(logfile, nsub1, nsub2, total_ewalk, Ehistory, xhistory, lowest_E):

    logfile = open(logfile,"w+")
    logfile.write( (" %s\t%s\t%s\t%s\t%s\n" %( "nsub1", "nsub2", " ", "label", "total_energy")) )
    for idx, e in enumerate(total_ewalk):
        logfile.write( ("%s\t%s\t%s\t%s\n" %( '{: 0.2f}'.format(float(nsub1)), '{: 0.2f}'.format(float(nsub2)), "walked_random_structure", '{: 0.5f}'.format(e))) )
    for i in range(len(xhistory)):
        total_e = float(Ehistory[i])
        logfile.write( ("%s\t%s\t%s\t%s\n" %( '{: 0.2f}'.format(float(nsub1)), '{: 0.2f}'.format(float(nsub2)), "outer", '{: 0.5f}'.format(total_e)) ))    
    logfile.write( ("%s\t%s\t%s\t%s\n" %( '{: 0.2f}'.format(float(nsub1)), '{: 0.2f}'.format(float(nsub2)), "lowest", '{: 0.5f}'.format(lowest_E))) )
    logfile.close()

def plot_sc_outer_vs_interation(outer_e, outfile_name="outer_vs_Iter.pdf"):
    import os, sys
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    from matplotlib import rc, rcParams
    
    width=8.0  
    ticksize = int(width * 1.5)
    golden_ratio = (math.sqrt(5) - 1.0) / 2.0
    labelsize = int(width * 3)
    height = int(width * golden_ratio)

    fig = plt.figure(1, figsize=(width, height))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='x', labelsize=ticksize)
    ax.tick_params(axis='y', labelsize=ticksize)

    x = np.arange(len(outer_e))
    
    ax.scatter(x, outer_e, marker='o',  s=20, facecolors="blue", edgecolors='none', zorder=0, label=('Outer Configs from Nested Sampling')) #edgecolors='blue', linewidth=1.0, 

    ax.legend(loc='best', frameon=False, fontsize='medium')
    ax.set_xlabel("Iteration", color='black', size=labelsize * 0.8) #* 0.5) # fontname="Times New Roman")
    ax.set_ylabel("Formation Energy (eV/cation)", color='black', size=labelsize * 0.8) #* 0.5) # fontname="Times New Roman")
   
    axes = fig.gca()
    axes.set_title(ax.get_title(), size=width) #* 2)
    axes.set_xlabel(ax.get_xlabel(), size=labelsize * 0.8) #* 0.5) # fontname="Times New Roman")
    axes.set_ylabel(ax.get_ylabel(), size=labelsize * 0.8)

    fig.savefig(outfile_name, bbox_inches="tight", dpi=300)
    plt.close(fig)



if __name__ == '__main__':
    nested_sampling(sys.argv[1:])

