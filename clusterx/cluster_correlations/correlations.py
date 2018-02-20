import subprocess
from subprocess import call
import clusterx as c
import os.path
import sys
import numpy as np
from clusterx.structures_set import StructuresSet
from clusterx.clusters_set import ClustersSet
from tempfile import NamedTemporaryFile
from clusterx.io import write

def write_correlations(corrfunc="trigo", structures_set=None, clusters_set=None):
    #corrfile = open(path_corr_file,"w+")
    
    corrmat = []
    err = None
    tmpf = NamedTemporaryFile(mode="w",dir=".",delete=False)

    sgen = structures_set.select() # returns AtomsRow generator
    for structure in sgen:
        atoms = structure.toatoms()
        write("kkk",atoms,"xyz")
        tmpf = NamedTemporaryFile(mode="w",dir=".",delete=False)
        c.structure_convert("ASE","ATAT",args={"ase_str_obj":atoms,"atat_file":tmpf.name},virtual=True)
        tmpf.close()
        process = subprocess.Popen(["corrdump","-c","-l=parlat.in","-crf="+corrfunc,"-s="+tmpf.name,"-sig="+str(c.PRECISION)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        corr, err = process.communicate()
        call(["rm","-f",tmpf.name])
        #corrfile.write(corr)
        if err:
            print "Error (gen_correlations): In structure %d\n"%structure["id"]
            print err
        print structure
        
    #corrfile.close()
    
    call(["rm","-f","corrdump.log","sym.out"])



def get_correlations(atoms, clusters_set):


    tmpf = NamedTemporaryFile(mode="w",dir=".",delete=False)
    c.structure_convert("ASE","ATAT",args={"ase_str_obj":atoms,"atat_file":tmpf.name},virtual=True)
    tmpf.close()
    process = subprocess.Popen(["corrdump","-c","-l=parlat.in","-crf="+corrfunc,"-s="+tmpf.name,"-sig="+str(c.PRECISION)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    corr, err = process.communicate()
    call(["rm","-f",tmpf.name])
    corrfile.write(corr)
    if err:
        print "Error (gen_correlations): In structure %d\n"%structure["id"]
        print err
    print structure
    process = subprocess.Popen(["corrdump","-c","-s="+str(str_out_path),"-sig="+str(c.PRECISION)], stdout=subprocess.PIPE)
    corr, err = process.communicate()

    return [ float(x) for x in corr.split() ]
    
def parse_correlations(path_corr="allcorr.out"):
    corr_file=open(path_corr,"r")
    corr=[]
    for l in corr_file:
        lel=l.split()
        lcorr=[float(e) for e in lel]
        corr.append(lcorr)
    return np.array(corr)

 
def comatrix(comat, clulist,strlist = None):
    ns = np.shape(comat)[0]
    ncl = np.shape(comat)[1]

    com = []
    if strlist is None:
        for istr in range(ns):
            com.append([])
            for icl in range(ncl):
                if icl in clulist:
                    com[istr].append(comat[istr,icl])

    else:
        istri = -1
        for istr in range(ns):
            if istr in strlist:
                istri = istri + 1
                com.append([])
                for icl in range(ncl):
                    if icl in clulist:
                        com[istri].append(comat[istr,icl])

    com = np.array(com)
    
    return com


def get_correlations_pristine(iconc):
    assert int(iconc) == 0 or int(iconc) == 1, "Wrong argument in call to energy_ce_pristine." 

    npoints = c.get_num_points_clusters()
    numcl = c.get_num_clusters()

        
    
    corr = []
    if c.BASIS == "trigonometric":
        if int(iconc)==0:
            for i in range(numcl):
                corr.append(pow(-1,npoints[i]))

        if int(iconc)==1:
            for i in range(numcl):
                corr.append(1)

    if c.BASIS == "linear":
        if int(iconc)==0:
            for i in range(numcl):
                corr.append(0)
            corr[0] = 1
                
        if int(iconc)==1:
            for i in range(numcl):
                corr.append(1)
    return corr
