from tempfile import NamedTemporaryFile
import subprocess
from subprocess import call
from clusterx.io.formats import write
import numpy as np
import clusterx as c

def get_correlations(atoms, basis):
    tmpf = NamedTemporaryFile(mode="w",dir=".",delete=False)
    #c.structure_convert("ASE","ATAT",args={"ase_str_obj":atoms,"atat_file":tmpf.name},virtual=True)
    write(tmpf.name, atoms, fmt="atat")
    process = subprocess.Popen(["corrdump","-c","-l=parlat.in","-crf="+basis,"-s="+tmpf.name,"-sig="+str(c.PRECISION)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    corr, err = process.communicate()
    tmpf.close()
    #call(["rm","-f",tmpf.name])
    #corrfile.write(corr)
    if err:
        print("Error (gen_correlations): corrdump ended with following error:\n")
        print(err)
        
    #call(["rm","-f","corrdump.log","sym.out"])

    return np.array([ float(x) for x in corr.split() ])

