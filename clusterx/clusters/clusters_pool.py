from ase.db.jsondb import JSONDatabase
from ase.atoms import Atoms
import clusterx as c
import os
import sys
import subprocess
import numpy as np
import json

class ClustersPool():
    def __init__(self,parent_lattice, npoints=None, radii=None, tool="clusterx", name="_clusters_pool", filename=None):

        
        self._name = name
        self._filename = name+".json"
        #self._atoms_db = JSONDatabase(filename=self._filename) # For visualization
        self._parent_lattice = parent_lattice
        #self.write(parent_lattice, parent_lattice=True)
        self.clusters_dict = {}

        self._npoints = npoints
        self._radii = radii
        self._tool = tool
        
    def gen_clusters(self):
        if self._tool=="corrdump":
            self._gen_clusters_corrdump()
            self.clusters_dict = self._parse_clusters_out()
            subprocess.call(["rm","clusters.out"])
        if self._tool=="clusterx":
            self._gen_clusters_clusterx()

    def _gen_clusters_clusterx(self):
        from clusterx.symmetry import get_spacegroup
        from numpy import linalg as LA
        from clusterx.super_cell import SuperCell
        npoints = self._npoints
        radii = self._radii

        # Estimate minimal supercell containing largest cluster and create supercell
        rmax = np.amax(radii)
        pcell = self._parent_lattice.get_cell()
        l = LA.norm(pcell, axis=1) # Lengths of the cell vectors
        n = [int(n) for n in np.ceil(rmax/l)] # number of repetitions of unit cell along each lattice vector to contain largest cluster
        scell =  SuperCell(self._parent_lattice, np.diag(n))
        nsc = scell.get_natoms()
        sites = scell.get_sites()
        idx_subs = scell.get_idx_subs()
        tags = scell.get_tags()
        # Example for these arrays:
        #a1.get_chemical_symbols() -> [Si,Si,Si,Ba,Ba,Na]
        #a2.get_chemical_symbols() -> [Al,Al,Al,Ba,Ba,Na]
        #a3.get_chemical_symbols() -> [Si,Si,Si, X, X,Na]
        #a4.get_chemical_symbols() -> [Si,Si,Si,Sr,Sr,Na]
        #
        #Then::
        #
        #  tags -> [0,0,0,1,1,2]
        #  idx_subs -> {0: [14,13], 1: [56,0,38], 2:[11]}
        #  sites -> {0: [14,13], 1: [14,13], 2: [14,13], 3: [56,0,38], 4: [56,0,38], 5:[11]}

        # Get symmetry operations of the pristine supercell
        scell_sg, scell_sym = get_spacegroup(scell.get_pristine(), tool="spglib")

        # Get list of substitutional sites
        ss = []
        for i in range(nsc):
            if len(sites[i]) > 1:
               ss.append(i)

        # Get list of inequivalent positions
        ineq_pos = []
        for i1 in ss:
            for r,t in zip(scell_sym['rotations'], scell_sym['translations']):
                ts = np.tile(t,(len(wrapped_scaled_pos),1)).T
                new_sca_pos = np.add(np.dot(r,wrapped_scaled_pos.T),ts).T
                new_car_pos = np.dot(new_sca_pos,self.scell.get_cell()).tolist()

                
        for si1,ch1 in sites.items():
            print(si1,ch1)
            for fu1 in ch1:
                print(fu1)
            
    def serialize(self, fmt):
        if fmt == "atomsdb":
            self.gen_atoms_database()

        if fmt == "atat":
            self._write_clusters_out_corrdump()

    def get_clusters_dict(self):
        return self.clusters_dict

    def dump_clusters_dict(self):
        print( json.dumps(self.clusters_dict,indent=4))
            
    def get_clusters_array(self):
        cld = self.get_clusters_dict()
        cla = []
        for key,item in cld.items():
            cla.append([])
            cla[key].append(item["multiplicity"])
            cla[key].append(item["radius"])
            cla[key].append(item["npoints"])
            if item["npoints"] > 0:
                cla[key].append(item["positions_lat"])
                cla[key].append(item["site_basis"])
        return cla

    def get_cluster(self, cln):
        return self.clusters_dict[cln]

    def _write_clusters_out_corrdump(self):
        """
        Writes clusters.out array as needed by corrdump program.

        Notes: First converts the clusters dictionary to array and then serializes.
               Should be improved by serializing directly from the dictionary. 
        """
        cla = self.get_clusters_array()
        f = open("clusters.out","w")

        for cl in cla:
            if len(cl) > 3:
                for cll in cl[:-1]:
                    if isinstance(cll,list) and len(cll[0]) == 3:
                        for site in zip(cll,cl[-1]):
                            f.write( '{:f} {:f} {:f} {:d} {:d}\n'.format(site[0][0],site[0][1],site[0][2],site[1][0],site[1][1]))
                    else:
                        f.write(str(cll)+"\n")
            else:
                for cll in cl:
                    f.write(str(cll)+"\n")
            f.write("\n")
                    

        f.close()
        
    def _parse_clusters_out(self):
        """
        Converts clusters.out file as used by corrdump to cell's 
        cluster-dictionary object.
        """
        cla = self._parse_clusters_out_to_array()
        prim_cell = self._parent_lattice.get_cell()
            
        cld = {}
        for icl,cl in enumerate(cla):
            cld[icl] = {}
            cld[icl]["multiplicity"] = cl[0]
            cld[icl]["radius"] = cl[1]
            cld[icl]["npoints"] = cl[2]
            if cl[2]==0:
                cld[icl]["positions_lat"] = []
                cld[icl]["positions_car"] = []
                cld[icl]["site_basis"] = []
            else:
                cld[icl]["positions_lat"] = cl[3]
                cld[icl]["positions_car"] = np.dot(cl[3],prim_cell).tolist()
                cld[icl]["site_basis"] = cl[4]
                
        return cld

    def get_cluster_orbit(self, super_cell, cluster_sites, tol = 1e-3):
        """
        cluster_sites, array of atom indices referred to the super_cell.
        """
        from clusterx.symmetry import get_spacegroup
        from scipy.spatial.distance import cdist
        
        sc_sg, sc_sym = get_spacegroup(super_cell.get_pristine(), tool="spglib")
        
        spos = super_cell.get_scaled_positions(wrap=True)

        sp0 = np.array([spos[site] for site in cluster_sites]) # Original scalar positions

        orbit = []
        for r,t in zip(sc_sym['rotations'], sc_sym['translations']):
            ts = np.tile(t,(len(sp0),1)).T
            _sp1 = np.add(np.dot(r,sp0.T),ts).T # Apply rotation, then translation

            for i, periodic in enumerate(super_cell.pbc):
                if periodic: # Following ASE's Atoms.get_scaled_positions, we do this twice
                    _sp1[:, i] %= 1.0
                    _sp1[:, i] %= 1.0
 
            distances = cdist(_sp1, spos, metric='euclidean') # Evaluate all (scaled) distances between cluster points to scell sites
            _cl = np.argwhere(np.abs(distances) < tol)[:,1] # Extract indices when distance is less than tol

            if _cl not in orbit:
                orbit.append(_cl)

        return np.array(orbit)
                    
    def gen_atoms_database(self, scell, db_filename_noextension="clusters"):
        """
        Builds an ASE's json database object (self._atoms_db). Atoms items in 
        the built database are a representation of the clusters
        embedded in a supercell appropriate for visualization
        with ASE's gui.
        """
        from ase.data import chemical_symbols as cs
        from ase import Atoms
        from ase.db.jsondb import JSONDatabase
        from clusterx.utils import isclose
        from subprocess import call

        rtol = 1e-3
        cld = self.get_clusters_dict()
        prim_cell = self._parent_lattice.get_cell()

        call(["rm","-f",db_filename_noextension+".json"])
        atoms_db = JSONDatabase(filename=db_filename_noextension+".json") # For visualization
        sites = scell.get_sites()
        for kcl,icl in cld.items():

            #wrap cluster positions
            chem = []
            for c in icl["site_basis"]:
                chem.append(cs[c[1]])

            atoms = Atoms(symbols=chem,positions=icl["positions_car"],cell=scell.get_cell(),pbc=scell.get_pbc())
            atoms.wrap(center=[0.5,0.5,0.5])
            wrapped_pos = atoms.get_positions()

            # Dummy species
            chem = []
            for i in range(scell.get_natoms()):
                chem.append("H")

            # Map cluster to supercell
            #for p,c in zip(icl["positions_car"],icl["site_basis"]):
            for p,c in zip(wrapped_pos,icl["site_basis"]):
                for ir,r in enumerate(scell.get_positions()):
                    if isclose(r,p,rtol=1e-2):
                        chem[ir] = cs[sites[ir][c[1]+1]]

            atoms = Atoms(symbols=chem,positions=scell.get_positions(),cell=scell.get_cell(),pbc=scell.get_pbc())
            atoms_db.write(atoms)

        
        """
        # Old implementation
        from clusterx.super_cell import SuperCell
        from ase.data import chemical_symbols as cs

        cld = self.get_clusters_dict()
        sc = 5
        parent =  self._parent_lattice
        prim_cell = parent.get_cell()
        for kcl,icl in cld.items():
            # Obtain cartesian coordinates of the cluster
            if icl["npoints"]!=0:
                cllat = np.array( icl["positions_lat"] )
                clcar = np.dot(cllat,prim_cell)

            # Build supercell
            super_cell = SuperCell(parent,[[sc,0,0],[0,sc,0],[0,0,sc]])
            super_cell.wrap(center=[0,0,0])
            
            # Set chemical symbols in the supercell
            chem = super_cell.get_chemical_symbols()
            sites = super_cell.get_sites()
            for ir,r in enumerate(super_cell.get_positions()):
                chem[ir] = "H"
                if icl["npoints"]!=0:
                    for p,c in zip(clcar,icl["site_basis"]):
                        if np.linalg.norm(r-p) < 1e-2:
                            chem[ir] = cs[sites[ir][c[1]+1]]
            #print chem
            super_cell.set_chemical_symbols(chem)    

            #view(super_cell)
            self._atoms_db.write(super_cell)
        """

        
    def _parse_clusters_out_to_array(self):
        if not os.path.isfile("clusters.out"):
            sys.exit("Error (clusterx.parse_clusters_out): file clusters.out does not exist.")

        f = open("clusters.out","r")

        flines = f.read().splitlines()

        clusters = []
        ic = -1
        new = True
        il = 0

        #fmt = '%.'+str(c.PRECISION)+'f'

        for l in flines:

            ls = l.split()

            if not new and len(ls)>0:
                il = il + 1
                if il == 1:
                    clusters[ic].append(float(ls[0]))
                if il == 2:
                    clusters[ic].append(int(ls[0]))
                if il == 3:    
                    clusters[ic].append([])
                    clusters[ic].append([])
                if il >= 3:
                    clusters[ic][3].append([float(ls[0]),float(ls[1]),float(ls[2])])
                    clusters[ic][4].append([int(ls[3]),int(ls[4])])
                      
            if not new and len(ls)==0:
                new = True
                il = 0
                
            if len(ls)>0 and new:
                ic = ic + 1
                clusters.append([])
                clusters[ic].append(int(ls[0]))
                new = False

        #print np.array(clusters)
        #sys.exit()
        #print clusters
        return clusters
 


    def _gen_clusters_corrdump(self, latfile_path="parlat.in"):

        #if not os.path.isfile(latfile_path):
        #    sys.exit("Error (clusterx.gen_clusters): lattice file %s not present."%(latfile_path))

        self._parent_lattice.serialize(fmt="ATAT")
        cmdarr = ["corrdump","-clus","-sig="+str(c.PRECISION)]
        for np,r in zip(self._npoints,self._radii):
            if np<2:
                continue
            else:
                cmdarr.append("-%d=%f"%(int(np),float(r)))

        cmdarr.append("-l=%s"%(latfile_path))

        eps = 1e-5

        if c.T2D:
            latin = c.parse_lat()
            v11 = latin[0][0][0]
            v12 = latin[0][0][1]
            v13 = latin[0][0][2]
            v21 = latin[0][1][0]
            v22 = latin[0][1][1]
            v23 = latin[0][1][2]
            v31 = latin[0][2][0]
            v32 = latin[0][2][1]
            v33 = latin[0][2][2]
            l1 = np.sqrt( v11*v11 + v12*v12 )
            l2 = np.sqrt( v21*v21 + v22*v22 )


            if abs(v13) > eps or abs(v23) > eps:
                sys.exit("Error (clusterx.gen_clusters): the lattice vectors defined in lat.in do not correspond to a 2d lattice in the x-y plane.")

            if abs(v31) > eps or abs(v32) > eps:
                sys.exit("Error (clusterx.gen_clusters): in 2D mode, the third lattice vector in lat.in should have the form 0.00 0.00 z, with z a real number.")

            for at in latin[2][0:]:
                if at[0][2] < 0 or at[0][2] >= 1:
                    sys.exit("Error (clusterx.gen_clusters): in 2D mode, the z lattice coordinates of atoms are expected to lie in the interval [0,1].")


            """ 
            # This commented modification takes care of a situation in which the unit unit cell vectors defined in lat.in 
            # are such that clusters in the plane can be equivalent to out-of-plane clusters. It patches the situation
            # by making sure this symmetry is broken. However, as it is now, the modification is always applied, even
            # when this symmetry is not present in lat.in. This is 
            # not a desired behaviour, so for the moment  the user must take care of the lat.in not having this symmetry.

            latin_c = deepcopy(latin)

            # make sure clusters are generated in some xy plane by breaking this symmetry.

            latin_c[0][2][2] = (np.pi - 2.0) * max(l1,l2)
            factor = latin_c[0][2][2]/abs(v33)

            for i in range(len(latin_c[2])):
                latin_c[2][i][0][2] = latin_c[2][i][0][2]/factor 

            c.str_to_file(latin_c,"./lat2d.tmp.in")

            cmdarr.append("-l=lat2d.tmp.in")
            """

        process = subprocess.Popen(cmdarr, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        corr, err = process.communicate()


        if c.T2D:
            #latin = c.parse_lat()
            """
            z = []
            for at in latin[2][0:]:
                z.append(at[0][2])

            z0 = min(z)
            z1 = max(z)
            """
            clusters = c.parse_clusters_out()
            clusters_new = []

            zs = []
            nsubsites = len(latin[2])
            for i in range(nsubsites):
                if len(latin[2][i][1])==2:
                    zs.append(latin[2][i][0][2])

            z0 = min(zs) - 1e-4
            z1 = max(zs) + 1e-4

            for cl in clusters:
                add = True
                if len(cl) > 3:
                    for i in range(cl[2]):
                        if cl[3][i][2] > z1 or cl[3][i][2] < z0:

                            add = False
                            break

                if add:
                    clusters_new.append(cl)
                    """
            for cl in clusters:
                if len(cl) > 3:
                    n = cl[3][0][2] // 1
                    for i in range(3,3+cl[2]):
                        # the application of this factor is related to symmetry breaking commented above. 
                        #cl[3][i-3][2] = factor * (cl[3][i-3][2] - n)
                        cl[3][i-3][2] = cl[3][i-3][2] - n

            for cl in clusters:
                add = True
                if len(cl) > 3:
                    zcl = []
                    for i in range(3,3+cl[2]):
                        zcl.append(cl[3][i-3][2])

                    for i in range(len(zcl)):
                        for j in range(len(zcl)):
                            if abs(zcl[i]-zcl[j]) >= 1:
                                add = False
                                break
                        if not add:
                            break

                if add:
                    clusters_new.append(cl)
            """
                    subprocess.call(["rm","clusters.out"])
                    # the erased file here is related to symmetry breaking commented above. 
                    #subprocess.call(["rm","lat2d.tmp.in"])

            c.write_clusters_file(clusters_new,fname="clusters.out")

        subprocess.call(["rm","-f","sym.out"])

        return

