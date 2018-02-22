import numpy as np
import clusterx as c

class CorrelationCalculator():
    def __init__(self, parent_lattice, clusters_pool, super_cell= None, tool="corrdump", tpar = False, basis = "trigo"):
        """
        Correlation calculator object.

        Parameters::
            parent_lattice: a ParentLattice object
            clusters_pool: a ClustersPool object
            super_cell: SuperCell object. Set this for fast calculation of correlations in configurational samplings, e.g. MC.
            tool: any of corrdump or clusterx
        
        """
        self.tool = tool
        self.tpar = tpar
        self.basis = basis
        self.parlat = parent_lattice
        self.clpool = clusters_pool
        self.scell = super_cell
        if self.tool == "corrdump":
            self.parlat.serialize(fmt="ATAT") # This creates the file parlat.in
            self.clpool.serialize("atat") # This creates the file clusters.out
            """
            1. Create cwd/parlat.in and keep it for subsequent calculations with corrdump calculator.
            2. Create cwd/clusters.out files for parallel execution. -> unfortunately, corrdump reads only from clusters.out named file, hardcoded in corrdump.cc.
            Possible solution, create a folder for each process with a copy of parlat.in and the corresponding clusters.out.
               
            """
            #pass
        
    def set_tool(self, tool):
        self.tool = tool

    def get_tool(self):
        return self.tool

    def get_basis(self):
        return self.basis
    
    def get_correlations(self, structure):
        """
        Calculate cluster correlations for structrue.

        structure is an atoms object. 
        """
        if self.tool == "corrdump":
            from clusterx.correlation_calculators.corrdump_calculator import get_correlations as get_correlations_corrdump
            corrs = get_correlations_corrdump(structure, self.basis)

        if self.tool == "clusterx":
            from ase.utils.timing import Timer
            from ase.data import chemical_symbols as cs
            from ase import Atoms
            from ase.db.jsondb import JSONDatabase
            from clusterx.utils import isclose
            from clusterx.symmetry import get_spacegroup
            from subprocess import call
            import sys

            timer = Timer()
            rtol = 1e-3
            cld = self.clpool.get_clusters_dict()
            prim_cell = self.parlat.get_cell()

            call(["rm","-f","clusters.json"])
            atoms_db = JSONDatabase(filename="clusters.json") # For visualization
            sites = self.scell.get_sites()
            timer.start("spglib")
            scell_sg, scell_sym = get_spacegroup(self.scell.get_pristine(), tool="spglib")
            timer.stop()
            timer.write()
            print(scell_sg)
            print(len(scell_sym["rotations"]))
            sys.exit()
            #scell_sg, scell_sym = get_spacegroup(self.parlat.get_pristine(), tool="spglib")
            
            for kcl,icl in cld.items():
                #wrap original cluster positions
                chem = []
                for c in icl["site_basis"]:
                    chem.append(cs[c[1]])
                    
                atoms = Atoms(symbols=chem,positions=icl["positions_car"],cell=self.scell.get_cell(),pbc=self.scell.get_pbc())
                wrapped_pos = atoms.get_positions()
                wrapped_scaled_pos = atoms.get_scaled_positions() # these are the scaled positions referred to the super cell
                
                for r,t in zip(scell_sym['rotations'], scell_sym['translations']):
                    ts = np.tile(t,(len(wrapped_scaled_pos),1)).T
                    new_sca_pos = np.add(np.dot(r,wrapped_scaled_pos.T),ts).T
                    new_car_pos = np.dot(new_sca_pos,self.scell.get_cell()).tolist()
                    chem = []
                    for c in icl["site_basis"]:
                        chem.append(cs[c[1]])

                    atoms = Atoms(symbols=chem,positions=new_car_pos,cell=self.scell.get_cell(),pbc=self.scell.get_pbc())
                    new_car_pos = atoms.get_positions(wrap=True)
                    new_sca_pos = atoms.get_scaled_positions(wrap=True)


                    # Dummy species
                    chem = []
                    for i in range(self.scell.get_natoms()):
                        chem.append("H")

                    # Map cluster to supercell
                    for p,c in zip(new_car_pos,icl["site_basis"]):
                        for ir,r in enumerate(self.scell.get_positions()):
                            if isclose(r,p,rtol=1e-2):
                                chem[ir] = cs[sites[ir][c[1]+1]]

                    atoms = Atoms(symbols=chem,positions=self.scell.get_positions(),cell=self.scell.get_cell(),pbc=self.scell.get_pbc())
                    atoms_db.write(atoms)

                
            corrs = []

            
        return corrs
