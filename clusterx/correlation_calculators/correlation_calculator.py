
class CorrelationCalculator():
    def __init__(self, parent_lattice, clusters_pool, tool="corrdump",tpar = False, basis = "trigo"):
        self.tool = tool
        self.tpar = tpar
        self.basis = basis
        self.parlat = parent_lattice
        self.clpool = clusters_pool
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
        if self.tool == "corrdump":
            from clusterx.correlation_calculators.corrdump_calculator import get_correlations as get_correlations_corrdump
            corrs = get_correlations_corrdump(structure, self.basis)
        
        return corrs
