from clusterx.super_cell import SuperCell

class Structure(SuperCell):
    def __init__(self, super_cell, decoration):
        self.scell = super_cell
        self.decor = decoration
        super(Structure,self).__init__(super_cell.get_parent_lattice(),super_cell.get_transformation())
        self.set_atomic_numbers(self.decor)

    def get_supercell(self):
        return self.scell
        
    def serialize(self, fmt="json", tmp=False, fname=None):
        from ase.io import write
            
        if fname is None:
            fname = "structure.json"

        write(fname,images=self,format=fmt)

        self._fname = fname
