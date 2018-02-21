from ase import Atoms
import clusterx as c
import numpy as np
import sys
from ase.visualize import view
from ase.calculators.calculator import Calculator
from ase.db.jsondb import JSONDatabase
from clusterx.super_cell import SuperCell
from clusterx.parent_lattice import ParentLattice
from ase.db.core import connect
from ase.db.core import Database
import inspect

class StructuresSet(JSONDatabase):
    def __init__(self,parent_lattice, name=None, filename=None, create_indices=True,
                 use_lock_file=False, serial=False, calculator = None):

        self._name = name
        filename = name+".json"
        super(StructuresSet,self).__init__(filename=filename)
        self._metadata = {}
        self._parent_lattice = parent_lattice
        self.write(parent_lattice, parent_lattice=True)
        if isinstance(calculator,Calculator):
            self.set_calculator(calculator)
        
    def write(self, super_cell, key_value_pairs={}, data={}, **kwargs):
        super(StructuresSet,self).write(super_cell,key_value_pairs,
                                        data={"tags":super_cell.get_tags(),"idx_subs":super_cell.get_idx_subs()},**kwargs)

    def get_structure(self, sid):
        """
        Return Atoms object for db row sid.
        """
        return self.get_atoms(id=sid)

        
    def get_json_string(self, super_cell):
        fn = self.filename

        #reset filename and stdout
        self.filename = None
        from cStringIO import StringIO
        import sys
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        self.write(super_cell)
        
        #restore filename and stdout
        sys.stdout = old_stdout
        self.filename = fn
        
        return  mystdout.getvalue()
        """
        #should try something like this, from ase jsondb.py
        bigdct, ids, nextid = self._read_json()
        self._metadata = bigdct.get('metadata', {})

        """

    
    def set_calculator(self, calc):
        """
        for row in self.select():
            self.update(row.id, calculator=calculator_name)
        """
        """
        ids = [row.id for row in self.select()]
        self.update(ids, cell_calculator = calculator_name)
        """
        #self.metadata = {"calculator" : calc.name.lower(), "calculator_parameters" : calc.todict()}
        self._calculator = calc 
        self.set_metadata({"calculator" : calc.name.lower(),
                           "calculator_parameters" : calc.todict()})

    def get_calculator(self):
        return self._calculator

    def set_metadata(self, dct):
        self._metadata.update(dct)
        self.metadata = self._metadata

    def get_metadata(self, dct):
        return self.metadata

    def calculate_property(self, prop):
        """
        Return calculated property for all structures in the structures set.

        Fix needed: protected keys in ase db class impede to 
        update the value of energy. Patching this now by using 
        a different key, e.g. energy2 instead of energy.
        """
        calc = self.get_calculator()
        for row in self.select():
            a = self.get_atoms(selection = row.id)
            a.set_calculator(calc)
            if prop == "energy":
                e = a.get_potential_energy()
                #self.update(row.id,energy2=e)
                self.update(row.id,energy2=e)


                