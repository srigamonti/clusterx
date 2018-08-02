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

class StructuresSet():
    """
    **StructureSet class**

    Objects of this class contain a set of structures. This set can be used for
    various purposes, for instance as a training data set for
    cluster expansion, or as a validation set for cross validation.

    **Parameters:**

    ``parent_lattice``: ParentLattice object
        All the structures on a structures set must derive from the same parent
        lattice given here.

    ``filename``: String
        The structures in the data set are added to a json database
        (specifically an ASE's JSONDatabase object), which can be serialized
        to a json file with the path given here.

    **Examples:**

    **Methods:**
    """
    def __init__(self,parent_lattice, filename="structures_set.json", serial=False, calculator = None):

        self._iter = 0
        self._nstructures = 0
        self._filename = filename
        self._metadata = {}
        self._structures = []
        self._parent_lattice = parent_lattice
        self._props = None
        #self.write(parent_lattice, parent_lattice=True)
        #self.json_db = JSONDatabase.__init__(filename=self._filename)
        self.json_db = JSONDatabase(filename=self._filename)
        if isinstance(calculator,Calculator):
            self.set_calculator(calculator)

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        if self._iter < self._nstructures:
            i = self._iter
            self._iter += 1
            return self._structures[i]
        else :
            raise StopIteration

    def __len__(self):
        return self._nstructures

    def get_nstr(self):
        return self._nstructures

    def write(self, structure, key_value_pairs={}, data={}, **kwargs):
        self.json_db.write(structure,key_value_pairs, data={"tags":structure.get_tags(),"idx_subs":structure.get_idx_subs()},**kwargs)

    def add_structure(self,structure, key_value_pairs={}, write_to_db = False, **kwargs):
        """Add a structure to the StructuresSet

        **Parameters:**

        ``structure``: Structure object
            Structure object for the structure to be added.
        ``key_value_pairs``: dictionary
            if ``write_to_db`` is ``True`` (see below), then this argument is passed
            to ASE's ``json_db.write`` method.
        ``write_to_db``: boolean (default: False)
            Whether to add the structure to the json database (see ``filename``
            parameter for StructuresSet initialization)
        ``kwargs``: dictionary
            passed to ASE's ``json_db.write`` method if ``write_to_db`` is ``True``
        """
        self._structures.append(structure)
        self._nstructures += 1
        if write_to_db:
            self.json_db.write(structure,key_value_pairs, data={"tags":structure.get_tags(),"idx_subs":structure.get_idx_subs()},**kwargs)

    def get_structure(self,sid):
        """Get one structure of the set

        **Parameters:**

        ``sid``: integer
            index of structure in the structure set.

        **Returns:**

        Structure object.
        """
        return self._structures[sid]

    def get_structures(self):
        """Get all structures of the set

        **Return:**

        list of Structure objects.
        """
        return self._structures

    def get_structure_atoms(self, sid):
        """
        Return Atoms object for db row sid.
        """
        return self._structures[sid].get_atoms()

    def iterimages(self):
        # Allows trajectory to convert NEB into several images
        return iter(self._structures)

    def __getitem__(self, i=-1):
        return self._structures[i]

    def get_images(self,remove_vacancies=True,n=None):
        """
        Return array of Atoms objects from structures set.

        **Parameters:**

        ``remove_vacancies``: Boolean
            whether the returned Atoms objects contain vacancies, i.e. atoms with
            species number 0 or chemical symbol X. If true, vacancy sites are eliminated
            in the returned Atoms objects
        ``n``: integer
            return the first ``n`` structures. If ``None``, return all structures.
        """
        images = []
        if n is None:
            nmax = len(self)
        else:
            nmax = n
            
        if not remove_vacancies:
            for i in range(nmax):
                images.append(self._structures[i].get_atoms())
                #images.append(self._structures[i])
        else:
            for i in range(nmax):
                atoms0 = self._structures[i].get_atoms()
                positions = []
                numbers = []
                indices = []
                for i,atom in enumerate(atoms0):
                    nr = atom.get('number')
                    if nr != 0:
                        indices.append(i)
                        positions.append(atom.get('position'))
                        numbers.append(nr)
                images.append(Atoms(cell=atoms0.get_cell(), pbc=atoms0.get_pbc(),numbers=numbers,positions=positions))

        return images

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

    def get_property(self):
        return self._props

    def calculate_property(self, prop="energy"):
        """
        Return calculated property for all structures in the structures set.

        Fix needed: protected keys in ase db class impede to
        update the value of energy. Patching this now by using
        a different key, e.g. energy2 instead of energy.
        """
        calc = self.get_calculator()
        props = np.zeros(len(self))
        for i,st in enumerate(self):
            ats = st.get_atoms()
            ats.set_calculator(calc)
            props[i] = ats.get_potential_energy()
        self._props = props
        return props

        """
        for row in self.select():
            a = self.get_atoms(selection = row.id)
            a.set_calculator(calc)
            if prop == "energy":
                e = a.get_potential_energy()
                #self.update(row.id,energy2=e)
                self.update(row.id,energy2=e)

        """
