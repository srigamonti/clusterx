from ase import Atoms
import clusterx as c
import numpy as np
import sys
from ase.visualize import view
from ase.calculators.calculator import Calculator
from ase.db.jsondb import JSONDatabase
from clusterx.super_cell import SuperCell
from clusterx.parent_lattice import ParentLattice
from clusterx.structure import Structure
from clusterx.utils import calculate_trafo_matrix
from ase.db.core import connect
from ase.db.core import Database
import inspect
try:
    import cPickle as pickle
except ImportError:
    import pickle

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
        The structures in the data set can be added to a database
        (specifically an ASE's JSONDatabase object), contained in a json file
        with the path given by ``filename``.

    **Examples:**

    .. todo::

        * Write function write_files (already drafted).
        * So far, the class allows to have only a subset of the structures written to the member json database. This may be dengerous in certain contexts, analyze whether changing this.


    **Methods:**
    """
    def __init__(self, parent_lattice, filename="structures_set.json", calculator = None):

        self._iter = 0
        self._nstructures = 0
        self._filename = filename
        self._metadata = {}
        self._structures = []
        self._parent_lattice = parent_lattice
        self._props = {}
        self._folders = []
        self._folders_db = None
        self._folders_db_fname = None
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
        """Return number of structures in the structures set.
        """
        return self._nstructures

    def write(self, structure, key_value_pairs={}, **kwargs):
        self.json_db.write(structure.get_atoms(),key_value_pairs, data={"pcell":structure.get_parent_lattice().get_cell(),"tmat":structure.get_transformation(), "tags":structure.get_tags(),"idx_subs":structure.get_idx_subs()},**kwargs)

    def add_structure(self,structure, key_value_pairs={}, write_to_db = False, **kwargs):
        """Add a structure to the StructuresSet object

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
            self.write(structure, key_value_pairs, **kwargs)
            #self.json_db.write(structure.get_atoms(),key_value_pairs, data={"tmat":structure.get_transformation(),"tags":structure.get_tags(),"idx_subs":structure.get_idx_subs()},**kwargs)

    def add_structures(self, structures = None, json_db_filepath = None):
        """Add structures to the StructureSet object

        **Parameters:**

        ``structures``: list of Structure objects
            Structures to be added

        ``json_db_filepath``: path to JSON file
            Json database file. The database must contain a data dict with keys
            tags and idx_subs
        """
        if structures is not None:
            for structure in structures:
                self.add_structure(structure)

        elif json_db_filepath is not None:
            db = connect(json_db_filepath)
            for row in db.select():
                atoms = row.toatoms()
                tmat = row.get("data",{}).get("tmat")
                if tmat is None:
                    tmat = calculate_trafo_matrix(self._parent_lattice.get_cell(),atoms.get_cell())
                scell = SuperCell(self._parent_lattice,tmat)
                self.add_structure(Structure(scell, decoration=atoms.get_atomic_numbers()))

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

    def get_parent_lattice(self):
        return self._parent_lattice

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

    def calculate_energies(self, calculator, structure_fname="geometry.json"):
        """
        Perform ab-initio calculation of energies contained in folders.
        """
        import os
        from ase.io import read
        from ase.calculators.emt import EMT

        cwd = os.getcwd()

        for folder in self._folders:
            os.chdir(folder)
            atoms = read(structure_fname)
            atoms.set_calculator(calculator)
            erg = atoms.get_potential_energy()
            f = open(os.path.join("energy.dat"),"w+")
            f.write(str(erg))
            f.close()
            os.chdir(cwd)


    def write_files(self, root = ".", prefix = '', suffix = '', fnames = None, formats = [], overwrite = True):
        """Create folders containing structure files for ab-initio calculations.

        Structure files are written to files with path::

            [[root] /] [prefix] id [suffix] / [filename]

        Where ``root``, ``prefix``, ``suffix``, and ``filename`` are explained
        below, and ``id`` + 1 is the structure id in a created JSON database with
        path::

            [[root] /] [prefix]id0-idN[suffix] . json

        where ``id0`` and ``idN`` are the smallest and largest ``id`` indices.
        The path of the created folders are stored in the json-database created
        by ``StructuresSet.write_files()``, under, for instance::

            {
            ...
            "key_value_pairs": {"folder": "./random_strs-14"},
            ...
            }


        Parameters:

        ``root``: String
            path to the root folder containing the set of created folders

        ``prefix``: String
            prefix for name of folder containing the files

        ``suffix``: String
            suffix for name of folder containing the files

        ``fnames``: array of Strings
            Array of file names for files contaning the structure. If not set defaults
            to ``geometry.json``.

        ``formats``: array of Strings, optional
            Array of file formats corresponding to the file names in ``fnames``. Possible
            formats are listed in `ase.io.write <https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.write>`_.
            If entirely ommited, or if an element of the array is ``None``, the format is
            guessed from the corresponding file name.

        ``overwrite``: boolean
            Whether to overrite content of existing folders.
        """
        import os
        from ase.io import write

        path = os.path.join(root,prefix+"0"+"-"+str(self.get_nstr()-1)+suffix+".json")
        self._folders_db_fname = path
        db = connect(path, type = "json", append = not overwrite)

        if overwrite:
            self._folders = []

        if fnames is None:
            fnames = ["geometry.json"]
        elif isinstance(fnames,str):
            fnames = [fnames]

        if len(formats) == 0:
            for i in range(len(fnames)):
                formats.append(None)

        for i in range(self.get_nstr()):
            path = os.path.join(root,prefix+str(i)+suffix)
            if not os.path.exists(path):
                os.makedirs(path)

            if i < len(self._folders):
                self._folders[i] = path
            else:
                self._folders.append(path)

            atoms = self.get_structure(i).get_atoms()

            for fname, format in zip(fnames,formats):
                path = os.path.join(root,prefix+str(i)+suffix,fname)
                if overwrite:
                    write(path, atoms, format)
                elif not os.path.isfile(path):
                    write(path, atoms, format)

            if overwrite:
                db.write(atoms, folder=self._folders[i])
            elif not os.path.isfile(path):
                db.write(atoms, folder=self._folders[i])

        self._folders_db = db
        db.metadata = {'db_path':path}


    def get_folders(self):
        """Get list folders containing structure files for ab-initio calculations
        as created by ``StructureSet.write_files()``
        """
        return self._folders

    def get_folders_db(self):
        """Get json database object corresponding to the list folders
        containing structure files for ab-initio calculations
        as created by ``StructureSet.writ_files()``
        """
        return self._folders_db

    def get_folders_db_fname(self):
        """Get file name of json database corresponding to the list folders
        containing structure files for ab-initio calculations
        as created by ``StructureSet.writ_files()``
        """
        return self._folders_db_fname

    def read_energy(folder,**kwargs):
        """Read value stored in ``energy.dat`` file.

        This is to be used as the ``read_property`` keyword argument of
        ``StructureSet.read_property_values()`` method. Can be used as a
        template for reading different properties to passed to ``StructureSet.read_property_values()``.

        **Parameters:**

        ``folder``: string
            absolute or relative path of the folder containing the file/s to be read.

        ``**kwargs``: keyword arguments
            Extra arguments needed for the property reading. See documentation of
            ``StructureSet.read_property_values()``.
        """
        import os
        f = open(os.path.join(folder,"energy.dat"),"r")
        erg = float(f.readlines()[0])
        return erg

    def read_property_values(self, property_name = "total_energy", read_property = read_energy, **kwargs):
        """Read calculated property values from ab-inito output files

        Read property values from ab-initio code output files. These files are
        contained in paths::

            [[root] /] [prefix] id [suffix] /

        as created by ``StructureSet.write_files()``. The folders to be searched
        for energy values are those returned by ``StructureSet.get_folders()``.

        The read property value is stored in the folders' json-database created
        by ``StructuresSet.write_files()``, under the "key_value_pairs" key. For instance::

            "key_value_pairs": {"folder": "./random_strs-14", [property_name]: 9.266617521975935},

        where [property_name] represents here the string value of the parameter ``property_name``.

        Parameters:

        ``property_name``: string
            key for the ``self._props`` dictionary of property values

        ``read_property``: function
            Function to extract property value from ab-initio files. Return value
            must be scalar and signature is::

                read_property(folder_path, args[0], args[1], ...)

            where ``folder_path`` is the path of the folder containing the relevant
            property files.

        ``**kwargs``: keyworded variable length argument list
            You may call this method as::

                sset_instance.read_property_values(read_property, arg1=arg1, arg2=arg2, ... argN=argN)

            where ``arg1`` to  ``argN`` are the keyworded arguments to the ``read_property(folder_path,**kwargs)`` function.
        """
        import os
        import glob
        from clusterx.utils import list_integer_named_folders

        db = self.get_folders_db()

        self._props[property_name] = []
        for i,folder in enumerate(self._folders):
            pval = read_property(folder,**kwargs)
            self._props[property_name].append(pval)
            db.update([i+1], **{property_name:pval})
            # Note: db.update([i+1], property_name=pval) sets the key to "property_name" and not the value of property_name.


    def get_property_names(self):
        """Return list of stored property names.
        """
        return list(self._props.keys())

    def get_property_values(self, property_name):
        """Return list of property values.

        **Parameters:**

        ``property_name``: String
            Name of the property. If not sure, a list of property names can be
            obtained ``StructuresSet.get_property_names()``.
        """
        return self._props[property_name]

    def set_property_values(self, property_name = "total_energy", property_vals = []):
        """Set property values

        Set the property values.

        If a folders' json-database (as created by ``StructuresSet.write_files()``)
        exists, it is updated.

        **Parameters:**

        ``property_name``: string
            Name of the property

        ``property_vals``: array
            Array of property values

        """
        db = self.get_folders_db()
        self._props[property_name] = property_vals

        if db is not None:
            for i,p in enumerate(property_vals):
                db.update([i+1], **{property_name:p})
