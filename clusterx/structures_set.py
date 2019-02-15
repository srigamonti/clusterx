# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

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

    ``folders_db_fname``: String
        if set, the structures set is initialized from a structures_set file, as created
        by ``StructuresSet.write_files()``.

    **Examples:**

    .. todo::

        * Use plat.as_dict to serialize metadata part of structures_set database.
        * Add idx_tags dict in metadata.
        * So far, the class allows to have only a subset of the structures written to the member json database. This may be dengerous in certain contexts, analyze whether changing this.
        * Refactor merge of two StructureSets


    **Methods:**
    """
    def __init__(self, parent_lattice=None, filename="structures_set.json", folders_db_fname=None, calculator = None):

        self._iter = 0
        self._filename = filename
        self._parent_lattice = parent_lattice
        self._props = {}
        self._structures = []
        self._nstructures = 0
        self.json_db = JSONDatabase(filename=self._filename)
        if isinstance(calculator,Calculator):
            self.set_calculator(calculator)

        if folders_db_fname is not None:
            self._folders_db_fname = folders_db_fname
            self._init_from_db()
        else:
            self._metadata = {}
            self._folders = []
            self._folders_db = None
            self._folders_db_fname = None


    def _init_from_db(self):
        self._folders_db = connect(self._folders_db_fname)
        self._metadata = self._folders_db.metadata
        self._folders  = self._metadata["folders"]
        self._props = self._metadata.get("properties",{})

        self.add_structures(structures = self._folders_db_fname)
        """
        if self._parent_lattice is None:
            pris = Atoms(cell=self._metadata["parent_lattice_pristine_unit_cell"],
                         pbc=self._metadata["parent_lattice_pbc"],
                         positions=self._metadata["parent_lattice_pristine_positions"],
                         numbers=self._metadata["parent_lattice_pristine_numbers"])
            self._parent_lattice = ParentLattice(atoms=pris, )
        """


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

    def get_scell_indices(self):
        """Return array of supercell indices.

        Every structure in a structure set, is a ("decorated") supercell.
        The index of a supercell is an integer number, equal to the super cell
        volume in units of the parent cell volume.

        This method returns an array of supercell indices, corresponding to each
        structure in the structures set.
        """
        nplat = len(self._parent_lattice)
        indices = np.zeros(self.get_nstr(),dtype=int)
        for i in range(self.get_nstr()):
            ni = len(self.get_structure(iall_SiGe_structures = StructuresSet(SiGe_parlat, filename='test_structures_set_merge.json')))
            indices[i] = ni//nplat
        return indices

    def write(self, structure, key_value_pairs={}, **kwargs):
        self.json_db.write(structure.get_atoms(),key_value_pairs, data={"pcell":structure.get_parent_lattice().get_cell(),"tmat":structure.get_transformation(), "tags":structure.get_tags(),"idx_subs":structure.get_idx_subs()},**kwargs)

    def write_to_db(self, json_db_name=None):
        """Creates ASE's JSON db object containing the structures in the structures set
        """
        from subprocess import call
        #from ase.db.jsondb import JSONDatabase
        from ase.db import connect
        call(["rm","-f",json_db_name])
        #atoms_db = JSONDatabase(filename=json_db_name)
        atoms_db = connect(json_db_name)
        for s in self:
            atoms_db.write(s.get_atoms())

    def add_structure(self, structure, key_value_pairs={}, write_to_db = False, **kwargs):
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

    def add_structures(self, structures = None, write_to_db = False):
        """Add structures to the StructureSet object

        **Parameters:**

        ``structures``: list of Structure objects, path to JSON file, or StructuresSet object
            Structures to be added

        .. todo::
            Look at possible issues when structures instance is str: it may fail if
            positions in actual structure are shuffled wrt parent lattice (a situation
            that is not common, but that may appear when using
            runs not designed for a CE ...).
        """
        if isinstance(structures, (list, np.ndarray)):
            for structure in structures:
                self.add_structure(structure, write_to_db = write_to_db)
        elif isinstance(structures, str):
            from clusterx.utils import sort_atoms
            db = connect(structures)
            sort_key = (2,1,0)
            for row in db.select():
                atoms = sort_atoms(row.toatoms(), key = sort_key)
                atoms.wrap()
                tmat = row.get("data",{}).get("tmat")
                if tmat is None:
                    tmat = calculate_trafo_matrix(self._parent_lattice.get_cell(),atoms.get_cell())
                scell = SuperCell(self._parent_lattice,tmat,sort_key=sort_key)
                #print(scell.get_scaled_positions())
                #print(atoms.get_scaled_positions())
                self.add_structure(Structure(scell, decoration=atoms.get_atomic_numbers()))
        elif isinstance(structures, StructuresSet):
            if not hasattr(self, 'metadata'):
                self.metadata = {}
            if self._folders_db == None:
                self._folders_db_fname = self._filename
                self._folders_db = connect(self._filename)
            if self._parent_lattice != structures._parent_lattice:
                raise ValueError("Parent lattices do not match.")
            n_old_structures = len(self._structures)
            for structure in structures._structures:
                self.add_structure(structure, write_to_db = write_to_db)
            if not 'folders_db_fname' in self.metadata.keys():
                self.metadata['folders_db_fname'] = self._filename
            if 'folders' in structures._metadata.keys(): #are we adding a new folder structure?
                if 'folders' in self._metadata.keys(): # do we already have folders stored?
                    if n_old_structures == len(self._metadata['folders']): #all structures have a folder
                        for folder in structures._metadata['folders']: #we merge them
                            self._metadata['folders'].append(folder)
                    else:
                        pass #not sure what to do in this case #TODO ask Santiago
                elif n_old_structures == 0:
                    self._metadata['folders'] = structures._metadata['folders']
                else:
                    pass #same here
            #n_existing_structure = self._metadata['nstr']
            if 'nstr' in self._metadata.keys():
                self._metadata['nstr'] += structures._metadata['nstr']
            else:
                self._metadata['nstr'] = structures._metadata['nstr']
            for new_prop_key in structures._props.keys():
                if new_prop_key in self._props.keys():
                    for prop in structures._props[new_prop_key]:
                        self._props[new_prop_key].append(prop)
                else:
                    nones = []
                    for d in range(n_old_structures):
                        nones.append('None')
                    for prop in structures._props[new_prop_key]:
                        nones.append(prop)
                    self._props[new_prop_key] = nones
            if write_to_db:
                for property_vals, property_name in zip([self._props[x] for x in self._props.keys()], self._props.keys()):
                    for i,p in enumerate(property_vals): #see "set_property_values"
                        self._folders_db.update(i+1, **{property_name:p})
                for meta_key in self._metadata.keys():
                    self._folders_db.metadata = {**self._folders_db.metadata,meta_key : self._metadata[meta_key]}
                self._folders_db.metadata = {**self._folders_db.metadata,"properties":self._props}

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
        self._props[prop] = props
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
        Perform ab-initio calculation of energies using an ASE calculator.

        The folders list as returned by ``StructuresSet.get_folders()`` is
        iterated. The current working directory (``cwd``) is set to the
        actual folder in the loop. The structure in the file ``structure_fname``
        is converted to an ``Atoms`` object, whose calculator is set to
        ``calulator``. The ``Atoms.get_potential_energy()`` method is called
        and the resulting total energy is stored in the file ``cwd/energy.dat``.
        """
        import os
        from ase.io import read

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


    def write_files(self, root = ".", prefix = '', suffix = '', fnames = None, formats = [], overwrite = True, fix_atoms=None, remove_vacancies=False):
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

        ``fix_atoms``: list of integer or ``None`` (Default:``None``)
            List of atom indices which should remain fixed during atomic relaxation.
            In ``None``, no constrain is applied. Constraints only take effect for
            certain output formats. Uses ``FixAtoms`` constraint of ``ASE``.

        ``remove_vacancies``: Boolean
            Vacancies are represented with chemical symbol ``X`` and atomic number
            0. Output file formats will contain lines with atomic positions corresponding
            to vacancies. If you want them absent in the files, set ``remove_vacancies``
            to ``True``.
        """
        import os
        from ase.io import write

        path = os.path.join(root,prefix+"0"+"-"+str(self.get_nstr()-1)+suffix+".json")
        self._folders_db_fname = path
        try:
            self._folders_db = connect(path, type = "json", append = not overwrite)
        except:
            self._folders_db = connect(path, type = "json", append = False)

        if overwrite:
            self._folders = []

        if fnames is None:
            fnames = ["geometry.json"]
        elif isinstance(fnames,str):
            fnames = [fnames]

        if len(formats) == 0:
            for i in range(len(fnames)):
                formats.append(None)

        images = self.get_images(remove_vacancies=remove_vacancies)
        if fix_atoms is not None:
            from ase.constraints import FixAtoms

        for i in range(self.get_nstr()):
            path = os.path.join(root,prefix+str(i)+suffix)
            if not os.path.exists(path):
                os.makedirs(path)

            if i < len(self._folders):
                self._folders[i] = path
            else:
                self._folders.append(path)

            #atoms = self.get_structure(i).get_atoms()
            atoms = images[i]

            if fix_atoms is not None:
                c = FixAtoms(indices=fix_atoms)
                atoms.set_constraint(c)

            for fname, format in zip(fnames,formats):
                path = os.path.join(root,prefix+str(i)+suffix,fname)
                if overwrite:
                    write(path, atoms, format)
                elif not os.path.isfile(path):
                    write(path, atoms, format)

            if overwrite:
                self._folders_db.write(atoms, folder=self._folders[i])
            elif not os.path.isfile(path):
                self._folders_db.write(atoms, folder=self._folders[i])

        if self.get_nstr() != 0:
            self._folders_db.metadata = {
                "folders" : self._folders,
                "folders_db_fname" : self._folders_db_fname,
                "nstr" : self.get_nstr(),
                "parent_lattice_pbc" : self._parent_lattice.get_pbc(),
                "parent_lattice_pristine_unit_cell" : self._parent_lattice.get_cell(),
                "parent_lattice_pristine_positions" : self._parent_lattice.get_positions(),
                "parent_lattice_pristine_numbers" : self._parent_lattice.get_atomic_numbers(),
                "parent_lattice_tags" : self._parent_lattice.get_tags(),
                "parent_lattice_idx_subs" : self._parent_lattice.get_idx_subs()
            }



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
        as created by ``StructureSet.write_files()``
        """
        return self._folders_db_fname

    def read_energy(i,folder,**kwargs):
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

    def read_property_values(self, property_name = "total_energy", write_to_file=True, read_property = read_energy, **kwargs):
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

                read_property(i,folder_path, args[0], args[1], ...)

            where ``i`` is the structure index, and ``folder_path`` is the path
            of the folder containing the relevant property files.

        ``write_to_file``: Boolean
            Whether to write property values to a file with name ``property_name.dat``.

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
            pval = read_property(i,folder,**kwargs)
            self._props[property_name].append(pval)
            db.update(i+1, **{property_name:pval})
            if write_to_file:
                f = open(os.path.join(folder,property_name+".dat"),"w+")
                f.write("%2.9f\n"%(pval))
                f.close()

        db.metadata = {**db.metadata,"properties":self._props}

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
                db.update(i+1, **{property_name:p})

    def get_predictions(self, cemodel):
        """Get predictions of CE model on structures set

        Applies the given cluster expansion model to every structure in the
        structrues set and returns an array with the computed values.

        **Parameters:**

        ``cemodel``: Model object
            Cluster expansion model for which predictions want to be computed.
        """
        predictions = []
        for s in self:
            predictions.append(cemodel.predict(s))
        return predictions

    def get_concentrations(self, site_type = 0, sigma = 1):
        """Get concentration values for a given site type
        """

        concentrations = []

        for s in self:
            fc = s.get_fractional_concentrations()
            concentrations.append(fc[site_type][sigma])

        return concentrations
