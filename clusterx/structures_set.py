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
    cluster expansion, or as a validation set for cross validation. All the
    structures contained in a StructuresSet object must derive from a single
    ParentLattice object.

    **Parameters:**

    ``parent_lattice``: ParentLattice object
        All the structures on a structures set must derive from the same parent
        lattice given here. This argument can be ommited if parsing from file (see
        below).

    ``filepath``: String
        if provided, the structures set is initialized from a structures_set file, as created
        by ``StructuresSet.serialize()`` or ``StructuresSet.write_files()``. In this case,
        the ``parent_lattice`` argument can be ommited (if present, it is overriden).

    ``json_db_filepath``: String
        Deprecated, use ``filepath`` instead. If set, overrides ``filepath``

    ``calculator``: ASE calculator object (default: None)

    ``quick_parse``: Boolean (default: ``False``)
        if True, it assumes that, in the json file to be parsed (see ``db_fname``),
        the atom indices of the structures are the same as those of the supercell.
        Otherwise, the atom positions of structures and supercell are verified for
        every structure in the structures set being parsed. This leads to a slower parsing
        but safer if not sure how the file was built.

    **Deprecated parameters:**
    
    ``db_fname``: replaced by ``json_db_filepath``

    **Examples:**

    .. todo::
        * Refactor merge of two StructureSets

    **Methods:**
    """
    def __init__(self, parent_lattice = None, filepath = None, json_db_filepath = None, calculator = None, quick_parse = False, **sset_opts):

        if json_db_filepath is not None:
            filepath = json_db_filepath
            
        self._iter = 0
        self._parent_lattice = parent_lattice
        self._nstructures = 0
        ######## Lists ##########
        self._structures = []
        self._props = {}
        self._folders = []
        self._ids = []
        #########################
        self._db_fname = sset_opts.pop("db_fname", filepath)
        
        if isinstance(calculator,Calculator):
            self.set_calculator(calculator)
        else:
            self._calculator = None

        if self._db_fname is not None:
            self._init_from_db(quick_parse=quick_parse)
        else:
            self._db = None

    def _init_from_db(self,quick_parse = False):
        from clusterx.utils import get_cl_idx_sc
        self._db = connect(self._db_fname)
        #_folders  = self._db.metadata["folders"]
        _folders  = self._db.metadata.get("folders",[])
        _props = self._db.metadata.get("properties",{})

        plat_dict = self._db.metadata.get("parent_lattice")
        if plat_dict is not None:
            self._parent_lattice = ParentLattice.plat_from_dict(plat_dict)
        else:
            self._parent_lattice = ParentLattice.plat_from_dict_obsolete(self._db.metadata)

        for i,row in enumerate(self._db.select()):
            atoms = row.toatoms()
            atoms.wrap()
            tmat = row.get("data",{}).get("tmat")
            if tmat is None:
                tmat = calculate_trafo_matrix(self._parent_lattice.get_cell(),atoms.get_cell())
            scell = SuperCell(self._parent_lattice,tmat)

            props = {}
            for k,v in _props.items():
                try:
                    props[k] = v[i]
                except:
                    props[k] = None

            if quick_parse:
                s = Structure(scell, decoration=atoms.get_atomic_numbers())
            else:
                idxs = get_cl_idx_sc(scell.get_positions(),atoms.get_positions(),method=1)
                s = Structure(scell, decoration=atoms.get_atomic_numbers()[idxs])

            self.add_structure(s,folder=_folders[i],**props)


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

    def __add__(self, anothersset):
        sset_union = StructuresSet(parent_lattice=self._parent_lattice)
        sset_union.add_structures(self)
        sset_union.add_structures(anothersset)
        return sset_union

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
            ni = len(self.get_structure(i))
            indices[i] = ni//nplat
        return indices

    def get_natoms(self):
        """Return array of number of atoms of every strucure in the structures set.

        """
        natoms = np.zeros(self.get_nstr(),dtype=int)
        for i in range(self.get_nstr()):
            ni = len(self.get_structure(i))
            natoms[i] = ni
        return natoms

    def add_structure(self, structure, folder="", **props):
        """Add a structure to the StructuresSet object

        **Parameters:**

        ``structure``: Structure object
            Structure object for the structure to be added.
        ``folder``: string (default:"")
            optionally, path of the folder containing ab-initio runs. Paths are
            created automatically when calling ``StructuresSet.write_files()``.
            See related documentation for more details.
        ``props``: keyword arguments
            keyword arguments to be stored in the properties dictionary of a
            StructuresSet object.
        """
        if self._calculator is not None:
            structure.set_calculator(self._calculator)

        self._structures.append(structure)
        self._nstructures += 1

        self._ids.append(len(self._structures))

        self._folders.append(folder)

        for k in self._props.keys():
            if k in list(props.keys()):
                self._props[k].append(props[k])
            else:
                self._props[k].append(None)


        for k in props.keys():
            if k not in list(self._props.keys()):
                self._props[k]=[]
                for i in range(self.get_nstr()-1):
                    self._props[k].append(None)
                self._props[k].append(props[k])

    def add_structures(self, structures = None, sort_key = None):
        """Add structures to the StructureSet object

        **Parameters:**

        ``structures``: list of Structure objects, path to JSON file, or StructuresSet object
            Structures to be added

        ``sort_key``: list of three integers (default:None)
            Only relevant if ``structures`` is a JSON file. Sort atomic
            positions after reading. For example, the value (2,1,0)
            will sort as: increasing z-coordinate first, increasing
            y-coordinate second, increasing x-coordinate third. Useful to get
            well ordered slab structures, for instance.

        .. todo::
            Look at possible issues when structures instance is str: it may fail if
            positions in actual structure are shuffled wrt parent lattice (a situation
            that is not common, but that may appear when using
            runs not designed for a CE ...).

            It would be very good that ``structures`` can be a list of Atoms objects too.
        """
        if isinstance(structures, (list, np.ndarray)):
            for structure in structures:
                self.add_structure(structure)

        elif isinstance(structures, str):
            from clusterx.utils import sort_atoms
            db = connect(structures)
            for row in db.select():
                atoms = sort_atoms(row.toatoms(), key = sort_key)
                atoms.wrap()
                tmat = row.get("data",{}).get("tmat")
                if tmat is None:
                    tmat = calculate_trafo_matrix(self._parent_lattice.get_cell(),atoms.get_cell())
                scell = SuperCell(self._parent_lattice,tmat,sort_key=sort_key)
                self.add_structure(Structure(scell, decoration=atoms.get_atomic_numbers()))

        elif isinstance(structures, StructuresSet):

            for i,s in enumerate(structures):
                props = {}
                for k,v in structures._props.items():
                    props[k] = v[i]
                self.add_structure(s,folder=structures._folders[i],**props)

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

    def get_images(self,rm_vac=True,n=None):
        """
        Return array of Atoms objects from structures set.

        **Parameters:**

        ``rm_vac``: Boolean
            whether the returned Atoms objects contain vacancies, i.e. atoms with
            species number 0 or chemical symbol X. If true, vacancy sites are eliminated
            in the returned Atoms objects
        ``n``: integer
            return the first ``n`` structures. If ``None``, return all structures.
        """
        from clusterx.utils import remove_vacancies
        images = []
        if n is None:
            nmax = len(self)
        else:
            nmax = n

        if not rm_vac:
            for i in range(nmax):
                images.append(self._structures[i].get_atoms())
        else:
            for i in range(nmax):
                images.append(remove_vacancies(self._structures[i].get_atoms()))

        return images

    def get_subset(self, structure_indices = [], transfer_properties = True):
        """Return structures set instance containing a subset of structures of the original structures set

        **Parameters**

        ``structure_indices``: list or array
            indices of the structures in the original StructuresSet to be included in the subset.
        ``transfer_properties``: Boolean
            if True (default), copy the properties from the original StructuresSet to the subset.
        """
        folders = self.get_folders()
        property_dict = self.get_property()
        calculator = self.get_calculator()

        subset = StructuresSet(self.get_parent_lattice())

        for s in structure_indices:
            if folders is not None:
                subset.add_structure(self.get_structure(s), folder = folders[s])
            else:
                subset.add_structure(self.get_structure(s))
                
            
        subset.set_calculator(calculator)

        if transfer_properties:
            for pn, pv in property_dict.items():
                _pv = []
                for i in structure_indices:
                    _pv.append(pv[i])
                subset.set_property_values(property_name = pn, property_vals = _pv)

        return subset

    def set_calculator(self, calc):
        """
        Assign calculator object to every structure in the structures set.

        **Parameters:**

        ``calc``: Calculator object
        """
        self._calculator = calc
        if self._calculator is not None: # May be called with None from __init__
            for structure in self._structures:
                structure.set_calculator(self._calculator)

    def get_calculator(self):
        """Get Calculator object associated to the structures set.
        """
        return self._calculator

    def get_property(self):
        """Get property dictionary of StructuresSet object

        All the properties in a StructuresSet object, are stored in a dictionary with the following structure::
        
           {"prop_name_1": [p10, p11, ...], "prop_name_2": [p20,p21, ..], ...}

        where ``"prop_name_i"`` is the name of the property ``i``, and ``pij`` is the value of property ``i`` for structure ``j``.  

        This dictionary is returned by this method.
        """
        return self._props

    def get_parent_lattice(self):
        """Get ParentLattice object of structures set.

        Returns the ParentLattice object from which all the structures in the StructuresSet object derive.
        """
        return self._parent_lattice

    # DEPRECATED, use compute_property_values instead
    def calculate_property(self, prop_name="energy", prop_func=None, rm_vac=True):
        self.compute_property_values(property_name=prop_name, property_calc=prop_func, rm_vacancies=rm_vac, update_json_db=False)
        
    def compute_property_values(self, property_name="energy", property_calc=None, rm_vacancies=True, update_json_db=True, **kwargs):
        """
        Return array of calculated property for all structures in the structures set.

        **Parameters**:

        ``property_name``: string (default: "energy")
            Name of the property to be calculated. This is used as a key for the
            ``self._props`` dictionary. The property values can be recovered by
            calling the method ``StructureSet.get_property_values(property_name)`` (see documentation).

        ``property_calc``: function (default: ``None``)
            If none, the property value is calculated with the calculator object assigned
            to the structures set with  the method ``StructuresSet.set_calculator()``. If not
            None, it must be a function with the following signature::

                my_function(i, structure, **kwargs)    

            where ``i`` is the structure index, ``structure``
            is the structure object for structure index ``i``, and ``**kwargs`` are
            any additional keyword arguments. The function must return a number.

        ``rm_vacancies``: Boolean (default:``True``)
            Only takes effect if ``property_func`` is ``None``, i.e., when an ASE calculator
            (or derived calculator) is used. If True, the "Atoms.get_potential_energy()" method
            is applied to a copy of Structure.atoms object with vacancy sites removed,
            i.e., atom positions containing species with species number 0 or species
            symbol "X".

        ``update_json_db``: Boolean (default:``True``)
            Whether to update the json database file (in case one is attached to the sset instance).

        ``**kwargs``: keyword argument list, arbitrary length
            keyword arguments directly passed to ``property_func`` function.
            You may call this method as::

                sset_instance.calculate_property(property_name="my_prop", property_func="my_func", arg1=arg1, ..., argN=argN)

            where ``arg1`` to  ``argN`` are the keyword arguments passed to the 
            ``my_func(i, structure, **kwargs)`` function.

        """
        props = []
        if rm_vacancies:
            from clusterx.utils import remove_vacancies
        for i,st in enumerate(self):
            if property_calc is None:
                ats = st.get_atoms()
                if rm_vacancies:
                    _ats = remove_vacancies(ats)
                else:
                    _ats = ats.copy()
                    _ats.set_calculator(ats.get_calculator()) # ASE's copy() forgets calculator
                props.append(_ats.get_potential_energy())
            else:
                props.append(property_calc(i, st, **kwargs))

        self._props[property_name] = props

        if update_json_db:
            self._update_properties_in_json_db()

        return props

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

    # Deprecated: Use write_input_files instead
    def write_files(self, root = ".", prefix = '', suffix = '', fnames = None, formats = [], overwrite = True, rm_vac=False):
        self.write_input_files(root, prefix, suffix, fnames, formats, overwrite, rm_vac)

    def write_input_files(self, root = ".", prefix = '', suffix = '', fnames = None, formats = [], overwrite = True, rm_vac=False):
        """Create folders containing structure input files for ab-initio calculations.

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

        ``remove_vacancies``: Boolean
            Vacancies are represented with chemical symbol ``X`` and atomic number
            0. Output file formats will contain lines with atomic positions corresponding
            to vacancies. If you want them absent in the files, set ``remove_vacancies``
            to ``True``.
        """
        from ase.io import write
        import os

        if overwrite:
            self._folders = []

        if fnames is None:
            fnames = ["geometry.json"]
        elif isinstance(fnames,str):
            fnames = [fnames]

        if len(formats) == 0:
            for i in range(len(fnames)):
                formats.append(None)

        images = self.get_images(rm_vac=rm_vac)

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

            for fname, format in zip(fnames,formats):
                path = os.path.join(root,prefix+str(i)+suffix,fname)
                if overwrite or not os.path.isfile(path):
                    write(path, atoms, format)

        db_path = os.path.join(root,prefix+"0"+"-"+str(self.get_nstr()-1)+suffix+".json")
        self.serialize(path=db_path,overwrite=True)

    # Deprecated
    def write_to_db(self, path="sset.json", overwrite=False, rm_vac=False):
        self.serialize(path=path, overwrite=overwrite, rm_vac=rm_vac)

    def serialize(self, filepath="sset.json", path=None, overwrite=False, rm_vac=False):
        """Serialize StructuresSet object

        The serialization creates a Json ASE database object and writes a json file.
        This file can be used to reconstruct a StructuresSet object, by initializing
        with::

            StructuresSet(filename="sset.json")

        where "sset.json" is the file written in ``filepath``.

        **Parameters:**

        ``filepath``: string
            Output file name.

        ``path``: string
            *DEPRECATED*, use filepath instead. Output file name.

        .. todo::
            * At the moment, if properties where calculated using a calculator from ASE,
                the last state of the calculator (energy, forces) is stored in the last
                structure of the database. If the calculator es set every time a property is
                evaluated for every structure, then every structure contains this state. Both are
                undesired behaviors. However, it is not obvious how to get rid of it without
                modifying ASE. So far we leave it like that, but should be removed in the
                future.

        """
        from ase.io import write
        import os

        if path is not None:
            filepath = path

        self._db_fname = filepath
        try:
            self._db = connect(filepath, type = "json", append = not overwrite)
        except:
            self._db = connect(filepath, type = "json", append = False)

        nstr = self.get_nstr()
        images = self.get_images(rm_vac=rm_vac)

        if (overwrite or not os.path.isfile(filepath)) and nstr != 0:
            for i in range(nstr):
                self._db.write(images[i])
                ppts = {}
                for k,v in self._props.items():
                    ppts[k] = v[i]
                    #self._db.update(i+1, **{k:v[i]}) # Writes to the "key_value_pairs" key of the db entry for this Atoms object
                self._db.update(i+1, data={"properties":ppts}) # Writes to the "data" key of the db entry for this Atoms object
                self._db.update(i+1, folder=self._folders[i])

            self._db.metadata = {
                "folders" : self._folders, # "folders" must be removed from the metadata, must stay only on the structures, as the properties
                "db_fname" : self._db_fname,
                "nstr" : self.get_nstr(),
                "parent_lattice" : self._parent_lattice.as_dict()
            }
            self._db.metadata = {**self._db.metadata,"properties":self._props}


    def get_folders(self):
        """Get list of folders containing structure files for ab-initio calculations
        as created by ``StructureSet.write_files()``
        """
        return self._folders

    def get_db(self):
        """Get json database object corresponding to the list of folders
        containing structure files for ab-initio calculations
        as created by ``StructureSet.write_files()``
        """
        return self._db

    def get_db_fname(self):
        """Get file name of json database corresponding to the list of folders
        containing structure files for ab-initio calculations
        as created by ``StructureSet.write_files()``
        """
        return self._db_fname

    def read_energy(i, folder, structure=None, **kwargs): # DEPRECATED, use energy_parser instead
        self.energy_parser(i, folder, structure=structure, **kwargs)
        
    def energy_parser(i, folder, structure=None, **kwargs):
        """Read value stored in ``energy.dat`` file.

        This is to be used as the default argument for the ``read_property``
        parameter of the ``StructureSet.read_property_values()``
        method. Can be used as a template for reading different properties to
        be passed to ``StructureSet.read_property_values()``.

        **Parameters:**

        ``i``: integer
            folder number

        ``folder``: string
            absolute or relative path of the folder containing the file/s to be read.
        
        ``structure``: Structure object
            structure object for structure index ``i``

        ``**kwargs``: keyword arguments
            Extra arguments needed for the property reading. See documentation of
            ``StructureSet.read_property_values()``.
        """
        import os
        with open(os.path.join(folder,"energy.dat"),"r") as f:
            erg = float(f.readlines()[0])
        return erg

    # DEPRECATED, use parse_property_values instead
    def read_property_values(self,
                             property_name = "total_energy",
                             write_to_file=True,
                             read_property = energy_parser,
                             root = "",
                             update_json_db=True,
                             **kwargs):
        self.parse_property_values(
                             property_name = property_name,
                             write_to_file=write_to_file,
                             property_parser = read_energy,
                             root = root,
                             update_json_db=update_json,
                             **kwargs)
        
    def parse_property_values(self,
                             property_name = "total_energy",
                             write_to_file=True,
                             property_parser = energy_parser,
                             root = "",
                             update_json_db=True,
                             **kwargs):
        """Read calculated property values from ab-inito output files

        Read property values from ab-initio code output files. These files are
        contained in paths::

            [[root] /] [prefix] id [suffix] / file_to_read

        as created by ``StructureSet.write_input_files()``. The folders to be searched
        for energy values are those returned by ``StructureSet.get_folders()``. These
        can be also obtained directly from the ``"metadata":{"folders":[ ... ]}`` elements
        of the json database file.

        The read property value is stored in the json-database of the StructuresSet object
        (i.e., that obtained from ``StructureSet.get_db_fname()``), under the
        key ``"data": {"properties": { ... }}`` dictionary for every structure. For instance::

             "data": {"properties": {"formation_energy_per_site": -0.05788398131602701, "total_energy": -9824740.09590308}},

        where ``"formation_energy_per_site"`` and  ``"total_energy"`` here are the
        string value of the parameter ``property_name`` in the call to ``read_property_values()``.

        **Parameters:**

        ``property_name``: string
            key for the ``self._props`` dictionary of property values

        ``write_to_file``: Boolean
            Whether to write property values to a file with name ``property_name.dat``.

        ``read_property``: function
            Function to extract property value from ab-initio files. Return value
            must be scalar and signature is::

                read_property(i,folder_path, structure = None, **kwargs)

            where ``i`` is the structure index, ``folder_path`` is the path
            of the folder containing the relevant ab-initio files, structure
            is the structure object for structure index ``i``, and ``**kwargs`` are
            any additional keyword arguments.
        
        ``root``: string
            the root folder containing the subfolders with ab-initio data. See description above.

        ``update_json_db``: Boolean (default:``True``)
            Whether to update the json database file (in case one is attached to the sset instance).

        ``**kwargs``: keyword argument list, arbitrary length
            keyword arguments directly passed to ``read_property`` function.
            You may call this method as::

                sset_instance.read_property_values(read_property, arg1=arg1, ..., argN=argN)

            where ``arg1`` to  ``argN`` are the keyword arguments passed 
            to the ``read_property(folder_path,**kwargs)`` function.

        """
        import os
        import glob
        from clusterx.utils import list_integer_named_folders

        self._props[property_name] = []
        for i,folder in enumerate(self._folders):
            if root != "":
                folder = os.path.join(root,os.path.relpath(folder))
            try:
                pval = property_parser(i, folder, structure=self.get_structure(i), **kwargs)
            except:
                print("Could not parse propery ", property_name, "from folder ",folder )
                pval = None

            self._props[property_name].append(pval)

            # Store property value in a file in "root/folder/"
            if write_to_file:
                f = open(os.path.join(root, folder, property_name+".dat"),"w+")
                f.write("%2.9f\n"%(pval))
                f.close()

        if update_json_db:
            self._update_properties_in_json_db()

    def _update_properties_in_json_db(self):
        db = self.get_db()
        
        if db is not None:
            for i in range(len(self)):
                # Update properties dict for structure id i+1
                ppts = {}
                for k,v in self._props.items():
                    ppts[k] = v[i]

                db.update(i+1, data={"properties":ppts}) # Writes to the "data" key of the db entry for this Atoms object

            db.metadata = {**db.metadata,"properties":self._props}


    def set_property_values(self, property_name = "total_energy", property_vals = [], update_json_db=True):
        """Set property values

        Set the property values.

        If a folders' json-database (as created by ``StructuresSet.write_files()``)
        exists, it is updated.

        **Parameters:**

        ``property_name``: string
            Name of the property

        ``property_vals``: array
            Array of property values

        ``update_json_db``: Boolean (default:``True``)
            Whether to update the json database file (in case one is attached to the sset instance).
        """
        self._props[property_name] = property_vals
        if update_json_db:
            self._update_properties_in_json_db()

    def set_property_values_from_files(self, property_name = "property", property_file_name = "property.dat", cwd = "./"):
        """Set property values read from files

        Consider a ``StructuresSet`` oject named ``sset``.

        The list of folders ``sset.get_folders()`` is iterated and the value stored in the file named
        ``property_file_name`` is parsed and assigned to the corresponding sample in the ``sset``. 
        The name of the property is ``property_name`` and can be recovered by calling
        ``sset.get_property_values(property_name)``.

        If an associated json database exists, it is updated with the new property.

        **Parameters:**

        ``property_name``: string
            The name used to label the property in the structures set. This label is then listed
            in ``sset.get_property_names()`` and the property values for this label can be obtained
            by calling ``sset.get_property_values(property_name)``

        ``property_file_name``: string
            In every folder of the list ``sset.get_folders()`` there must be a file named 
            ``property_file_name`` containing a real number with the value of the property
        """
        import os
        fl = self.get_folders()

        props = []
        
        for folder in fl:
            fpath = os.path.join(cwd, str(folder), property_file_name)
            with open(fpath) as f:
                props.append(float(f.readline().strip()))
                
        self.set_property_values(property_name = property_name, property_vals = props)
                
        return props
 
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

        **Returns:**

        ``props``: python array
            A python array with the property values
        """
        return self._props[property_name]

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
        return np.array(predictions)

    def get_concentrations(self, site_type = 0, sigma = 1):
        """Get concentration values for a given site type
        """

        concentrations = []

        for s in self:
            fc = s.get_fractional_concentrations()
            concentrations.append(fc[site_type][sigma])

        return concentrations
