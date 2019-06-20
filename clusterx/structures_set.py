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
        lattice given here.

    ``db_fname``: String
        if set, the structures set is initialized from a structures_set file, as created
        by ``StructuresSet.serialize()`` or ``StructuresSet.write_files()``.

    ``calculator``: ASE calculator object (default: None)

    ``quick_parse``: Boolean (default: ``False``)
        if True, it assumes that, in the json file to be parsed (see ``db_fname``),
        the atom indices of the structures are the same as those of the supercell.
        Otherwise, the atom positions of structures and supercell are verified for
        every structure in the structures set being parsed. This leads to a slower parsing
        but safer if not sure how the file was built.

    ``folders_db_fname``: String (deprecated)
        same as ``db_name``. Use ``db_name`` instead. If set overrides ``db_name``.

    **Examples:**

    .. todo::
        * Refactor merge of two StructureSets

    **Methods:**
    """
    def __init__(self, parent_lattice=None, db_fname=None, calculator = None, folders_db_fname=None, quick_parse=False):

        self._iter = 0
        self._parent_lattice = parent_lattice
        self._nstructures = 0
        ######## Lists ##########
        self._structures = []
        self._props = {}
        self._folders = []
        self._ids = []
        #########################
        self._db_fname = db_fname
        if folders_db_fname is not None:
            self._db_fname = folders_db_fname

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
        Assign calculator object to every structure in the structures set.

        **Parameters:**

        ``calc``: Calculator object
        """
        self._calculator = calc
        if self._calculator is not None: # May be called with None from __init__
            for structure in self._structures:
                structure.set_calculator(self._calculator)

    def get_calculator(self):
        return self._calculator

    def get_property(self):
        return self._props

    def get_parent_lattice(self):
        return self._parent_lattice

    def calculate_property(self, prop_name="energy", prop_func=None, rm_vac=True):
        """
        Return array of calculated property for all structures in the structures set.

        **Parameters**:

        ``prop_name``: string (default: "energy")
            Name of the property to be calculated. This is used as a key for the
            ``self._props`` dictionary. The property values can be recovered by
            calling the method ``StructureSet.get_property_values(prop_name)`` (see documentation).

        ``prop_func``: function (default: ``None``)
            If none, the property value is calculated with the calculator object assigned
            to the structures set with  the method ``StructuresSet.set_calculator()``. If not
            None, it must be a function that recieves a Structure object as argument, and
            returns a number.

        ``rm_vac``: Boolean (default:``True``)
            Only takes effect if ``prop_func`` is ``None``, i.e., when an ASE calculator
            (or derived calculator) is used. If True, the "Atoms.get_potential_energy()" method
            is applied to a copy of Structure.atoms object with vacancy sites removed,
            i.e., atom positions containing species with species number 0 or species
            symbol "X".
        """
        props = []
        if rm_vac:
            from clusterx.utils import remove_vacancies
        for i,st in enumerate(self):
            if prop_func is None:
                ats = st.get_atoms()
                if rm_vac:
                    _ats = remove_vacancies(ats)
                else:
                    _ats = ats.copy()
                    _ats.set_calculator(ats.get_calculator()) # ASE's copy() forgets calculator
                props.append(_ats.get_potential_energy())
            else:
                #if rm_vac:
                #    st.atoms = remove_vacancies(st.atoms) # This leaves an inconsistent Structure object
                props.append(prop_func(st))

        self._props[prop_name] = props
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

    def serialize(self, path="sset.json", overwrite=False, rm_vac=False):
        """Serialize StructuresSet object

        The serialization creates a Json ASE database object and writes a json file.
        This file can be used to reconstruct a StructuresSet object, by initializing
        with::

            StructuresSet(filename="sset.json")

        where "sset.json" is the file written in ``path``.

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

        self._db_fname = path
        try:
            self._db = connect(path, type = "json", append = not overwrite)
        except:
            self._db = connect(path, type = "json", append = False)

        nstr = self.get_nstr()
        images = self.get_images(rm_vac=rm_vac)

        if (overwrite or not os.path.isfile(path)) and nstr != 0:
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
        """Get list folders containing structure files for ab-initio calculations
        as created by ``StructureSet.write_files()``
        """
        return self._folders

    def get_db(self):
        """Get json database object corresponding to the list folders
        containing structure files for ab-initio calculations
        as created by ``StructureSet.writ_files()``
        """
        return self._db

    def get_db_fname(self):
        """Get file name of json database corresponding to the list folders
        containing structure files for ab-initio calculations
        as created by ``StructureSet.write_files()``
        """
        return self._db_fname

    def read_energy(i,folder,structure=None, **kwargs):
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

        ``**kwargs``: keyword arguments
            Extra arguments needed for the property reading. See documentation of
            ``StructureSet.read_property_values()``.

        .. todo:

            Remove parameter ``i`` from this template.
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

        Parameters:

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
            is the structure object for structure index ``i``, and **kwars are
            any additional keyword arguments.

        ``**kwargs``: keyworded argument list, arbitrary length
            You may call this method as::

                sset_instance.read_property_values(read_property, arg1=arg1, arg2=arg2, ... argN=argN)

            where ``arg1`` to  ``argN`` are the keyworded arguments to the ``read_property(folder_path,**kwargs)`` function.
        """
        import os
        import glob
        from clusterx.utils import list_integer_named_folders

        db = self.get_db()

        self._props[property_name] = []
        for i,folder in enumerate(self._folders):
            try:
                pval = read_property(i,folder,structure=self.get_structure(i),**kwargs)
            except:
                pval = None

            self._props[property_name].append(pval)
            #db.update(i+1, **{property_name:pval})
            ppts = {}
            for k,v in self._props.items():
                ppts[k] = v[i]

            db.update(i+1, data={"properties":ppts}) # Writes to the "data" key of the db entry for this Atoms object

            #db.update(i+1, property_name:pval})
            if write_to_file:
                f = open(os.path.join(folder,property_name+".dat"),"w+")
                f.write("%2.9f\n"%(pval))
                f.close()

        db.metadata = {**db.metadata,"properties":self._props}

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
        db = self.get_db()
        self._props[property_name] = property_vals

        for i in range(len(self)):
            ppts = {}
            for k,v in self._props.items():
                ppts[k] = v[i]

            db.update(i+1, data={"properties":ppts}) # Writes to the "data" key of the db entry for this Atoms object

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
        return predictions

    def get_concentrations(self, site_type = 0, sigma = 1):
        """Get concentration values for a given site type
        """

        concentrations = []

        for s in self:
            fc = s.get_fractional_concentrations()
            concentrations.append(fc[site_type][sigma])

        return concentrations
