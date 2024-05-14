# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from ase import Atoms
import numpy as np
import sys
from ase.visualize import view
#from ase.build import make_supercell
from clusterx.utils import make_supercell
from clusterx.parent_lattice import ParentLattice
from clusterx.symmetry import get_internal_translations, get_scaled_positions, wrap_scaled_positions, get_spacegroup



from clusterx.utils import get_cl_idx_sc
import clusterx
from ase.db import connect


class SuperCell(ParentLattice):
    """
    **Build a super cell**

    A :class:`SuperCell <clusterx.super_cell.SuperCell>` object acts as a blue-print for the creation of structures
    with arbitrary decorations. 

    The :class:`SuperCell <clusterx.super_cell.SuperCell>` class inherits from the :class:`ParentLattice <clusterx.parent_lattice.ParentLattice>` class. Therefore,
    all methods available to the :class:`ParentLattice <clusterx.parent_lattice.ParentLattice>` class are available to the
    :class:`SuperCell <clusterx.super_cell.SuperCell>` class. Refer to the documentation of :class:`ParentLattice <clusterx.parent_lattice.ParentLattice>` for more
    methods.

    **Parameters:**

    ``parent_lattice``: :class:`ParentLattice <clusterx.parent_lattice.ParentLattice>` object
        In **CELL**, a super cell is a periodic repetition of a parent lattice
        object. This parameter defines such a parent lattice for the created
        super cell.
    ``p``: integer, or 1x3, 2x2, or 3x3 integer array
        Transformation matrix :math:`P`. The cartesian coordinates of the
        latttice vectors defining the created SuperCell object, are the rows of
        the matrix :math:`S` defined by :math:`S = PV` where the rows of
        :math:`V` are the cartesian coordinates of the lattice
        vectors of the ParentLattice object. That is, the value of
        :py:meth:`ParentLattice.get_cell() <clusterx.parent_lattice.ParentLattice.get_cell()>`. The given matrix ``p`` must be compatible
        with the periodic boundary conditions of the parent lattice, i.e. the
        resulting super cell must not contain translations along the
        non-periodic directions.
    ``sort_key``: list of three integers, optional
        Create supercell object with sorted atomic coordinates

        Example:``sort_key=(2,1,0)`` will result in atoms sorted
        according to increasing z-coordinate first, increasing
        y-coordinate second, increasing x-coordinate third. Useful to get
        well ordered slab structures, for instance. Sorting can be changed
        by appropriately setting the ``sort_key`` argument, with the same
        effect as in the argument to ``itemgetter`` below::

            from operator import itemgetter
            sp = sorted(p, key=itemgetter(2,1,0))

        here p is a Nx3 array of vector coordinates.
    ``json_db_filepath``: string (default: None)
        Overrides all the above. Used to initialize from file. Path of a json
        file containing a serialized superCell object, as generated
        by the ``SuperCell.serialize()`` method.

    .. todo::
        Proper check of pbc when building the super cell. Either ignore
        translation in ``p`` along non-periodic directions or warn in some way
        if ``p`` is not compatible with pbc.


    **Methods:**
    """

    def __init__(self, parent_lattice=None, p=None, sort_key=None, json_db_filepath=None, sym_table=False):

        if json_db_filepath is not None:
            db = connect(json_db_filepath)

            plat_dict = db.metadata.get("parent_lattice",{})
            self._plat = ParentLattice.plat_from_dict(plat_dict)
            p = db.metadata.get("tmat",np.identity(3,dtype=int))
            self._sort_key = db.metadata.get("sort_key",None)
        else:
            self._plat = parent_lattice
            self._sort_key = sort_key

        pbc = self._plat.get_pbc()
        
        if not isinstance(p,int):
            p = np.array(p)
            if p.shape == (1,):
                self._p = np.diag([p[0],1,1])
            elif p.shape == (2,):
                self._p = np.diag([p[0],p[1],1])
            elif p.shape == (3,):
                self._p = np.diag(p)
            elif p.shape == (2,2):
                self._p = np.array([[p[0,0],p[0,1],0],[p[1,0],p[1,1],0],[0,0,1]])
            elif p.shape == (3,3):
                self._p = p

        if isinstance(p,int):
            if pbc[0]==1 and pbc[1]==0 and pbc[2]==0:
                self._p = np.diag([p,1,1])
            elif pbc[0]==1 and pbc[1]==1 and pbc[2]==0:
                self._p = np.diag([p,p,1])
            elif pbc[0]==1 and pbc[1]==1 and pbc[2]==1:
                self._p = np.diag([p,p,p])
            else:
                sys.exit("SuperCell(Error)")

        self.index = int(round(np.linalg.det(self._p)))
        prist = make_supercell(self._plat.get_pristine(),self._p)
        subs = [make_supercell(atoms,self._p) for atoms in self._plat.get_substitutions()]

        prist.wrap()
        for i in range(len(subs)):
            subs[i].wrap()

        if self._sort_key is not None:
            from clusterx.utils import sort_atoms
            prist = sort_atoms(prist,key=self._sort_key)
            for i in range(len(subs)):
                subs[i] = sort_atoms(subs[i],key=self._sort_key)

        super(SuperCell,self).__init__(atoms = prist, substitutions = subs)
        self._natoms = len(self)
        self.set_pbc(self._plat.get_pbc())
        
        if sym_table == True:
            self._sym_table = self.get_symmetry_table()
        else: 
            self._sym_table = []

        self.sc_sg, self.sc_sym = self.get_sym()
        self.sc_sg_pl, self.sc_sym_pl = self.get_sym_platt()
        #self.internal_trans = get_internal_translations(self._plat, self) # Scaled to super_cell
        self.internal_trans = None # Scaled to super_cell
        #self.all_distances_mic = None
        #self.all_distances_nomic = None
        #self.scaled_positions = None
        self.sym_perm = None
        self.sym_perm_platt = None

    def compute_sym_perm(self, include_sc_trans = True):
        if include_sc_trans:
            self.sym_perm = []
        else:
            self.sym_perm_platt = []

        tol = 1e-3/np.cbrt(self.get_natoms())
        internal_trans = self.get_internal_translations() # Scaled to super_cell
        spos1 = get_scaled_positions(self.get_positions(wrap = True), self.get_cell(), pbc = self.get_pbc(), wrap = True) # Super-cell scaled positions
        spos = np.around(spos1,6) # Wrapping in ASE doesn't always work. Here a hard external fix by rounding and then applying again ASE's style wrapping.

        for i, periodic in enumerate(self.get_pbc()):
            if periodic:
                spos[:, i] %= 1.0
                spos[:, i] %= 1.0

        if include_sc_trans:
            sg, sym = self.get_sym()
        else:
            sg, sym = self.get_sym_platt()
            
        rot = sym["rotations"]
        tra = sym["translations"]

        _sym_perm = []
        #for r,t_ in zip(rot,internal_trans):
        for r,t_ in zip(rot,tra):
            t = np.round(t_, decimals=8)
            ts = np.tile(t,(len(spos),1)).T # Every column represents the same translation for every cluster site
            spos_rt = np.add(np.dot(r,spos.T),ts).T # Apply rotation, then translation

            _sp1_ = wrap_scaled_positions(spos_rt, self.get_pbc())

            _sp1 = np.around(_sp1_,5) # Wrapping in ASE doesn't always work. Here a hard external fix by rounding and then applying again ASE's style wrapping.
                        
            for i, periodic in enumerate(self.get_pbc()):
                if periodic:
                    _sp1[:, i] %= 1.0
                    _sp1[:, i] %= 1.0

            _cl = get_cl_idx_sc(_sp1, spos, method=1, tol=tol)
            _sym_perm.append(_cl)

        if include_sc_trans:
            self.sym_perm = np.unique(_sym_perm, axis = 0)
        else:
            self.sym_perm_platt = np.unique(_sym_perm, axis = 0)
        #self.sym_perm = _sym_perm

    def get_sym_perm(self, include_sc_trans = True):
        if include_sc_trans:
            if self.sym_perm is None:
                self.compute_sym_perm(include_sc_trans = True)
                return self.sym_perm
            else:
                return self.sym_perm
        else:
            if self.sym_perm_platt is None:
                self.compute_sym_perm(include_sc_trans = False)
                return self.sym_perm_platt
            else:
                return self.sym_perm_platt

    def get_sym(self):
        """Get space symmetry of a ParentLattice object.
        """
        try:
            return self.sc_sg, self.sc_sym
        except:
            self.sc_sg, self.sc_sym = self._compute_sym()
            return self.sc_sg, self.sc_sym

    def get_sym_platt(self):
        """Get space symmetry of a ParentLattice object.
        """
        try:
            return self.sc_sg_pl, self.sc_sym_pl
        except:
            self.sc_sg_pl, self.sc_sym_pl = self.get_parent_lattice()._compute_sym()
            return self.sc_sg_pl, self.sc_sym_pl

    def _compute_sym(self):
        return get_spacegroup(self)

    """
    def get_scaled_positions():
        if self.scaled_positions == None:
            self.scaled_positions = get_scaled_positions(wrap = True)
            return self.scaled_positions
        else:
            return self.scaled_positions
    
    def get_all_distances(self, mic=False):
        vector = False
        if mic:
            if self.all_distances_mic == None:
                self.all_distances_mic = super(ParentLattice, self).get_all_distances(mic, vector)
                return self.all_distances_mic
            else:
                return self.all_distances_mic

        if not mic:
            if self.all_distances_nomic == None:
                self.all_distances_nomic = super(ParentLattice, self).get_all_distances(mic, vector)
                return self.all_distances_nomic
            else:
                return self.all_distances_nomic
    """

    def get_internal_translations(self):
        """Get internal translations of parent lattice in supercell, scaled to supercell
        """
        if self.internal_trans is None:
            self.internal_trans = get_internal_translations(self._plat, self)
            return self.internal_trans
        else:
            return self.internal_trans
        
        
    def copy(self):
        """Return a copy
        """

        sc = self.__class__(self._plat, self._p, sort_key = self._sort_key)

        return sc

    def scell_from_dict(dict):
        """Generates SuperCell object from a dictionary as returned by SuperCell.as_dict()
        """
        plat_dict = dict.get("parent_lattice",{})
        plat = ParentLattice.plat_from_dict(plat_dict)
        p = dict.get("tmat",np.identity(3,dtype=int))
        sort_key = dict.get("sort_key",None)

        return SuperCell(plat,p,sort_key)

    def as_dict(self):
        """Return dictionary with object definition
        """
        plat_dict = self._plat.as_dict()
        self._dict = {"tmat":self._p,"parent_lattice":plat_dict,"sort_key":self._sort_key}

        return self._dict

    def get_index(self):
        """Return index of the SuperCell

        The index of the supercell is an integer number, equal to the super cell
        volume in units of the parent cell volume.
        """
        return self.index

    def get_parent_lattice(self):
        """Return the parent lattice object which defines the supercell.
        """
        return self._plat

    def get_transformation(self):
        """Return the transformation matrix that defines the supercell from the
        defining parent lattice.
        """
        return self._p

    def get_positions(self, wrap=False, **wrap_kw):
        return super(SuperCell, self).get_positions(wrap, **wrap_kw)

    def plot(self):
        """Plot the pristine supercell object.
        """
        view(self)

    def get_pristine_structure(self):
        """Return Structure object corresponding to pristine structure.
        """
        return self.gen_structure(sigmas=np.zeros(len(self.get_tags()),dtype=np.int8))


    def gen_structure(self, sigmas = None, mc = False):
        """
        Generate a Structure with given configuration.

        **Parameters:**

        ``sigmas``: array of integers
            sigmas[idx] must be the ocupation variable for crystal site idx.
            For instance, supose that::
                
                ThisObject.get_sublattice_types()

            returns ``{0:[14,13], 1:[56,47,38]}``, and that::
                
                ThisObject.get_tags()

            returns ``[0,0,1,1,0]``, meaning that for atom idx=0,1,4 the sublattice types are 0
            and for idx=2,3 the sublattice types is 1.
            Then, the argument ``sigmas=[0,1,0,2,1]`` will produce the configuration::

                [14,13,56,38,13]
        
        """
        import clusterx.structure
        return clusterx.structure.Structure(SuperCell(self._plat, self._p, self._sort_key, sym_table = bool(self._sym_table)), sigmas=sigmas, mc = mc)
        
    def gen_random(self, nsubs = None, mc = False):
        return self.gen_random_structure(nsubs = nsubs, mc = mc)
        
    def gen_random_structure(self, nsubs = None, mc = False):
        """
        Generate a random Structure with given number of substitutions.

        **Parameters:**

        ``nsubs``: None, int or dictionary (default: None)
            If dictionary, the ``nsubs`` argument must have the format ``{site_type0:[nsub01,nsub02,...], site_type1: ...}``
            where ``site_type#`` and ``nsub##`` are, respectively, an integer indicating
            the site type index and the number of substitutional species of each kind, as returned by
            ``SuperCell.get_sublattice_types()``.

            If integer, the SuperCell object must be correspond to a binary, i.e.
            the output of ``scell.is_nary(2)`` must be True. In this case, nsub indicates
            the number of substitutional atoms in the binary.

            If ``None``, the number of substitutions in each sublattice is
            generated at random.

        **Notes:**

        This function can be also called with the method ``gen_random``, with the same signature. 
        Use ``gen_random_structure``, since ``gen_random`` is deprecated.
        """
        import clusterx.structure
        
        if nsubs is None:
            import numpy as np
            slts = self.get_sublattice_types() #  e.g.  {0: [14,13], 1: [56,0,38], 2:[11]}
            tags = self.get_tags()  # tags[atom_index] = site_type
            _nsubs = {}
            for k,v in slts.items():
                if len(v) != 1:
                    _nsubs[k] = [np.random.randint(0,len(np.where(self.get_tags() == k)[0])+1)]

        elif isinstance(nsubs,int):
            if self.is_nary(2):
                slts = self.get_sublattice_types()
                _nsubs = {}
                for k,v in slts.items():
                    if len(v) != 1:
                        _nsubs[k] = [nsubs]

            else:
                return None
        else:
            _nsubs = nsubs

        decoration, sigmas = self.gen_random_decoration(_nsubs)

        return clusterx.structure.Structure(SuperCell(self._plat, self._p, self._sort_key, sym_table = bool(self._sym_table)), sigmas=sigmas, mc = mc)

    def gen_random_decoration(self,nsubs):
        """Generate a random decoration of the super cell with given number of substitutions.

        Parameters:

        nsubs: dictionary
            The format of the dictionary is as follows::

                {site_type1:[n11,n12,...], site_type2:[n21,n22,...], ...}

            it indicates how many substitutional atoms of every kind (n#1, n#2, ...)
            may replace the pristine species for sites of site_type#.

        Returns:

        decoration: list of int
            list with the same length as the pristine structure.
            decoration[atom_idx] = species number sitting at site idx.
        sigmas: list of int
            sigma[atom_index] is an integer between 0 and M-1, where M is the
            total number of different species which may substitute an atom
            in this position.
        """
        idx_subs = self.get_idx_subs()
        tags = self.get_tags() # tags[atom_index] = site_type

        decoration = self.get_atomic_numbers()
        sigmas = np.zeros(len(tags),dtype=np.int8)
        for tag, nsub in nsubs.items():
            #list all atom indices with the given tag or site_type
            sub_idxs = np.where(tags==tag)[0]

            #select nsub such indices for each substitutional
            #atom of given site tag
            sub_list = np.asarray([])
            for i,n in enumerate(nsub):
                sub_idxs = np.setdiff1d(sub_idxs,sub_list)
                sub_list = np.random.choice(sub_idxs,n,replace=False)

                for atom_index in sub_list:
                    decoration[atom_index] = idx_subs[tag][i+1]
                    sigmas[atom_index] = i+1
        return decoration, sigmas


    def enumerate_decorations(self, npoints=None, radii=None):
        from ase.db.jsondb import JSONDatabase
        from ase.neighborlist import NeighborList
        from subprocess import call

        atoms = self.get_pristine()
        natoms = len(atoms)
        sites = self.get_sites()
        call(["rm","-f","neighbors.json"])
        atoms_db = JSONDatabase(filename="neighbors.json") # For visualization

        rmax = np.full(len(atoms),np.amax(radii)/2.0)
        nl = NeighborList(rmax, self_interaction=True, bothways=True, skin=0.0)
        nl.build(atoms)
        distances = atoms.get_all_distances(mic=True)

        for id1 in range(natoms):
            neigs = atoms.copy()
            indices, offsets = nl.get_neighbors(id1)

            for i, offset in zip(indices,offsets):
                #pos.append(atoms.positions[i] + np.dot(offset, atoms.get_cell()))
                #pos.append(atoms.positions[i])
                #chem_num.append(atoms.numbers[i])
                spi = len(sites[i])-1
                neigs[i].number = sites[i][spi]
                neigs[i].position = atoms.positions[i] + np.dot(offset, atoms.get_cell())


            #neigs = Atoms(numbers=chem_num,positions=pos,cell=self.get_cell(),pbc=self.get_pbc())

            atoms_db.write(neigs)

    def serialize(self,filepath="scell.json", fname=None):
        """
        Serialize a SuperCell object

        Writes a `ASE's json database <https://wiki.fysik.dtu.dk/ase/ase/db/db.html>`_
        file containing a representation of the super cell.
        The created database can be visualized using
        `ASE's gui <https://wiki.fysik.dtu.dk/ase/ase/gui/gui.html>`_, e.g.: ::

            ase gui scell.json

        **Parameters:**

        ``filepath``: string
            Output file name.

        ``fname``: string
            *DEPRECATED*, use filepath instead. Output file name.
        """
        if fname is not None:
            filepath = fname
            
        super(SuperCell,self).serialize(filepath = filepath)
    
    def get_symmetry_table(self, symprec=1e-12, tol=1e-3):
        '''
        Takes a SuperCell as a parameter and returns the final indices of every atom 
        after all symmetry operations have been applied, see below:
        {(index of the atom, index of the symmetry operation): index of the atom after the symmetry operation}
        '''
        
        import numpy as np
        from clusterx.parent_lattice import ParentLattice
        from clusterx.symmetry import get_scaled_positions, wrap_scaled_positions, get_internal_translations
        from clusterx.utils import get_cl_idx_sc
        
        table = {}
        space_group, symmetry = self._plat.get_sym()
        rotations = symmetry['rotations']
        translations = symmetry['translations']
        internal_trans = get_internal_translations(self._plat, self)
        #Get rotations and translations
        pos = self.get_positions(wrap=True)
        p0 = pos
        
        spos1 = self.get_scaled_positions() # Super-cell scaled positions
        spos = np.around(spos1,8) 
        for i, periodic in enumerate(self.get_pbc()):
            if periodic:
                spos[:, i] %= 1.0
                spos[:, i] %= 1.0
        
        sp0 = get_scaled_positions(p0, self._plat.get_cell(), pbc = self.get_pbc(), wrap = False)
        
        atom_index = 0
        rot_index = 0
        intt_index = 0
        for r,t in zip(rotations,translations):
            ts = np.tile(t,(len(sp0),1)).T # Every column represents the same translation for every cluster site
            sp1 = np.add(np.dot(r,sp0.T),ts).T # Apply rotation, then translation
            # Get cartesian, then scaled to supercell
            p1 = np.dot(sp1, self._plat.get_cell())
            sp1 = get_scaled_positions(p1, self.get_cell(), pbc = self.get_pbc(), wrap = True)
        
            for itr,tr in enumerate(internal_trans): # Now apply the internal translations
                sp2 = np.add(sp1, tr)
                sp2 = wrap_scaled_positions(sp2,self.get_pbc())
                new_pos = get_cl_idx_sc(sp2, spos, method=1, tol=1e-3) #Get indices
                for new_idx in new_pos:
                    table[atom_index, rot_index, intt_index] = new_idx #Create the table object
                    atom_index += 1
                atom_index = 0
                intt_index += 1
            intt_index = 0
            rot_index += 1
            
        return table
