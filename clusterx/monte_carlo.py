## packages needed for MonteCarlo
import random
#from clusterx.structures_set import StructuresSet
from clusterx.structure import Structure
## packages needed for MonteCarloTrajectory
import json
import numpy as np
from copy import deepcopy


class MonteCarlo():
    """MonteCarlo class

    Description:
        Perform MonteCarlo samplings

    Parameters:

    ``energy_model``: Model object
        Model used for acceptance and rejection. Usually, the Model enables to
        calculate the total energy of a given configuration.

    ``scell``: SuperCell object
        Super cell in which the sampling is performed.

    ``nsubs``: dictionary
        The format of the dictionary is as follows::

            {site_type1:[n11,n12,...], site_type2:[n21,n22,...], ...}

        It indicates how many substitutional atoms of every kind (n#1, n#2, ...)
        may replace the pristine species for sites of site_type#.

        The list of site types can be obtained with the method ``ParentLattice.get_idx_subs()``
        (see related documentation).

        Defines the number of substituted atoms in each sublattice
        Supercell in which the sampling is performed.

    ``filename``: string
        Trajectory can be written to a json file with the name ``filename``.

    ``sublattice_indices``: list of int
        Sampled sublattices. Each index in the list gives the site_type defining the sublattice.
        If the list is empty (default), the site_type of the sublattices are read from ``nsubs``
        Non-substituted sublattices are excluded for canonical samplings.

    ``models``: Model object
        List of Models for structural dependent properties. Models of properties
        can also be defined and calculated after the sampling is finished
        by using the Class MonteCarloTrajectory

    ``no_of_swaps``: int
        Number of swaps per sampling step

    ``ensemble``: string
        "canonical" allows only for swapping of atoms inside scell
        "grandcanonical"  allows for replacing atoms within the given scell
            (the number of substitutents in each sublattice is not kept)

    .. todo:
        Samplings in the grand canonical ensemble are not yet possible.

        Properties given in models are not calculated during the sampling.


    """

    def __init__(self, energy_model, scell, nsubs, filename = "trajectory.json", last_visited_structure_name = "last-visited-structure-mc.json", sublattice_indices = [], models = [], no_of_swaps = 1, ensemble = "canonical"):
        self._em = energy_model
        self._scell = scell
        self._nsubs = nsubs
        print(self._nsubs)
        self._filename = filename
        self._last_visited_structure_name = last_visited_structure_name

        if not sublattice_indices:
            try:
                self._sublattice_indices = [k for k in self._nsubs.keys()]

                if ensemble == "canonical":
                    for key in self._nsubs.keys():
                        if all([ subs == 0 for subs in self._nsubs[key] ]):
                            self._sublattice_indices.remove(key)


            except AttributeError:
                raise AttributeError("Index of sublattice is not properly assigned, look at the documentation.")
        else:
            self._sublattice_indices = sublattice_indices

        if not self._sublattice_indices:
            import sys
            sys.exit('Indices of sublattice are not porperly assigned, look at the documatation.')

        self._models = []
        if models:
            self._models = models

        self._ensemble = ensemble
        self._no_of_swaps = no_of_swaps

    def metropolis(self, scale_factor, nmc, initial_decoration = None, write_to_db = False):
        """Perform metropolis simulation

        **Description**: Perfom Metropolis sampling for nmc sampling
             steps at scale factor :math:`k_B T`.  The total energy
             :math:`E` for visited structures in the sampling is
             calculated from the Model ``energ_model`` of the total
             energy. During the sampling, a new structure is accepted
             with the probability given by::

                 :math:`min \big( 1, \exp( - E / ( k_B T ) ) \big)`

        **Parameters**:

        ``scale_factors``: list of floats
            From the product of the float in the list, the scale factor for the energy :math:`k_B T` is obtained.

            E.g. [:math:`k_B`, :math:`T`] with :math:`k_B`:: as the Boltzmann constant and :math:`T` as the temperature for the Metropolis simulation.
            The product :math:`k_B T` defines the scale factor in the Boltzmann distribution.

            Note: The unit of the product :math:`k_B T` and :math:`T` must be the same as for the total energy :math:`E`.

        ``nmc``: integer
            Number of sampling steps

        ``initial_decoration``: Structure object
            Sampling starts with the structure defined by this Structure object.
            If initial_structure = None: Sampling starts with a structure randomly generated.

        ``write_to_db``: boolean (default: False)                                                                                                                                                                                                                             
            Whether to add the structure to the json database (see ``filename`` parameter for MonteCarloTrajectory initialization)

        **Returns**: MonteCarloTrajectory object
            Trajecotoy containing all decorations visited during the sampling

        """
        import math

        scale_factor_product = 1
        for el in scale_factor:
            scale_factor_product *= float(el)

        if initial_decoration is not None:
            print('1')
            struc = Structure(self._scell, initial_decoration)
        else:
            print('0')
            struc = self._scell.gen_random(self._nsubs)
        
        e = self._em.predict(struc)
        
        traj = MonteCarloTrajectory(self._scell, filename=self._filename, models = self._models)
    
        if self._models:
            key_value_pairs = {}
            for m, mo in enumerate(self._models):
                key_value_pairs.update({mo.property: mo.predict(struc)})
            traj.add_decoration(0, e, [], decoration = struc.decor, key_value_pairs = key_value_pairs)

        else:
            traj.add_decoration(0, e, [], decoration = struc.decor)

        for i in range(1,nmc+1):
            indices_list = []
            for j in range(self._no_of_swaps):
                #ind1,ind2 = struc.swap_random_binary(self._sublattice_index)
                ind1, ind2 = struc.swap_random(self._sublattice_indices)
                indices_list.append([ind1, ind2])

            e1 = self._em.predict(struc)

            if e >= e1:
                accept_swap = True
                boltzmann_factor = 0
            else:
                boltzmann_factor = math.exp((e-e1)/(scale_factor_product))
                if np.random.uniform(0,1) <= boltzmann_factor:
                    accept_swap = True
                else:
                    accept_swap = False

            if accept_swap:
                e=e1

                if self._models:
                    key_value_pairs = {}
                    for m, mo in enumerate(self._models):
                        key_value_pairs.update({mo.property: mo.predict(struc)})
                    traj.add_decoration(i, e, indices_list, key_value_pairs = key_value_pairs)

                else:
                    traj.add_decoration(i, e, indices_list)

            else:
                for j in range(self._no_of_swaps-1,-1,-1):
                    struc.swap(indices_list[j][0],indices_list[j][1])

        if write_to_db:
            traj.write_to_file()
            struc.serialize(fname=self._last_visited_structure_name)

        return traj

class MonteCarloTrajectory():
    """MonteCarloTrajectory class

    **Description**:
        Trajectory of decorations visited during the sampling performed in the supercell scell.
        For each visited decoration, it is retained the sampling step number (sampling_step_no),
        the decoration (decor), the total energy predicted from a cluster expansion model (energy_model),
        and additional properties stored as key-value pair in dictionary key_value_pairs.

    **Parameters**:

    ``scell``: SuperCell object
        Super cell in which the sampling is performed.

    ``filename``: string
        The trajectoy can be stored in a json file with the path given by ``filename``.

    ``**kwargs``: keyword arguments

        ``save_nsteps``: integer
            Trajectory is saved after save_nsteps.

        ``models``: List of Model objects

    .. todo::
        Saving the trajectory after ``save_steps`` is not yet implemented.

        Improve appearance of json file - decoration array in one line

    """
    
    def __init__(self, scell, filename="trajectory.json", **kwargs):
        self._trajectory = []

        self._scell = scell
        self._save_nsteps = kwargs.pop("save_nsteps",1000000)

        self._models = kwargs.pop("models",[])

        self._filename = filename

    def calculate_model_properties(self, models):
        """Calculate the property for all decoration in the trajectory
        """
        for mo in models:
            if mo not in self._models:
                self._models.append(mo)
        print(self._models)
        sx = Structure(self._scell, decoration = self._trajectory[0]['decoration'])
        print(sx.decor)

        for t,tr in enumerate(self._trajectory):
            indices_list = tr['swapped_positions']
            for j in range(len(indices_list)):
                sx.swap(indices_list[j][0],indices_list[j][1])
                
            sdict={}
            for m,mo in enumerate(self._models):
                sdict.update({mo.property: mo.predict(sx)})

            self._trajectory[t]['key_value_pairs'] = sdict
        print(sx.decor)

    def add_decoration(self, step, energy, indices_list, decoration = None, key_value_pairs = {}):
        """Add decoration of Structure object to the trajectory
        """
        
        if indices_list:
            self._trajectory.append(dict([('sampling_step_no', int(step)), ('model_total_energy', energy), ('swapped_positions', indices_list), ('key_value_pairs', key_value_pairs)]))
        else:
            self._trajectory.append(dict([('sampling_step_no', int(step)), ('model_total_energy', energy), ('swapped_positions', [[0,0]]) , ('decoration', deepcopy(decoration)), ('key_value_pairs', key_value_pairs)]))
            

    def get_sampling_step_entry_at_step(self, nstep):
        """Get the dictionary at the n-th sampling step in the trajectory

        **Parameters:**

        ``nstep``: int
            sampling step

        **Returns:**
            Dictionary at the n-th sampling step.

        """
        return self.get_sampling_step_entry(self.get_id_sampling_step(nstep))

    def get_sampling_step_entry(self, nid):
        """Get the dictionary (entry) at index nid in the trajectory
        """
        return self._trajectory[nid]

    def get_structure_at_step(self, nstep):
        """Get the structure from decoration at the n-th sampling step in the trajectory.

        **Parameters:**

        ``nstep``: integer
            sampling step

        **Returns:**
            Structure object at the n-th sampling step.

        """
        return self.get_structure(self.get_id_sampling_step(nstep))

    def get_structure(self, nid):
        """Get structure from entry at index nid in the trajectory

        **Parameters:**

        ``nid``: integer
            index of structure in the trajectory.

        **Returns:**
            Structure object at index nid.

        """
        sx = Structure(self._scell, decoration = deepcopy(self._trajectory[0]['decoration']))

        for t,tr in enumerate(self._trajectory[0:nid+1]):
            indices_list = tr['swapped_positions']
            for j in range(len(indices_list)):
                sx.swap(indices_list[j][0],indices_list[j][1])
                
        return sx

    def get_lowest_non_degenerate_structure(self):
        """Get lowest-non-degenerate structure from trajectory

        **Returns:**
            Structure object.

        """
        _energies = self.get_model_total_energies()
        _emin = np.min(_energies)
        _nid = np.where(_energies == _emin)[0][0]

        return self.get_structure(_nid)

    def get_sampling_step_nos(self):
        """Get sampling step numbers of all entries in trajectory
        """
        steps=[]
        for tr in self._trajectory:
            steps.append(tr['sampling_step_no'])

        return np.int8(steps)

    def get_sampling_step_no(self, nid):
        """Get sampling step number of entry at index nid in trajectory
        """
        return self._trajectory[nid]['sampling_step_no']


    def get_id_sampling_step(self, nstep):
        """Get entry index at the n-th sampling step.
        """
        steps = self.get_sampling_step_nos()

        nid=0
        try:
            nid = np.where(steps == nstep)[0][0]
        except ValueError:
            if nstep > steps[-1]:
                nid = steps[-1]
            else:
                for i,s in enumerate(steps):
                    if s > nstep:
                        nid = steps[i-1]
        return nid

    def get_model_total_energies(self):
        """Get total energies of all entries in trajectory.
        """
        energies = []
        for tr in self._trajectory:
            energies.append(tr['model_total_energy'])

        return np.asarray(energies)

    def get_model_total_energy(self, nid):
        """Get total energy of entry at index nid in trajectory.
        """
        return self._trajectory[nid]['model_total_energy']

    def get_model_properties(self, prop):
        """Get property of all entries in the trajectory.
        """
        try:
            props = []
            for tr in self._trajectory:
                props.append(tr['key_value_pairs'][prop])
            return np.asarray(props)
        
        except:
            if prop not in [mo.property for mo in self._models]:
                print("Model of property is not given, look at the documentation.")
            else:
                print("Property not calculated, look at the documentation.")

    def get_model_property(self, nid, prop):
        """Get property of entry at index nid in trajectory.
        """
        return self._trajectory[nid]['key_value_pairs'][prop]

    def get_id(self, prop, value):
        """Get indices of entries in trajectory which contain the key-value pair trajectory.

        **Parameters:**

        ``prop``: string
            property of interest

        ``value``: float
            value of the property

        **Returns:**
            array of int.

        """
        arrayid = []

        if prop in ['sampling_step_no','swapped_positions','model_total_energy','decoration']:
            for i,tr in enumerate(self._trajectory):
                if tr[prop] == value:
                    arrayid.append(i)
        else:
            for i,tr in enumerate(self._trajectory):
                if tr['key_value_pairs'][prop] == value:
                    arrayid.append(i)

        return np.asarray(arrayid)

    def write_to_file(self, filename = None):
        """Write trajectory to file (default filename trajectory.json).
        """

        if filename is not None:
            self._filename = filename

        trajdic={}
        for j,dec in enumerate(self._trajectory):
            dec_string=""
            if j == 0:
                dec['decoration'] = dec['decoration'].tolist()
            #print(dec)
            #jsonindent=2, sort_keys=True


            #dsw=[]
            #for sw in dec['swapped_positions']:
            #    dsw.append(sw.tolist())
            #dec['swapped_positions'] = dsw
            trajdic.update({str(j):dec})

        with open(self._filename, 'w', encoding='utf-8') as outfile:
            json.dump(trajdic,outfile, cls=NumpyEncoder, indent = 1 , separators = (',',':'))

    def read(self, filename = None , append = False):
        """Read trajectory from file (default filename trajectory.json)
        """
        if filename is not None:
            trajfile = open(filename,'r')
        else:
            trajfile = open(self._filename,'r')

        data = json.load(trajfile)

        if not append:
            self._trajectory = []

        data_keys = sorted([int(el) for el in set(data.keys())])

        for key in data_keys:
            tr = data[str(key)]
            #tr['swapped_positions'] = np.asarray(tr['decoration'],dtype=np.int8)
            self._trajectory.append(tr)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types 
    https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable/32850511

    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): 
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
