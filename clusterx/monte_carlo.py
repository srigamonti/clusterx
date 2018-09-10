## packages needed for MonteCarlo 
import random
from clusterx.structures_set import StructuresSet
from clusterx.structure import Structure
## packages needed for MonteCarloTrajectory
from ase.db.jsondb import JSONDatabase
#from ase.db.core import Database
import json
import numpy as np
from copy import deepcopy

class MonteCarlo():
    """MonteCarlo class
    
    Description:
        Perform MonteCarlo samplings

    Parameters:
    
    ``energy_model``: Model object
        Model used for acceptance and rejection. Usually, the Model enables to calculate the total energy of a given configuration. 

    ``scell``: SuperCell object
        Super cell in which the sampling is performed.

    ``nsub``: dictionary
        The format of the dictionary is as follows::

            {site_type1:[n11,n12,...], site_type2:[n21,n22,...], ...}
    
        It indicates how many substitutional atoms of every kind (n#1, n#2, ...)                                                                               
        may replace the pristine species for sites of site_type#.

        Defines the number of substituted atoms in each sublattice 
        Supercell in which the sampling is performed.

    ``models``: Model object

    ``ensemble``: String
        "canonical" allows only for swapping of atoms inside scell
        "grandcanonical"  allows for replacing atoms within the given scell 
            (the number of subsitutents in each sub-lattice is not kept)

    .. todo:
        Samplings in the grand canonical ensemble are not yet possible.

        Optional parameter for calculating properties with other ce_models during the sampling or after.
        For the moment, it is calculated after the sampling.


    """
    
    def __init__(self, energy_model, scell, nsubs, models = [], filename = "trajectory.json", ensemble = "canonical", no_of_swaps = 1):
        self._em = energy_model
        self._scell = scell
        self._nsubs = nsubs
        self._models = models
        self._filename = filename
        self._ensemble = ensemble
        self._no_of_swaps = no_of_swaps

    def metropolis(self, scale_factors, nmc, initial_structure = None, write_to_db = False):
        """Perform metropolis simulation
        
        **Description**: Perfom Metropolis sampling for nmc sampling steps at temperature temp in the super cell scell.
        
        **Parameters**:

        ``nmc``: integer
            Number of sampling steps
        
        ``temp``: integer
            Temperature in the Boltzmann distribution exp(-E/(kb*temp)) defining the acceptance probability

        ``initial_structure``: Structure object
            Sampling starts with the structure defined by this Structure object. 
            If initial_structure = None: Sampling starts with a structure randomly generated.

        ``write_to_db``: boolean (default: False)                                                                                                                                                                                                                             
            Whether to add the structure to the json database (see ``filename`` parameter for MonteCarloTrajectory initialization)

        **Returns**: MonteCarloTrajectory object
            Trajecotoy containing all decorations visited during the sampling
           
        """
        import math

        scale_factor_product = 1
        for el in scale_factors:
            scale_factor_product *= float(el)
        
        if initial_structure == None:
            struc = self._scell.gen_random(self._nsubs)
        
        e = self._em.predict_prop(struc)

        traj = MonteCarloTrajectory(self._scell, filename=self._filename, models=self._models)
        
        traj.add_decoration(struc, 0, e)
                
        for i in range(1,nmc+1):
            ind1_list = []
            ind2_list = []
            for j in range(self._no_of_swaps):
                ind1,ind2 = struc.swap_random_binary(0)
                ind1_list.append(ind1)
                ind2_list.append(ind2)
                
            e1 = self._em.predict_prop(struc)

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
                traj.add_decoration(struc, i, e)

            else:
                for j in range(self._no_of_swaps):
                    m=int(self._no_of_swaps-j)
                    ind1 = ind1_list[j]
                    ind2 = ind2_list[j]
                    struc.swap(0,ind1,ind2)

        if len(self._models) > 0 :
            traj.calculate_model_properties(self._models)
                    
        if write_to_db:
            traj.write_to_file()

        return traj                                        

class MonteCarloTrajectory():
    """MonteCarloTrajectory class
    
    **Description**:
        Trajectory of decorations visited during the sampling performed in the supercell scell. 
        For each visited decoration, the sampling step no (sampling_step_no), the decoration (decor), the total energy predicted 
        the cluster expansion model (ce_energy), and additional properties stored as element in dictionary key_value_pairs is stored.

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
        
    """
    
    def __init__(self, scell, filename="trajectory.json", **kwargs):
        self._trajectory = []
        
        self._scell = scell
        self._save_nsteps = kwargs.pop("save_nsteps",1000000)
        
        self._save_nsteps = kwargs.pop("models",[])

        self._filename = filename
        self.json_db = JSONDatabase(filename=self._filename)
        
    def calculate_model_properties(self, models):
        """
        Calculate the property for all decoration in the trajectory
        """
        sx=Structure(self._scell, decoration = self._trajectory[0]['decoration'])
                     
        for t,tr in enumerate(self._trajectory):
            sx.update_decoration(decoration = tr['decoration'])
            sdict={}
            for m,mo in enumerate(models):
                sdict.update({mo._prop: mo.predict_prop(sx)})

            self._trajectory[t]['key_value_pairs'] = sdict
                
    def add_decoration(self, struc, step, energy, key_value_pairs={}):
        """
        Add a decoration to the trajectory
        """
        self._trajectory.append(dict([('sampling_step_no',int(step)), ('decoration', deepcopy(struc.decor) ), ('model_total_energy',energy), ('key_value_pairs', key_value_pairs)]))

    def get_sampling_step_entries_at_step(self, nstep):
        """Get the dictionary at the n-th sampling step in the trajectory
        """
        return self.get_sampling_step_entries(self.get_id_sampling_step(nstep))

    def get_sampling_step_entries(self, nid):
        """Get the dictionary with index nid in the trajectory
        """

        return self._trajectory[nid]
        

    def get_decoration_at_step(self, nstep):
        """Get the decoration at the n-th sampling step in the trajectory.
            
        **Parameters:**

        ``nstep``: integer
            sampling step

        **Returns:**
            Decoration at the n-th sampling step.
                                                                                                                                 
        """        
        return self.get_decoration(self.get_id_sampling_step(nstep))


    def get_decoration(self, nid):
        """Get the decoration with the index nid in the trajectory
                                                                                                                                                                                                                             
        **Parameters:**

        ``nid``: integer
            index of structure in the trajectory.

        **Returns:**
            Decoration at index nid.
                                                                                                                                                                                                                                                     
        """
        return self._trajectory[nid]['decoration']
        
        #return Structure(self._scell, decoration = struc_parameters['decoration'])

    def get_sampling_step_nos(self):
        """Get sampling step numbers from the trajectory where the sampling accepts a new structure
        """
        steps=[]
        for tr in self._trajectory:
            steps.append(tr['sampling_step_no'])
            
        return steps

    def get_sampling_step_no(self, nid):
        """Get sampling step number trajectory element with index nid
        """
        return self._trajectory[nid]['sampling_step_no']


    def get_id_sampling_step(self, nstep):
        """Get decoration index at the n-th sampling step.
        """
        steps = self.get_sampling_step_nos()
        
        nid=0
        try:
            nid = steps.index(nstep)
        except ValueError:
            if nstep>steps[-1]:
                nid=steps[-1]
            else:
                for i,s in enumerate(steps):
                    if s>nstep:
                        nid=steps[i-1]
                    
        return nid


    def get_model_total_energies(self):
        """Get cluster expansion energies of the full trajectory
        """
        energies = []
        for tr in self._trajectory:
            energies.append(tr['model_total_energy'])
 
        return energies

    def get_model_total_energy(self, nid):
        """Get cluster expansion energy of decoration with index nid
        """
        return self._trajectory[nid]['model_total_energy']

    def get_model_properties(self, prop):
        """Get property predicted by a cluster expansion model of the full trajectory
        """
        props = []
        for tr in self._trajectory:
            props.append(tr['key_value_pairs'][prop])
 
        return props
    
    def get_model_property(self, prop):
        """Get property predicted by a cluster expansion model of decoration with index nid
        """        
        return self._trajectory[nid]['key_value_pairs'][prop]

    
    def get_id(self, prop, value):
        """Get indizes of decorations from the trajectory which contain the key-value pair trajectory
                                                                                                                                                                                                                             
        **Parameters:**

        ``prop``: string
            property of interest 

        ``value``: 
            value/values of the property

        **Returns:**
            array of decoration IDs for which the value of the property prop matches
                                                                                                                                                                                                                                                     
        """
        arrayid = []

        for i,tr in enumerate(self._trajectory):
            if tr[prop] == value:
                arrayid.append(i)

        return arrayid

    def write_to_file(self, filename = None):
        """Write trajectory to file
        """
        if filename is not None:
            self._filename = filename
            self.json_db = JSONDatabase(filename = self._filename)
            
        for j,dec in enumerate(self._trajectory):
            self.write(dec)
            
    def write(self, decor):
        
        self.json_db.write(decor)

