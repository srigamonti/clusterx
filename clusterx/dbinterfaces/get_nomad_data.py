""""
This module is designed to extract data from NOMAD or
a NOMAD OASIS database (such as total energy values, structures, etc.)
and use this data to generate structure sets along with their
associated properties in CELL, which can then be used to create models.

It is especially useful if one groups the data that should be used
for one model into a NOMAD dataset and then uses this module to
retrieve the properties using the dataset id.
"""
import json
import requests
import numpy as np
from ase import Atoms
from scipy.constants import electron_volt
from clusterx.structure import Structure


BASE_URL_NOMAD = 'http://nomad-lab.eu/prod/v1/api/v1/'


class SingleEntry:
    """"
    This class is applicable for single entries in NOMAD.
    Using its entry id the data of the entry can be accessed.

    **Parameters:**
    ``entry_id``: *string*
        ID of the entry in the ONOMAD database

    ``token``: *string*
        Default: None
        This only has to be overwritten if you want to access a NOMAD OASIS
        database instead of NOMAD.
        Refers to the access token for the OASIS which you can
        create if you have a registered account that has access to
        the OASIS.
        You may use the function get_nomad_oasis_access_token to retrieve it or
        create the token by using
        'https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#/auth/get_token_via_query_auth_token_get'

    ``base_url``: *string*
        base URL of the OASIS database
        The default is the base URL for NOMAD:
        'http://nomad-lab.eu/prod/v1/api/v1/'
        You may use the base URL of your OASIS instead.
    """
    def __init__(self, entry_id, token=None, base_url=BASE_URL_NOMAD):
        self.entry_id = entry_id
        self.token = token
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {self.token}'} if self.token is not None else {}

    def get_archive(self):
        """"
        Returns the archive of a single entry in OASIS.
        """
        # defines how the data should be queried
        query_type = {
            'required': '*'
        }
        # The entry archive of the entry with the corresponding entry id is
        # queried according to the query_type to get a repsonse.
        # The section archive of the data of the response is stored as json
        # in archive.
        request_url = f'{self.base_url}entries/{self.entry_id}/archive/query'
        try:
            response = requests.post(
                                request_url,
                                headers=self.headers,
                                json=query_type
                                )
            # check for HTTPErrors
            # if the response is successfull, this will not raise anything
            response.raise_for_status()
            response_json = response.json()
            archive = response_json['data']['archive']
            return archive
   
        except requests.exceptions.RequestException as error:
            print('The request was not successful.'
                  'Check the NOMAD docs for information on the error code:')
            # check post /entries/{entry_id}/archive/query for explanation of error codes
            print('https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#')
            print(f'The following error occured: {error}')

    def get_total_energy(self):
        """"
        Returns the total energy value in eV for a single entry in NOMAD
        """
        archive = self.get_archive()
        # To get the converged energy result access the last calculation
        result = archive['run'][0]['calculation'][-1]
        # if the last calculation does not calculate the total energy
        # (it could be a DOS calculation instead)
        # access the result from the previous calculation
        if 'total' not in result['energy']:
            result = archive['run'][0]['calculation'][-2]
        # convert the total energy value to eV
        total_energy_ev = result['energy']['total']['value']*1/electron_volt
        return total_energy_ev

    def get_atoms_object(self):
        """"
        Returns an ase Atoms object from the data stored in a NOMAD entry
        """
        archive = self.get_archive()
        # get the last entry for system to ensure to get the relaxed geometry
        nomad_atoms = archive['run'][0]['system'][-1]['atoms']
        positions_meter = nomad_atoms['positions']
        cell_meter = nomad_atoms['lattice_vectors']
        atoms = Atoms(
            symbols=nomad_atoms['labels'],
            positions=np.array(positions_meter)*10**10,  # convert to Angstrom
            cell=np.array(cell_meter)*10**10,  # convert to Angstrom
            pbc=nomad_atoms['periodic']
        )
        return atoms

    def get_structure_object(self, filename: str = 'structure.json'):
        """
        Returns the CELL structure object of the entry.
        This requires that the structure object is saved as json file in the
        same directory as the associated run before uploading to NOMAD.

        **Parameters:**
        ``filename``: *string*
            Name of the structure object json file.
            Default: 'structure.json'
        """
        request_url = f'{self.base_url}entries/{self.entry_id}/raw/{filename}'
        try:
            response = requests.get(request_url,
                                    headers=self.headers)
            # check for HTTPErrors
            response.raise_for_status()
            # load json file as json dictionary
            structure_json_dict = json.loads(response.text)
            # build structures object from json dictionary
            structure_object = Structure.from_dict(structure_json_dict)
            return structure_object
        except requests.exceptions.RequestException as error:
            print('The request was not successful.'
                  'Check the NOMAD docs for information on the error code:')
            print('https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#')
            print(f'The following error occured: {error}')


class Dataset:
    """"
    This class is applicable for datasets in NOMAD.
    Using its dataset id the data of all entries inside the dataset
    can be accessed.

    **Parameters:**
    ``dataset_id``: *string*
        Rrefers to the dataset id that is assigned to the dataset in NOMAD

    ``pagination_page_size``: *int*
        Maximum number of items contained in one response
        Set this to number of entries in dataset or lower to retrieve only a
        subset of the entries. In case of timout errors try if decreasing
        the pagination_page_size solves your problem

    ``token``: *string*
        Default ''
        This only has to be overwritten if you want to access a NOMAD OASIS
        database instead of NOMAD.
        Refers to the access token for the OASIS which you can
        create if you have a registered account that has access to
        the OASIS.
        You may use the function get_access_token to retrieve it or
        create the token by using
        'https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#/auth/get_token_via_query_auth_token_get'

    ``base_url``: *string*
        base url of the OASIS database
        The default is the base url for NOMAD:
        'http://nomad-lab.eu/prod/v1/api/v1/'
        You may use the base url of your OASIS instead.
    """
    def __init__(self, dataset_id, pagination_page_size, token='',
                 base_url=BASE_URL_NOMAD, dataset_data=None):
        self.dataset_id = dataset_id
        self.token = token
        self.pagination_page_size = pagination_page_size
        self.base_url = base_url
        self.dataset_data = dataset_data
        self.headers = {'Authorization': f'Bearer {self.token}'}

    def get_data(self):
        """"
        Returns the data section of the dataset
        """
        query_type = {
            # solves visibility issues; enables you to see
            # published entries without embargo or unpublished entries
            # that belong to you or are shared with you
            'owner': 'visible',
            'query': {'datasets.dataset_id': self.dataset_id},
            'pagination': {'page_size': self.pagination_page_size}
        }
        request_url = f'{self.base_url}entries/archive/query'

        try:
            # The OASIS entries are queried according to the query_type above
            # The response is then stored as json
            # The section data is returned
            response = requests.post(request_url,
                                     headers=self.headers,
                                     json=query_type)
            response.raise_for_status()
            dataset = response.json()
            return dataset['data']

        except requests.exceptions.RequestException as error:
            print('The request was not successful.'
                  'Check the NOMAD docs for information on the error code:')
            print('https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#')
            print(f'The following error occured: {error}')

    def download_data(self):
        """
        Saves the data section of the dataset by either
        retrieving the data for the first time or caching and reusing it
        when called again
        """
        if self.dataset_data is None:
            print("Accessing NOMAD to retrieve data")
            self.dataset_data = self.get_data()
        else:
            print("Using cached data")
        return self.dataset_data

    def get_entry_ids(self):
        """"
        Get entry ids of the entries contained in the dataset
        """
        entry_ids = [
            entry['archive']['metadata']['entry_id']
            for entry in self.download_data()
        ]
        return entry_ids

    def get_total_energies(self):
        """
        Returns a list of the total energy values in eV
        of the entries contained in the dataset
        """
        energy_values_ev = []
        for entry in self.download_data():
            # To get the converged energy result access the last calculation
            result = entry['archive']['run'][0]['calculation'][-1]
        # if the last calculation does not calculate the total energy
        # (it could be a DOS calculation instead)
        # access the result from the previous calculation
            if 'energy' in result and 'total' in result['energy']:
                total_energy = result['energy']['total']['value'] * 1/electron_volt
            else:
                result = entry['archive']['run'][0]['calculation'][-2]
                total_energy = result['energy']['total']['value'] * 1/electron_volt
            energy_values_ev.append(total_energy)
        return energy_values_ev

    def get_structure_objects(self, filename='structure.json'):
        """
        Returns a list of structure objects for the entrys in the dataset.
        This requires that the structure objects are saved as json files in the
        same directory as the associated runs before uploading to NOMAD.
        The structure objects can also have a different filename, but it has to be 
        the same filename for every entry in the dataset. If it is not, please use
        get_structure_object of the SingleEntry class and specify the filename for 
        each entry separately.

        **Parameters:**
        ``filename``: *string*
            Name of the structure object json file.
            Default: 'structure.json'
        """
        list_of_entry_ids = self.get_entry_ids()
        list_of_structures = []
        try:
            for entry_id in (list_of_entry_ids):
                request_url = f'{self.base_url}entries/{entry_id}/raw/{filename}'
                response = requests.get(request_url,
                                        headers=self.headers)
                # check for HTTPErrors
                response.raise_for_status()
                # load response content as json dictionary
                structure_json_dict = json.loads(response.text)
                # build structure object from json dictionary
                structure_object = Structure.from_dict(structure_json_dict)
                # append structure to list of structures of the dataset
                list_of_structures.append(structure_object)
            return list_of_structures
        except requests.exceptions.RequestException as error:
            print('The request was not successful.'
                  'Check the NOMAD docs for information on the error code:')
            print('https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#')
            print(f'The following error occured: {error}')
