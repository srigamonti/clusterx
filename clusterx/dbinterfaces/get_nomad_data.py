""""
This module is designed to obtain data from NOMAD or a NOMAD OASIS
and to make it usable for CELL.
It is especially useful if one groups the data that should be used 
for one model into a NOMAD dataset and then uses this module to 
retrieve the properties using the dataset id.
"""
import json
import requests
import numpy as np
from ase import Atoms


BASE_URL_SOL = 'https://sol-oasis.physik.hu-berlin.de/nomad-oasis/api/v1/'
BASE_URL_NOMAD = 'http://nomad-lab.eu/prod/v1/api/v1/'
e = 1.602176634*10**(-19)  # elementary charge in C
# as defined in NOMAD constants_en.txt


class SingleEntry:
    """"
    This class is applicable for single entries in the NOMAD OASIS.
    Using its entry id the data can be accessed.

    **Parameters:**
    ``entry_id``: *string*
        ID of the entry in the OASIS database

    ``token``: *string*
        Access token for the OASIS
        You may use the function get_access_token to retrieve it.

    ``base_url``: *string*
        base url of the OASIS database
        The default is the SOL OASIS:
        'https://sol-oasis.physik.hu-berlin.de/nomad-oasis/api/v1/'
        You may overwrite it with a different OASIS URL.
    """
    def __init__(self, entry_id, token, base_url=BASE_URL_SOL):
        self.entry_id = entry_id
        self.token = token
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {self.token}'}

    def get_archive(self):
        """"
        Returns the archive of a single entry in OASIS.
        """
        # defines how the data should be queried
        query_type = {
            'required': '*'
        }
        # The archive of the entry with the corresponding entry id is queried
        # according to the query_type to get a repsonse.
        # The section archive of the data of the response is stored as json
        # in archive.
        f_string = f'{self.base_url}entries/{self.entry_id}/archive/query'
        try:
            response = requests.post(
                                f_string,
                                headers=self.headers,
                                json=query_type
                                )
            response.raise_for_status()
            response_json = response.json()
            archive = response_json['data']['archive']
            return archive
        
        except requests.exceptions.RequestException as error:
            print('The request was not successful. Check the NOMAD docs:')
            print('https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#')
            print(f'The following error occured: {error}')

    def download_archive(self):
        """
        Downloads the archive section of data of a single entry
        """
        self.download = self.get_archive()
        return self.download

    def get_total_energy(self):
        """"
        Returns the total energy value in eV from a single entry in the OASIS
        """
        archive = self.download_archive()
        result = archive['run'][0]['calculation'][-1]
        if 'total' not in result['energy']:
            result = archive['run'][0]['calculation'][-2]
        total_energy_ev = result['energy']['total']['value']*1/e
        return total_energy_ev

    def get_atoms_object(self):
        """"
        Returns an ase Atoms object from the data stored in an OASIS entry
        """
        archive = self.download_archive()
        nomad_atoms = archive['run'][0]['system'][-1]['atoms']
        positions_meter = nomad_atoms['positions']
        cell_meter = nomad_atoms['lattice_vectors']
        atoms = Atoms(
            symbols=nomad_atoms['labels'],
            positions=np.array(positions_meter)*10**10,  # in Angstrom
            cell=np.array(cell_meter)*10**10,  # in Angstrom
            pbc=nomad_atoms['periodic']
        )
        return atoms

    def get_structure_object(self, filename='structure.json'):
        """
        Returns the structure object as json file.
        This requires that the structure object is saved as json file in the
        same directory as the converged run as filename.json.

        **Parameters:**
        ``filename``: *string*
            Name of the structure object file as filename.json
        """
        f_string = f'{self.base_url}entries/{self.entry_id}/raw/{filename}'
        try:
            response = requests.get(f_string,
                                    headers=self.headers)
            response.raise_for_status()
            structure_json = json.loads(response.text)
            return structure_json
        
        except requests.exceptions.RequestException as error:
            print('The request was not successful. Check the NOMAD docs:')
            print('https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#')
            print(f'The following error occured: {error}')

        # note:
        # Currently there is not yet an option in CELL to load a
        # structures object via Structure(filepath="structure.json")
        # When this is implemented the code here will be altered
        # to return the structures object using the json file


class Dataset:
    """"
    This class is applicable for datasets in the NOMAD OASIS.
    Using its dataset id the data of the entries inside the dataset
    can be accessed.

    **Parameters:**
    ``dataset_id``: *string*
        Rrefers to the dataset id that is assigned to the dataset in OASIS

    ``pagination_page_size``: *int*
        Maximum number of items contained in one response
        Set this to number of entries in dataset or lower to retrieve only a
        subset of the entries

    ``token``: *string*
        Access token for the OASIS
        You may use the function get_access_token to retrieve it.

    ``base_url``: *string*
        base url of the OASIS database
        The default is the SOL OASIS:
        'https://sol-oasis.physik.hu-berlin.de/nomad-oasis/api/v1/'
        You may overwrite it with a different OASIS URL.
    """
    def __init__(self, dataset_id, token, pagination_page_size,
                 base_url=BASE_URL_SOL, dataset_data=None):
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
            'owner': 'shared',
            'query': {'datasets.dataset_id': self.dataset_id},
            'pagination': {'page_size': self.pagination_page_size}
        }
        f_string = f'{self.base_url}entries/archive/query'
        try:
            # The OASIS entries are queried according to the query_type above
            # The response is then stored as json
            # The section data is returned
            response = requests.post(f_string,
                                     headers=self.headers,
                                     json=query_type)
            response.raise_for_status()
            dataset = response.json()
            return dataset['data']
        except requests.exceptions.RequestException as error:
            print('The request was not successful. Check the NOMAD docs:')
            print('https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#')
            print(f'The following error occured: {error}')

    def download_data(self):
        """
        Saves the data section of the dataset by either 
        retrieving the data for the first time or saving and reusing
        when called again
        """
        if self.dataset_data is None:
            print("Accessing NOMAD/OASIS to retrieve data")
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
            # Check if 'energy' and 'total' exist in the result
            # If DOS calculation was performed after structure optimization
            # you have to adjust the index
            if 'energy' in result and 'total' in result['energy']:
                total_energy = result['energy']['total']['value'] * 1/e
            else:
                result = entry['archive']['run'][0]['calculation'][-2]
                total_energy = result['energy']['total']['value'] * 1/e
            energy_values_ev.append(total_energy)
        return energy_values_ev

    def get_structure_objects(self, filename='structure.json'):
        """
        Returns the structure objects as list of json files.
        This requires that the structure object is saved as json file in the
        same directory as the converged run as filename.json.

        **Parameters:**
        ``filename``: *string*
            Name of the structure object file as filename.json
            The default is structure.json.
        """
        list_of_entry_ids = self.get_entry_ids()
        list_of_structures = []
        try:
            for entry_id in (list_of_entry_ids):
                f_string = f'{self.base_url}entries/{entry_id}/raw/{filename}'
                response = requests.get(f_string,
                                        headers=self.headers)
                response.raise_for_status()
                structure_json = json.loads(response.text)
                list_of_structures.append(structure_json)
            return list_of_structures
        except requests.exceptions.RequestException as error:
            print('The request was not successful. Check the NOMAD docs:')
            print('https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#')
            print(f'The following error occured: {error}')
