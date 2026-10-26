""""
This module is designed to extract data from NOMAD or
a NOMAD Oasis database. The data (e.g. materials properties, such as
total energy values) can be used to generate structure sets along with their
associated properties in CELL.

It is especially useful if one groups the calculations for structures
of a training set in one dataset. Then only the dataset id is required
to get property values for all structures in the dataset/training set.
"""

import json
import requests
import numpy as np
from ase import Atoms
from scipy.constants import electron_volt
from clusterx.structure import Structure


BASE_URL_NOMAD = "http://nomad-lab.eu/prod/v1/api/v1/"


def _get_lattice_parameters(cell, sort_by_length=False):
    """Return cell lengths and angles, optionally ordering vectors by length."""
    cell = np.asarray(cell)
    if sort_by_length:
        cell = cell[np.argsort(np.linalg.norm(cell, axis=1))]

    a, b, c = np.linalg.norm(cell, axis=1)

    def angle(vector_1, vector_2):
        cosine = np.dot(vector_1, vector_2) / (
            np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
        )
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    alpha = angle(cell[1], cell[2])
    beta = angle(cell[0], cell[2])
    gamma = angle(cell[0], cell[1])
    return a, b, c, alpha, beta, gamma


def _collect_lattice_parameters(cells, sort_by_length=False):
    """Collect cell parameters into six lists, one for each lattice property."""
    parameters = [[] for _ in range(6)]
    for cell in cells:
        for values, value in zip(parameters, _get_lattice_parameters(cell, sort_by_length)):
            values.append(value)
    return tuple(parameters)


class SingleEntry:
    """ "
    This class is applicable for single entries in NOMAD.
    Using its entry id the data of the entry can be accessed.

    **Parameters:**
    ``entry_id``: *string*
        ID of the entry in the ONOMAD database

    ``token``: *string*
        Default: None
        This only has to be overwritten if you want to access a NOMAD OASIS
        database or private data in NOMAD.
        Refers to the access token, which you can
        create if you have a registered account that has access to
        the NOMAD OASIS or the private data.
        You may use the function get_nomad_oasis_access_token to retrieve it.

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
        self.headers = (
            {"Authorization": f"Bearer {self.token}"} if self.token is not None else {}
        )

    def get_archive(self):
        """
        Returns the archive of a single entry in NOMAD.
        """
        # defines how the data should be queried
        query_type = {"required": "*"}
        # The archive of the entry with the corresponding entry id is
        # queried according to the query_type to get a repsonse.
        # The section 'archive' of the response data is stored as archive in json format.
        request_url = f"{self.base_url}entries/{self.entry_id}/archive/query"
        try:
            response = requests.post(request_url, headers=self.headers, json=query_type)
            # check for HTTPErrors
            # if the response is successfull, this will not raise anything
            response.raise_for_status()
            response_json = response.json()
            archive = response_json["data"]["archive"]
            return archive

        except requests.exceptions.RequestException as error:
            print(
                "The request was not successful."
                "Check the NOMAD docs for information on the error code:"
            )
            # check post /entries/{entry_id}/archive/query for explanation of error codes
            print("https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#")
            print(f"The following error occured: {error}")

    def get_total_energy(self):
        """
        Returns the total energy value in eV for a single NOMAD entry
        """
        archive = self.get_archive()
        # To get the converged energy result call the last calculation
        result = archive["run"][0]["calculation"][-1]
        # if the last calculation does not calculate the total energy
        # (it could be a DOS calculation instead)
        # access the result from the previous calculation
        if "total" not in result["energy"]:
            result = archive["run"][0]["calculation"][-2]
            if "total" not in result["energy"]:
                raise ValueError("No total energy result found in the up to two last calculations.")
        # convert the total energy value to eV
        total_energy_ev = result["energy"]["total"]["value"] * 1/electron_volt
        return total_energy_ev

    def get_atoms_object(self):
        """
        Returns an ase Atoms object from the data stored in a NOMAD entry.

        This is useful, if you do not upload CELL structure objects
        together with your entries and do not want to use get_structure_objects.
        If you use get_structure_objects, you can access the ase Atoms objects
        directly frum the structure objects, by calling structure.get_atoms().
        """
        archive = self.get_archive()
        # get the last entry to ensure to get the relaxed geometry
        nomad_atoms = archive["run"][0]["system"][-1]["atoms"]
        positions_meter = nomad_atoms["positions"]
        cell_meter = nomad_atoms["lattice_vectors"]
        atoms = Atoms(
            symbols=nomad_atoms["labels"],
            positions=np.array(positions_meter) * 10**10,  # convert to Angstrom
            cell=np.array(cell_meter) * 10**10,  # convert to Angstrom
            pbc=nomad_atoms["periodic"],
        )
        return atoms

    def get_structure_object(self, filename: str = "structure.json"):
        """
        Returns the CELL structure object of a NOMAD entry.
        This requires that the structure object is saved as json file in the
        same directory as the associated run before uploading to NOMAD.

        **Parameters:**
        ``filename``: *string*
            Name of the structure object json file.
            Default: 'structure.json'
        """
        request_url = f"{self.base_url}entries/{self.entry_id}/raw/{filename}"
        try:
            response = requests.get(request_url, headers=self.headers)
            # check for HTTPErrors
            response.raise_for_status()
            # load json file as json dictionary
            structure_json_dict = json.loads(response.text)
            # build structures object from json dictionary
            structure_object = Structure.from_dict(structure_json_dict)
            return structure_object
        except requests.exceptions.RequestException as error:
            print(
                "The request was not successful."
                "Check the NOMAD docs for information on the error code:"
            )
            print("https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#")
            print(f"The following error occured: {error}")

    def get_lattice_constant(self, normalize=False):
        """"
        Returns the three lattice constants of the structure.
        The smallest lattice constant is returned first,
        the largest is returned last.

        **Parameters:**
        ``normalize``: bool
            Default:False
            If set to True the lattice constants are normalized to the
            parent lattice.
        """
        if not normalize:
            atoms_object = self.get_atoms_object()
            cell = atoms_object.cell
            lattice_constant_1 = np.sqrt(cell[0][0]**2+cell[0][1]**2+cell[0][2]**2)
            lattice_constant_2 = np.sqrt(cell[1][0]**2+cell[1][1]**2+cell[1][2]**2)
            lattice_constant_3 = np.sqrt(cell[2][0]**2+cell[2][1]**2+cell[2][2]**2)
        else:
            structure = self.get_structure_object()
            atoms_object = self.get_atoms_object()
            cell = atoms_object.cell
            # get transformation from parent lattice to super cell
            transformation_matrix = structure.get_supercell().get_transformation()
            # apply inverse transformation matrix to cell
            normalized_cell = np.dot(np.linalg.inv(transformation_matrix), cell)
            # obtain lattice constants normalized to parent lattice
            lattice_constant_1 = np.sqrt(normalized_cell[0][0]**2+normalized_cell[0][1]**2+normalized_cell[0][2]**2)
            lattice_constant_2 = np.sqrt(normalized_cell[1][0]**2+normalized_cell[1][1]**2+normalized_cell[1][2]**2)
            lattice_constant_3 = np.sqrt(normalized_cell[2][0]**2+normalized_cell[2][1]**2+normalized_cell[2][2]**2)
            # sort lattice constants such that smalles lattice constant is returned first,
            # and largest last
        a, b, c = np.sort([lattice_constant_1, lattice_constant_2, lattice_constant_3])
        return a, b, c


class Dataset:
    """
    This class is applicable to datasets in NOMAD.
    Using its dataset id the data of all entries inside the dataset
    can be accessed.

    **Parameters:**
    ``dataset_id``: *string*
        Rrefers to the dataset id that is assigned to the dataset in NOMAD

    ``pagination_page_size``: *int*
        Maximum number of items contained in one response
        Set this to the number of entries in the dataset or lower to retrieve only a
        subset of the entries. In case of timeout errors try if decreasing
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
        base URL of the OASIS database
        The default is the base URL for NOMAD:
        'http://nomad-lab.eu/prod/v1/api/v1/'
        You may use the base URL of your OASIS instead.
    """

    def __init__(
        self,
        dataset_id,
        pagination_page_size,
        token="",
        base_url=BASE_URL_NOMAD,
        dataset_data=None,
    ):
        self.dataset_id = dataset_id
        self.token = token
        self.pagination_page_size = pagination_page_size
        self.base_url = base_url
        self.dataset_data = dataset_data
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def get_data(self):
        """
        Returns the data section of the dataset
        """
        query_type = {
            # set "owner": "visible" to ensure that you can get
            # entries of yours/shared with you
            # even when they are not yet published
            "owner": "visible",
            "query": {"datasets.dataset_id": self.dataset_id},
            "pagination": {"page_size": self.pagination_page_size},
        }
        request_url = f"{self.base_url}entries/archive/query"

        try:
            # The OASIS entries are queried according to the query_type above
            # The response is then stored as json
            # The section data is returned
            response = requests.post(request_url, headers=self.headers, json=query_type)
            response.raise_for_status()
            dataset = response.json()
            return dataset["data"]

        except requests.exceptions.RequestException as error:
            print(
                "The request was not successful."
                "Check the NOMAD docs for information on the error code:"
            )
            print("https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#")
            print(f"The following error occured: {error}")

    def download_data(self):
        """
        Saves the data section of the dataset by either
        retrieving the data for the first time or caching and reusing it
        when called again.
        """
        if self.dataset_data is None:
            print("Accessing NOMAD to retrieve data")
            self.dataset_data = self.get_data()
        else:
            print("Using cached data")
        return self.dataset_data

    def get_entry_ids(self):
        """
        Get entry ids of the entries contained in the NOMAD dataset.
        """
        entry_ids = [
            entry["archive"]["metadata"]["entry_id"] for entry in self.download_data()
        ]
        return entry_ids

    def get_total_energies(self):
        """
        Returns a list of the total energy values in eV
        of the entries contained in the dataset
        """
        energy_values_ev = []
        for entry in self.download_data():
            # To get the converged energy result call upon the last calculation
            result = entry["archive"]["run"][0]["calculation"][-1]
            # check if total energy is stored in last calculation
            if "total" not in result["energy"]:
                result = entry["archive"]["run"][0]["calculation"][-2]
                if "total" not in result["energy"]:
                    raise ValueError("No total energy result found in the up to two last calculations.")
            # convert total energy to eV
            total_energy_ev = result["energy"]["total"]["value"] * 1/electron_volt
            energy_values_ev.append(total_energy_ev)
        return energy_values_ev

    def get_band_gaps(self):
        """
        Returns a list of band gap values in eV
        of the entries contained in the dataset
        """
        values = []
        for entry in self.download_data():
            archive = entry["archive"]
            calculations = archive["run"][0]["calculation"]

            def first_value(gaps):
                if isinstance(gaps, dict):
                    gaps = [gaps]
                for gap in gaps or []:
                    if gap.get("value") is not None:
                        return gap["value"]
                return None

            band_gap = next(
                (
                    value
                    for value in (first_value(c.get("band_gap")) for c in reversed(calculations))
                    if value is not None
                ),
                None,
            )
            if band_gap is None:
                result_band_gap = archive.get("results", {}).get("properties", {}).get("electronic", {}).get("band_gap")
                band_gap = first_value(result_band_gap)
            if band_gap is None:
                raise ValueError("No band gap result found in the NOMAD entry.")
            values.append(band_gap / electron_volt)
        return values

    def get_structure_objects(self, filename="structure.json"):
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
            for entry_id in list_of_entry_ids:
                request_url = f"{self.base_url}entries/{entry_id}/raw/{filename}"
                response = requests.get(request_url, headers=self.headers)
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
            print(
                "The request was not successful."
                "Check the NOMAD docs for information on the error code:"
            )
            print("https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#")
            print(f"The following error occured: {error}")

    def get_atoms_objects(self):
        """ "
        Returns a list of ase Atoms object for the dataset
        """
        list_of_atoms_objects = []
        # list_of_entry_ids = self.get_entry_ids()
        # for entry_id in (list_of_entry_ids):
        #    entry = SingleEntry(entry_id=entry_id, token=self.token)
        for entry in self.download_data():
            # To get the converged energy result access the last calculation
            nomad_atoms = entry["archive"]["run"][0]["system"][-1]["atoms"]
            positions_meter = nomad_atoms["positions"]
            cell_meter = nomad_atoms["lattice_vectors"]
            atoms = Atoms(
                symbols=nomad_atoms["labels"],
                positions=np.array(positions_meter) * 10**10,  # convert to Angstrom
                cell=np.array(cell_meter) * 10**10,  # convert to Angstrom
                pbc=nomad_atoms["periodic"],
            )
            list_of_atoms_objects.append(atoms)
        return list_of_atoms_objects

    def get_lattice_constants(self):
        """Return raw cell lengths and angles for the dataset's structures.

        Returns six lists in this order: ``a``, ``b``, ``c`` (lengths in
        Angstrom), then ``alpha``, ``beta``, ``gamma`` (angles in degrees).
        The lengths and angles follow the cell-vector order in each atoms
        object. To get lattice parameters normalized to the parent lattice,
        use :meth:`get_normalized_lattice_constants`.
        """
        atoms_objects = self.get_atoms_objects()
        return _collect_lattice_parameters(atoms.cell for atoms in atoms_objects)

    def get_normalized_lattice_constants(self):
        """Return lattice lengths and angles normalized to the parent lattice.

        Returns six lists in this order: ``a``, ``b``, ``c`` (lengths in
        Angstrom), then ``alpha``, ``beta``, ``gamma`` (angles in degrees).
        For each structure, its supercell transformation is inverted and
        applied to the cell from the corresponding atoms object. The resulting
        vectors are ordered by ascending length, with the angles calculated
        from the correspondingly ordered vectors.
        """
        structures = self.get_structure_objects()
        atoms_objects = self.get_atoms_objects()
        normalized_cells = []
        for structure, atoms in zip(structures, atoms_objects):
            transformation = structure.get_supercell().get_transformation()
            normalized_cells.append(np.linalg.inv(transformation) @ np.asarray(atoms.cell))
        return _collect_lattice_parameters(normalized_cells, sort_by_length=True)
