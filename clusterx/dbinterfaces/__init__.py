""""
This module is designed to extract data from NOMAD or
a NOMAD OASIS database (such as total values, structures, etc.)
and use this data to generate structure sets along with their
associated properties in CELL, which can then be used to create models.

**Requirements:**
    - Installation of CELL
    Only if you want to make use of get_nomad_oasis_access_token:
    - Installation of dotenv ($pip install python-dotenv)
    Alternatively you can create the token using
    'https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#/auth/get_token_via_query_auth_token_get'
"""
