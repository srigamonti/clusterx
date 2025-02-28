""""
This module is designed to obtain the access token for a NOMAD OASIS database.
"""
import os
import requests
from dotenv import load_dotenv


def get_nomad_oasis_access_token(path_to_dotenv=None, username=None, userpassword=None):
    """
    This creates an access token to access data stored in the OASIS.
    It requires having a NOMAD (NOMAD OASIS) account with permission
    to use the corresponding NOMAD OASIS.

    Instead of using this function you can also create an access token using:
    'https://nomad-lab.eu/prod/v1/api/v1/extensions/docs#/auth/get_token_via_query_auth_token_get'

    **Parameters**
    ``path_to_dotenv``: *string*
        It is advised to the user to create a .env file to access the username
        and password to the OASIS as environment variables
        instead of putting them in the code visibly.
        The .env file should contain two lines:
            OASIS_USER_NAME="your_username"
            OASIS_PASSWORD="your_password"
        Give the path to the .env file as parameter.
        If such a file does not exist or is structured differently
        use the parameters username and userpassword instead.

    ``username``: *string*
        The username to the OASIS.
        It is advised to use environment variables
        instead of writing the username in the code visibly.

    ``userpassword``: *string*
        The username to the OASIS.
        It is strongly advised to use environment variables
        instead of writing the password in the code visibly.
    """
    if path_to_dotenv is not None:
        load_dotenv(path_to_dotenv)
        myname = os.getenv('OASIS_USER_NAME')
        mypassword = os.getenv('OASIS_PASSWORD')
    else:
        myname = username
        mypassword = userpassword
    
    response_to_authentification = requests.get(
            'https://nomad-lab.eu/prod/v1/staging/api/v1/auth/token',
            params={"username": myname, "password": mypassword}
            )
    token = response_to_authentification.json()['access_token']
    return token
