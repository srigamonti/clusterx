""" "
This module is designed to obtain the access token for a NOMAD OASIS database.
"""

import getpass
import os

import requests
from dotenv import load_dotenv


def get_nomad_oasis_access_token(
    base_url: str = "https://nomad-lab.eu/prod/v1/staging/api/v1/",
    path_to_dotenv: str | None = None,
    username: str | None = None,
    userpassword: str | None = None,
    prompt: bool = False,
):
    """This creates an access token to access data stored in NOMAD.
    The token is needed if you use a NOMAD OASIS that requires your login information
    or if you want to access private data on NOMAD.
    It requires having a NOMAD (NOMAD OASIS) account with permission
    to use the corresponding NOMAD (OASIS) data.

    Parameters
    ----------
    base_url : str
        Specifies the URL of NOMAD.
        Defaults to the general NOMAD URL.
        In case, you use a NOMAD OASIS specify its URL here instead.
        Members of the HU Berlin sol group should use:
        https://sol-oasis.physik.hu-berlin.de/nomad-oasis/api/v1/

    path_to_dotenv : str
        It is advised to the user to create a .env file to access the username
        and password of your NOMAD (OASIS) account as environment variables,
        instead of putting them in the code visibly.
        The .env file should contain two lines::

            OASIS_USER_NAME="your_username"
            OASIS_PASSWORD="your_password"

        Give the path to the .env file as parameter.
        If such a file does not exist or is structured differently
        use the parameters username and userpassword instead.

    username : str
        The username of your NOMAD or NOMAD OASIS account.
        It is advised to use environment variables
        instead of writing the username in the code visibly.

    userpassword : str
        The password of your NOMAD or NOMAD OASIS account.
        It is strongly advised to use environment variables
        instead of writing the password in the code visibly.

    prompt : bool
        If True, prompts securely for username and/or password on the command line
        if they are not provided via arguments or environment variables.
    """
    myname = username
    mypassword = userpassword

    if path_to_dotenv is not None:
        load_dotenv(path_to_dotenv)
        myname = os.getenv("OASIS_USER_NAME", myname)
        mypassword = os.getenv("OASIS_PASSWORD", mypassword)

    if prompt:
        if not myname:
            myname = input("NOMAD OASIS Username: ")
        if not mypassword:
            mypassword = getpass.getpass("NOMAD OASIS Password: ")

    if not myname or not mypassword:
        raise ValueError("Username and password must be provided.")

    url = f"{base_url.rstrip('/')}/auth/token"
    response = requests.post(
        url,
        data={
            "username": myname,
            "password": mypassword,
            "grant_type": "password",
        },
    )
    response.raise_for_status()
    token = response.json()["access_token"]

    return token
