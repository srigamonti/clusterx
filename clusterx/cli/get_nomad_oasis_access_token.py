# Copyright (c) 2015-2025, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import plac

from clusterx.cli.config_utils import cmd_message
from clusterx.dbinterfaces.get_nomad_oasis_access_token import (
    get_nomad_oasis_access_token as _get_nomad_oasis_access_token,
)

commands = ["get_nomad_oasis_access_token"]


@plac.opt(
    "base_url",
    abbrev="u",
    help="Base URL of the NOMAD / NOMAD OASIS API.",
)
@plac.opt(
    "path_to_dotenv",
    abbrev="e",
    help="Path to .env file containing OASIS_USER_NAME and OASIS_PASSWORD.",
)
@plac.opt(
    "username",
    abbrev="n",
    help="Username for the NOMAD / NOMAD OASIS account.",
)
@plac.opt(
    "userpassword",
    abbrev="p",
    help="Password for the NOMAD / NOMAD OASIS account (if not prompted).",
)
@plac.opt(
    "output_file",
    abbrev="o",
    help="Optional file path to save the access token.",
)
def get_nomad_oasis_access_token(
    base_url="https://nomad-lab.eu/prod/v1/staging/api/v1/",
    path_to_dotenv=None,
    username=None,
    userpassword=None,
    output_file=None,
):
    """Obtain an access token for NOMAD or NOMAD OASIS.

    If credentials are not provided via arguments or a .env file, you will be
    prompted interactively and securely for your username and password.

    Parameters
    ----------
    base_url : str, optional
        Base URL of the NOMAD / NOMAD OASIS API.
        Default is ``"https://nomad-lab.eu/prod/v1/staging/api/v1/"``.

    path_to_dotenv : str, optional
        Path to .env file containing OASIS_USER_NAME and OASIS_PASSWORD.

    username : str, optional
        Username for the NOMAD / NOMAD OASIS account.

    userpassword : str, optional
        Password for the NOMAD / NOMAD OASIS account (if not prompted).

    output_file : str, optional
        Optional file path to save the access token.

    Examples
    --------
    Prompt for username and password interactively:

    .. code-block:: bash

        $ cell get_nomad_oasis_access_token -u https://sol-oasis.physik.hu-berlin.de/nomad-oasis/api/v1/

    Prompt and save token directly to a file:

    .. code-block:: bash

        $ cell get_nomad_oasis_access_token \\
            -u https://sol-oasis.physik.hu-berlin.de/nomad-oasis/api/v1/ \\
            -o nomad_token.txt
    """
    cmd_message("head")

    token = _get_nomad_oasis_access_token(
        base_url=base_url,
        path_to_dotenv=path_to_dotenv,
        username=username,
        userpassword=userpassword,
        prompt=True,
    )

    print(f"\nAccess token:\n{token}")

    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(token.strip() + "\n")
        print(f"Token saved to {output_file}")

    return token
