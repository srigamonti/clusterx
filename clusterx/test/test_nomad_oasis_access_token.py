from unittest.mock import MagicMock, patch

import pytest

from clusterx.cli.get_nomad_oasis_access_token import (
    get_nomad_oasis_access_token as cli_get_token,
)
from clusterx.dbinterfaces.get_nomad_oasis_access_token import (
    get_nomad_oasis_access_token,
)


def test_get_token_explicit_credentials():
    mock_response = MagicMock()
    mock_response.json.return_value = {"access_token": "fake_token_123"}

    with patch("requests.post", return_value=mock_response) as mock_post:
        token = get_nomad_oasis_access_token(
            base_url="https://example-oasis.org/api/v1/",
            username="testuser",
            userpassword="testpassword",
        )
        assert token == "fake_token_123"
        mock_post.assert_called_once_with(
            "https://example-oasis.org/api/v1/auth/token",
            data={
                "username": "testuser",
                "password": "testpassword",
                "grant_type": "password",
            },
        )


def test_get_token_prompt():
    mock_response = MagicMock()
    mock_response.json.return_value = {"access_token": "fake_prompted_token"}

    with (
        patch("requests.post", return_value=mock_response),
        patch("builtins.input", return_value="prompted_user"),
        patch("getpass.getpass", return_value="prompted_pass"),
    ):
        token = get_nomad_oasis_access_token(
            base_url="https://example-oasis.org/api/v1",
            prompt=True,
        )
        assert token == "fake_prompted_token"


def test_get_token_missing_credentials_raises():
    with pytest.raises(ValueError, match="Username and password must be provided"):
        get_nomad_oasis_access_token(
            base_url="https://example-oasis.org/api/v1",
            prompt=False,
        )


def test_cli_get_token_with_output_file(tmp_path):
    mock_response = MagicMock()
    mock_response.json.return_value = {"access_token": "cli_secret_token"}
    output_file = tmp_path / "token.txt"

    with (
        patch("requests.post", return_value=mock_response),
        patch("builtins.input", return_value="my_cli_user"),
        patch("getpass.getpass", return_value="my_cli_pass"),
    ):
        token = cli_get_token(
            base_url="https://example-oasis.org/api/v1",
            output_file=str(output_file),
        )
        assert token == "cli_secret_token"
        assert output_file.exists()
        assert output_file.read_text().strip() == "cli_secret_token"
