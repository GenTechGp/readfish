"""The fork's version marker, so patient run logs show which build ran."""

import pytest
from packaging.version import Version

from readfish.__about__ import __version__
from readfish._cli_base import main


def test_version_is_gtg_build():
    version = Version(__version__)
    assert version.public == "2026.1.0"
    assert version.local == "gtg.1"


def test_cli_version_prints_gtg_build(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "readfish 2026.1.0+gtg.1"
