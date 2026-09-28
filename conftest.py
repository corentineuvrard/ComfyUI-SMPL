"""Pytest configuration for the repository root.

The repository root is a ComfyUI custom-node package: its __init__.py uses
relative imports, but the folder name is not a Python identifier, so pytest
cannot import it as a test package. A small global plugin makes pytest collect
the root as a plain directory instead.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent


class _CollectRootAsDirectory:
    @pytest.hookimpl(tryfirst=True)
    def pytest_collect_directory(self, path, parent):
        if Path(path).resolve() == ROOT:
            return pytest.Dir.from_parent(parent, path=Path(path))
        return None


def pytest_configure(config):
    config.pluginmanager.register(_CollectRootAsDirectory(), "smpl-avatar-root-as-directory")
