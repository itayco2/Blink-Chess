"""Shared test setup: marker-based skips so one suite runs on the GPU box and on torch-free CI."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _cuda_available() -> bool:
    if not _torch_available():
        return False
    import torch

    return torch.cuda.is_available()


def pytest_collection_modifyitems(config, items):
    skip_torch = pytest.mark.skip(reason="needs the train extra (torch)")
    skip_cuda = pytest.mark.skip(reason="needs a CUDA GPU")
    has_torch = _torch_available()
    has_cuda = _cuda_available()
    for item in items:
        if "torch" in item.keywords and not has_torch:
            item.add_marker(skip_torch)
        if "cuda" in item.keywords and not has_cuda:
            item.add_marker(skip_cuda)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def repo_files(repo_root: Path) -> list[Path]:
    """Every file git tracks or would track (cached plus untracked-but-not-ignored)."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    return [repo_root / line for line in out.splitlines() if line and (repo_root / line).is_file()]
