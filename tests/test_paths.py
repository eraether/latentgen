"""Checkpoint path resolution (`latest`, run names, run dirs, files). No torch needed."""

import importlib.util
from pathlib import Path

import pytest

# load latentgen/training/checkpoint.py on its own: importing the package would pull in torch
_spec = importlib.util.spec_from_file_location(
    "checkpoint", Path(__file__).resolve().parent.parent / "latentgen/training/checkpoint.py"
)
checkpoint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checkpoint)


def _make_run(runs: Path, stage: str, name: str, steps: list[int]) -> Path:
    run = runs / stage / name
    (run / "checkpoints").mkdir(parents=True)
    for s in steps:
        (run / "checkpoints" / f"step_{s:08d}.pt").write_bytes(b"x")
    return run


def test_latest_resolves_under_runs_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no `latest` directory in cwd must confuse the bare keyword
    runs = tmp_path / "runs_x"
    old = _make_run(runs, "vqvae", "run_a", [10, 20])
    new = _make_run(runs, "vqvae", "run_b", [5])
    assert checkpoint.find_checkpoint("latest", runs, "vqvae") == new / "checkpoints" / "step_00000005.pt"
    assert checkpoint.find_checkpoint("run_a", runs, "vqvae") == old / "checkpoints" / "step_00000020.pt"
    assert checkpoint.find_checkpoint(old, runs, "vqvae") == old / "checkpoints" / "step_00000020.pt"
    assert (
        checkpoint.find_checkpoint(runs / "vqvae" / "latest", runs, "vqvae")
        == new / "checkpoints" / "step_00000005.pt"
    )
    file = old / "checkpoints" / "step_00000010.pt"
    assert checkpoint.find_checkpoint(file, runs, "vqvae") == file


def test_missing_checkpoint_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="vqvae checkpoint 'latest' not found"):
        checkpoint.find_checkpoint("latest", tmp_path / "nothing", "vqvae")
    with pytest.raises(FileNotFoundError, match="tried"):
        checkpoint.find_checkpoint("no_such_run", tmp_path, "vqvae")
