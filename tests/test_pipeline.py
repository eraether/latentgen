"""
End-to-end smoke test: every script in order on tiny synthetic data, on CPU, inside a temp dir.
Takes a minute or two. Run with:  pytest tests/test_pipeline.py -v
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def run(script: str, *args: str, cwd: Path) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            f"{script} failed\n--- stdout ---\n{result.stdout[-4000:]}\n--- stderr ---\n{result.stderr[-4000:]}"
        )


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("latentgen_smoke")
    run("make_smoke_data.py", "--out", str(d / "data/smoke/images"), "--count", "32", cwd=d)
    return d


def common(workdir: Path) -> list[str]:
    return [
        "--config",
        str(ROOT / "configs/smoke_test.yaml"),
        "--set",
        "project.allow_cpu=true",
        f"project.runs_dir={workdir / 'runs'}",
        f"data.image_dir={workdir / 'data/smoke/images'}",
        f"data.encoded_file={workdir / 'data/encoded.pt'}",
        f"generate.out_dir={workdir / 'generated'}",
    ]


def test_stage_1a_vqvae_train_and_resume(workdir):
    run("01a_train_vqvae.py", *common(workdir), "vqvae.train.epochs=1", cwd=workdir)
    runs = list((workdir / "runs/vqvae").iterdir())
    assert len(runs) == 1 and any((runs[0] / "checkpoints").glob("step_*.pt"))
    assert any((runs[0] / "images").glob("*.png"))
    # resume for one more epoch into the same run dir (also crosses the cutover step)
    run(
        "01a_train_vqvae.py",
        *common(workdir),
        "vqvae.train.epochs=2",
        "--resume",
        str(workdir / "runs/vqvae/latest"),
        cwd=workdir,
    )
    assert len(list((workdir / "runs/vqvae").iterdir())) == 1


def test_stage_1b_autoencoder(workdir):
    run("01b_train_autoencoder.py", *common(workdir), "autoencoder.train.epochs=1", cwd=workdir)


def test_stage_1c_encode(workdir):
    run("01c_encode_dataset.py", *common(workdir), cwd=workdir)
    import torch

    blob = torch.load(workdir / "data/encoded.pt", weights_only=False)
    assert blob["codes"].shape == (2, 32, 4, 4) and blob["latents"].shape == (2, 32, 4, 4, 4)


def test_stage_2_maskgit(workdir):
    run("02_train_maskgit.py", *common(workdir), "maskgit.train.epochs=1", cwd=workdir)


def test_stage_3_cgan(workdir):
    run("03_train_cgan.py", *common(workdir), "cgan.train.epochs=1", cwd=workdir)


def test_generate(workdir):
    run("04_generate.py", *common(workdir), "generate.num_images=2", "generate.batch_size=2", cwd=workdir)
    assert len(list((workdir / "generated").glob("*.png"))) == 1


def test_audio_pipeline_end_to_end(tmp_path_factory):
    """The 1-D variant: every stage on synthetic wav clips."""
    d = tmp_path_factory.mktemp("latentgen_audio")
    run("make_smoke_data.py", "--audio", "--out", str(d / "data/smoke/audio"), "--count", "32", cwd=d)
    args = [
        "--config",
        str(ROOT / "configs/smoke_test_audio.yaml"),
        "--set",
        "project.allow_cpu=true",
        f"project.runs_dir={d / 'runs'}",
        f"data.image_dir={d / 'data/smoke/audio'}",
        f"data.encoded_file={d / 'data/encoded.pt'}",
        f"generate.out_dir={d / 'generated'}",
    ]
    for script, extra in [
        ("01a_train_vqvae.py", ["vqvae.train.epochs=2"]),
        ("01b_train_autoencoder.py", ["autoencoder.train.epochs=1"]),
        ("01c_encode_dataset.py", []),
        ("02_train_maskgit.py", ["maskgit.train.epochs=1"]),
        ("03_train_cgan.py", ["cgan.train.epochs=1"]),
        ("04_generate.py", ["generate.num_images=2"]),
    ]:
        run(script, *args, *extra, cwd=d)
    assert len(list((d / "generated").glob("*.wav"))) == 4  # 2 samples x (vq, gan)
