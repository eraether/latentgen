"""Checkpoint correctness of the model manager (CPU, tiny models)."""

import torch

from latentgen.config import OptimizerConfig
from latentgen.training.manager import ModelManager


def _tiny_model():
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))


def _train_some_steps(manager: ModelManager, steps: int = 5) -> None:
    for _ in range(steps):
        loss = manager(torch.randn(16, 4)).pow(2).mean()
        loss.backward()
        ok, _ = manager.clip_and_step(1.0)
        assert ok


def test_schedule_free_checkpoint_holds_the_averaged_weights():
    """Checkpoints must hold the weights you evaluate with (x), not the ones the optimizer steps on (y)."""
    manager = ModelManager(
        "m",
        _tiny_model(),
        OptimizerConfig(type="adamw_schedulefree", lr=1e-2, warmup_steps=1),
        torch.device("cpu"),
        compile_model=False,
    )
    _train_some_steps(manager)
    saved = manager.state_dict()["model"]
    with manager.eval_weights():
        expected = {k: v.detach().clone() for k, v in manager.model.state_dict().items()}
    live = manager.model.state_dict()  # back in training mode: y, which differs from x after a few steps
    for k in expected:  # allclose: schedule-free swaps x<->y with in-place lerps, which are not bit-exact
        assert torch.allclose(saved[k], expected[k], atol=1e-6), k
    assert any(not torch.equal(live[k], expected[k]) for k in expected), (
        "x and y should differ after training"
    )


def test_resume_round_trip_and_lr_override():
    cfg = OptimizerConfig(type="adamw_schedulefree", lr=1e-2, warmup_steps=1)
    a = ModelManager("m", _tiny_model(), cfg, torch.device("cpu"), compile_model=False)
    _train_some_steps(a)
    state = a.state_dict()

    b = ModelManager(
        "m",
        _tiny_model(),
        OptimizerConfig(type="adamw_schedulefree", lr=5e-3, warmup_steps=1),
        torch.device("cpu"),
        compile_model=False,
    )
    b.load_state_dict(state)
    assert b.step_count == a.step_count
    with a.eval_weights(), b.eval_weights():
        for (k, va), (_, vb) in zip(a.model.state_dict().items(), b.model.state_dict().items(), strict=True):
            assert torch.allclose(va, vb, atol=1e-6), k
    assert b.optimizer.opt.param_groups[0]["lr"] == 5e-3  # the new config's LR wins over the checkpoint's


def test_adamw_with_ema_round_trip():
    cfg = OptimizerConfig(type="adamw", lr=1e-3, warmup_steps=2)
    a = ModelManager(
        "g", _tiny_model(), cfg, torch.device("cpu"), compile_model=False, ema_decay=0.5, ema_every=1
    )
    _train_some_steps(a, steps=3)
    assert a.ema.num_updates == 3
    state = a.state_dict()
    b = ModelManager(
        "g", _tiny_model(), cfg, torch.device("cpu"), compile_model=False, ema_decay=0.5, ema_every=1
    )
    b.load_state_dict(state)
    assert b.ema.num_updates == 3
    for (k, va), (_, vb) in zip(
        a.ema.model.state_dict().items(), b.ema.model.state_dict().items(), strict=True
    ):
        assert torch.equal(va, vb), k
