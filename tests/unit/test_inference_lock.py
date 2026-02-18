from __future__ import annotations

from types import SimpleNamespace

from src import inference_lock as lock_mod


def _patch_config_tmp(monkeypatch, tmp_path):
    cfg = SimpleNamespace(paths=SimpleNamespace(tmp_dir=tmp_path))
    monkeypatch.setattr(lock_mod, "get_config", lambda: cfg)


def test_default_heavy_role_uses_heavy_lock(monkeypatch, tmp_path):
    _patch_config_tmp(monkeypatch, tmp_path)
    monkeypatch.delenv("ORCHESTRATOR_INFERENCE_LOCK_FILE", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_INFERENCE_LOCK_EMBEDDER_FILE", raising=False)

    path = lock_mod._lock_path("frontdoor")
    assert path == tmp_path / "heavy_model.lock"


def test_embedder_role_uses_isolated_lock_by_default(monkeypatch, tmp_path):
    _patch_config_tmp(monkeypatch, tmp_path)
    monkeypatch.delenv("ORCHESTRATOR_INFERENCE_LOCK_EMBEDDER_FILE", raising=False)

    path = lock_mod._lock_path("embedder")
    assert path == tmp_path / "embedder_model.lock"


def test_embedder_lock_filename_override(monkeypatch, tmp_path):
    _patch_config_tmp(monkeypatch, tmp_path)
    monkeypatch.setenv("ORCHESTRATOR_INFERENCE_LOCK_EMBEDDER_FILE", "custom_embed.lock")

    path = lock_mod._lock_path("embedder_2")
    assert path == tmp_path / "custom_embed.lock"


def test_acquire_lock_aborts_on_cancel_check(monkeypatch, tmp_path):
    _patch_config_tmp(monkeypatch, tmp_path)
    monkeypatch.setattr(lock_mod.fcntl, "flock", lambda *_args, **_kwargs: (_ for _ in ()).throw(BlockingIOError()))
    monkeypatch.setattr(lock_mod.time, "sleep", lambda _s: None)

    with open(tmp_path / "heavy_model.lock", "a") as fh:
        try:
            lock_mod._acquire_lock_with_timeout(
                fh.fileno(),
                lock_mod.fcntl.LOCK_EX,
                role="architect_coding",
                mode="exclusive",
                lock_file=tmp_path / "heavy_model.lock",
                timeout_s=180.0,
                poll_s=0.01,
                log_every_s=999.0,
                cancel_check=lambda: True,
                deadline_s=None,
            )
            assert False, "expected TimeoutError"
        except TimeoutError as e:
            assert "cancelled" in str(e)


def test_acquire_lock_aborts_on_request_deadline(monkeypatch, tmp_path):
    _patch_config_tmp(monkeypatch, tmp_path)
    monkeypatch.setattr(lock_mod.fcntl, "flock", lambda *_args, **_kwargs: (_ for _ in ()).throw(BlockingIOError()))
    monkeypatch.setattr(lock_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(lock_mod.time, "perf_counter", lambda: 100.0)

    with open(tmp_path / "heavy_model.lock", "a") as fh:
        try:
            lock_mod._acquire_lock_with_timeout(
                fh.fileno(),
                lock_mod.fcntl.LOCK_EX,
                role="architect_coding",
                mode="exclusive",
                lock_file=tmp_path / "heavy_model.lock",
                timeout_s=180.0,
                poll_s=0.01,
                log_every_s=999.0,
                cancel_check=None,
                deadline_s=99.0,
            )
            assert False, "expected TimeoutError"
        except TimeoutError as e:
            assert "deadline exceeded" in str(e)
