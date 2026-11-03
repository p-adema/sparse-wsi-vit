"""The entire approved batch must finish validation selection before any test."""

import copy
import json

import pytest
import torch

from scripts import probe_mondriaan_attention_batch as batch


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(batch.benchmark, "build_model", lambda *args: torch.nn.Linear(1, 2))
    root = tmp_path / "batch"
    protocol = batch.freeze(root)
    return root, protocol


def training_artifacts(protocol, seed, name, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = batch.model_settings(batch.individual_protocol(protocol, seed), name, seed)
    steps = [0, *range(protocol["eval_every"], protocol["steps"] + 1, protocol["eval_every"])]
    history = [{"step": step, "validation": {"worst_case_accuracy": i / 20,
                "binary_macro_f1": 0.7, "accuracy": 0.8}}
               for i, step in enumerate(steps)]
    report = {"settings": settings, "parameters": settings["parameter_count"],
              "history": history, "selected_step": steps[-1],
              "selected_rank": list(batch.benchmark.checkpoint_rank(history[-1]["validation"])),
              "seconds": 1.0}
    state = {"model_state": {"weight": torch.ones(1)},
             "optimizer_state": {"state": {0: {"step": torch.tensor(float(protocol["steps"]))}}},
             "sampler_state": torch.Generator().get_state(), "torch_rng_state": torch.get_rng_state(),
             "step": protocol["steps"], "settings": settings}
    torch.save(state, run_dir / "best.pt")
    torch.save(state, run_dir / "final.pt")
    batch.write_json(run_dir / "training.json", report)
    return report


def complete_seed(protocol, seed):
    directory = batch.seed_root(protocol["output_dir"], seed)
    hashes = {}
    for name in protocol["models"]:
        run_dir = directory / f"{name}_seed{seed}"
        training_artifacts(protocol, seed, name, run_dir)
        hashes[name] = {filename: batch.file_hash(run_dir / filename)
                        for filename in ("best.pt", "final.pt", "training.json")}
    batch.write_json(directory / "fit_complete.json", {"batch_id": protocol["batch_id"],
        "seed": seed, "steps": protocol["steps"], "models": protocol["models"],
        "artifact_sha256": hashes})


def test_freeze_has_no_data_generation_and_individual_protocols_match(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("Freeze must not prepare data or construct test images")
    monkeypatch.setattr(batch.benchmark, "prepare_task", forbidden)
    root = tmp_path / "batch"
    protocol = batch.freeze(root)
    assert protocol["parameter_counts"] == {"vit5": 2_381_954, "transmil": 2_172_434,
                                            "mean": 2_373_858}
    assert "scripts/probe_mondriaan_attention_batch.py" in protocol["source_sha256"]
    assert protocol["library_versions"]["torch"]
    for seed in (42, 7, 2026):
        saved = json.loads((root / f"seed{seed}" / "protocol.json").read_text())
        assert saved == batch.individual_protocol(protocol, seed)
        assert saved["seeds"] == [seed]
    assert not list(root.rglob("*test*"))
    assert batch.load_batch(root) == protocol


def test_frozen_protocol_cannot_be_silently_overwritten(frozen):
    root, _ = frozen
    with pytest.raises(ValueError, match="overwrite"):
        batch.freeze(root)


@pytest.mark.parametrize("finished", [0, 1, 2])
def test_evaluation_rejects_partial_batch_before_preparing_any_data(frozen, monkeypatch, finished):
    root, protocol = frozen
    for seed in protocol["seeds"][:finished]:
        complete_seed(protocol, seed)
    called = []
    monkeypatch.setattr(batch.benchmark, "prepare_task", lambda *args: called.append(args))
    with pytest.raises(ValueError, match="All nine fits"):
        batch.evaluate_batch(root)
    assert called == []
    assert not list(root.rglob("*test*"))


def test_all_nine_fits_are_verified_before_the_first_data_preparation(frozen, monkeypatch):
    root, protocol = frozen
    for seed in protocol["seeds"]:
        complete_seed(protocol, seed)
    calls = []

    def stop_at_first_preparation(args, seed):
        calls.append(seed)
        assert len(batch.preflight(root)[1]) == 9
        raise RuntimeError("verified barrier reached before test creation")

    monkeypatch.setattr(batch.benchmark, "prepare_task", stop_at_first_preparation)
    with pytest.raises(RuntimeError, match="verified barrier"):
        batch.evaluate_batch(root)
    assert calls == [42]
    assert not list(root.rglob("*test*"))


def test_fit_never_calls_final_test_closure_and_copies_frozen_settings(frozen, monkeypatch):
    root, protocol = frozen
    prepared = []

    def prepare(args, seed):
        prepared.append(seed)
        def forbidden_test():
            pytest.fail("fit must not construct final test images")
        return {}, {"split": "train"}, {"split": "validation"}, forbidden_test

    def train(model, training, validation, *, seed, output_dir, settings, **kwargs):
        assert training["split"] == "train" and validation["split"] == "validation"
        assert settings == batch.model_settings(batch.individual_protocol(protocol, seed),
                                                settings["model"], seed)
        return training_artifacts(protocol, seed, settings["model"], output_dir)

    monkeypatch.setattr(batch.benchmark, "prepare_task", prepare)
    monkeypatch.setattr(batch.benchmark, "train", train)
    runs = batch.fit(root, 42)
    assert len(runs) == 3
    assert prepared == [42]
    assert not list(root.rglob("*test*"))
    assert (root / "seed42/fit_complete.json").is_file()
    # Repeated fit validates saved completion rather than overwriting it.
    assert len(batch.fit(root, 42)) == 3
    assert prepared == [42]


def test_one_changed_or_incomplete_checkpoint_blocks_the_global_barrier(frozen):
    root, protocol = frozen
    for seed in protocol["seeds"]:
        complete_seed(protocol, seed)
    final_path = root / "seed2026/transmil_seed2026/final.pt"
    state = torch.load(final_path, weights_only=True)
    state["step"] = 3999
    torch.save(state, final_path)
    marker_path = root / "seed2026/fit_complete.json"
    marker = json.loads(marker_path.read_text())
    marker["artifact_sha256"]["transmil"]["final.pt"] = batch.file_hash(final_path)
    batch.write_json(marker_path, marker)
    with pytest.raises(ValueError, match="Incomplete training checkpoint"):
        batch.preflight(root)


def test_selection_must_follow_the_frozen_validation_rank(frozen):
    root, protocol = frozen
    for seed in protocol["seeds"]:
        complete_seed(protocol, seed)
    report_path = root / "seed7/mean_seed7/training.json"
    report = json.loads(report_path.read_text())
    report["selected_step"] = 0
    batch.write_json(report_path, report)
    marker_path = root / "seed7/fit_complete.json"
    marker = json.loads(marker_path.read_text())
    marker["artifact_sha256"]["mean"]["training.json"] = batch.file_hash(report_path)
    batch.write_json(marker_path, marker)
    with pytest.raises(ValueError, match="Invalid validation selection"):
        batch.preflight(root)


def test_source_changes_and_individual_protocol_changes_are_rejected(frozen, monkeypatch):
    root, protocol = frozen
    changed = copy.deepcopy(protocol["source_sha256"])
    changed["scripts/probe_mondriaan_attention_batch.py"] = "changed"
    original = batch.source_hashes
    monkeypatch.setattr(batch, "source_hashes", lambda: changed)
    with pytest.raises(ValueError, match="source changed"):
        batch.load_batch(root)
    monkeypatch.setattr(batch, "source_hashes", original)
    path = root / "seed7/protocol.json"
    saved = json.loads(path.read_text())
    saved["test_seed_offset"] += 1
    batch.write_json(path, saved)
    with pytest.raises(ValueError, match="Individual protocol"):
        batch.load_batch(root)
