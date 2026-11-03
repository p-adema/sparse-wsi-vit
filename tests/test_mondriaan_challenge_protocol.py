"""The matched challenge must never become a checkpoint-selection dataset."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import probe_mondriaan_matched_challenge as challenge
from sparse_wsi_vit.datasets.mondriaan_dataset.controlled import (
    make_appearance_pools, make_controlled_dataset,
)
from sparse_wsi_vit.datasets.mondriaan_dataset.matched import (
    build_matched_subcase_challenge, fit_matching_policy,
)


@pytest.fixture
def config(tmp_path, monkeypatch):
    batch = tmp_path / "batch_protocol.json"
    batch.write_text("{}\n")
    settings = {"seeds": [7, 42], "models": ["vit5", "transmil", "mean"],
            "run_dirs": [str(tmp_path / f"seed{seed}") for seed in (7, 42)],
            "batch_protocol": str(batch), "variant": "pixels16", "appearance_pool_size": 32,
            "appearance_holdout": True, "min_distance": 0, "palette": "shared",
            "placement": "aligned", "representation": "scene", "object_counts": [4],
            "train_size": 64, "validation_size": 32, "test_size": 32,
            "appearance_seed_offset": 103000, "test_seed_offset": 366000,
            "steps": 4000, "batch_size": 32, "lr": 2e-4, "transmil_landmarks": 32,
            "bins": 4, "challenge_size": 16, "challenge_seed_offset": 266000,
            "proposal_pairs": 128, "max_rounds": 40, "threads": 1,
            "output_dir": str(tmp_path)}
    batch.write_text(json.dumps(settings))
    # The production loader additionally verifies frozen primary source/library
    # hashes; its own tests cover that. Keep these small-data challenge tests local.
    monkeypatch.setattr(challenge, "load_primary_batch", lambda path: json.loads(Path(path).read_text()))
    monkeypatch.setattr(challenge, "preflight_primary", lambda path: (settings, []))
    return settings


def populate_selections(config):
    for seed, directory in zip(config["seeds"], config["run_dirs"], strict=True):
        root = Path(directory)
        root.mkdir()
        (root / "protocol.json").write_text(json.dumps({**config, "seeds": [seed]}))
        for model in config["models"]:
            run = root / f"{model}_seed{seed}"
            run.mkdir()
            report = {"history": [{"step": config["steps"]}], "selected_step": 2000,
                      "selected_rank": [.5, .5, .5], "parameters": 10}
            (run / "training.json").write_text(json.dumps(report))
            state = {"settings": {"model": model, "seed": seed}, "step": 2000,
                     "model_state": {"weights": torch.ones(1)}}
            torch.save(state, run / "best.pt")
            torch.save({**state, "step": config["steps"]}, run / "final.pt")


def test_freeze_never_calls_final_generator_and_refuses_refit(tmp_path, config, monkeypatch):
    original = challenge.make_local_splits
    def without_test(args, seed):
        pools, training, validation, _ = original(args, seed)
        def forbidden():
            pytest.fail("Freeze generated final-test images")
        return pools, training, validation, forbidden
    monkeypatch.setattr(challenge, "make_local_splits", without_test)
    output = tmp_path / "challenge"
    _, sources = challenge.freeze(output, config)
    assert set(sources) == {"7", "42"}
    assert set(sources["7"]["controls"]) == {
        f"{population}/{statistic}_{kind}"
        for population in ("full_training", "ambiguous_training")
        for statistic in challenge.STATISTICS for kind in ("threshold", "interval")
    } | {f"{population}/frozen_hamming_bin"
         for population in ("full_training", "ambiguous_training")}
    stored = json.loads((output / "training_policies.json").read_text())
    restored = challenge.restore_policy(stored["7"]["policy"])
    assert restored.as_dict() == sources["7"]["policy"]
    assert not list(output.glob("*.npz"))
    with pytest.raises(ValueError, match="already frozen"):
        challenge.freeze(output, config)


def test_freeze_requires_prior_primary_protocol_and_no_started_fit(tmp_path, config):
    Path(config["batch_protocol"]).unlink()
    with pytest.raises(ValueError, match="batch protocol"):
        challenge.freeze(tmp_path / "challenge", config)
    Path(config["batch_protocol"]).write_text("{}")
    Path(config["batch_protocol"]).write_text(json.dumps(config))
    run = Path(config["run_dirs"][0]) / "vit5_seed7"
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"already started")
    with pytest.raises(ValueError, match="before learned fitting"):
        challenge.freeze(tmp_path / "challenge", config)


def test_freeze_rejects_source_mismatch_before_writing(tmp_path, config):
    changed = {**config, "appearance_holdout": False}
    with pytest.raises(ValueError, match="appearance_holdout"):
        challenge.freeze(tmp_path / "challenge", changed)
    assert not (tmp_path / "challenge" / "protocol.json").exists()


def test_gate_requires_every_model_and_seed(tmp_path, config, monkeypatch):
    output = tmp_path / "challenge"
    challenge.freeze(output, config)
    populate_selections(config)
    selected = challenge.selection_gate(config)
    assert len(selected) == 6
    assert all(len(item["checkpoint_sha256"]) == 64 for item in selected.values())
    missing = Path(config["run_dirs"][-1]) / "mean_seed42" / "final.pt"
    missing.unlink()
    def forbidden(*args):
        pytest.fail("Challenge generation before the final model selection")
    monkeypatch.setattr(challenge, "make_challenge", forbidden)
    with pytest.raises(ValueError, match="not ready"):
        challenge.evaluate_frozen(output, tmp_path / "reports")
    assert not (output / "selection_manifest.json").exists()
    assert not list(output.glob("*.npz"))


def test_gate_rejects_source_or_checkpoint_budget_changes(config):
    populate_selections(config)
    primary = Path(config["run_dirs"][0]) / "protocol.json"
    source = json.loads(primary.read_text())
    source["appearance_seed_offset"] += 1
    primary.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="appearance_seed_offset"):
        challenge.selection_gate(config)
    source["appearance_seed_offset"] -= 1
    primary.write_text(json.dumps(source))
    checkpoint = primary.parent / "transmil_seed7" / "final.pt"
    state = torch.load(checkpoint, weights_only=True)
    state["step"] = 3999
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="budget mismatch"):
        challenge.selection_gate(config)


def test_completed_fits_still_require_primary_final_manifests(tmp_path, config, monkeypatch):
    output = tmp_path / "challenge"
    challenge.freeze(output, config)
    populate_selections(config)
    def forbidden(*args):
        pytest.fail("Challenge generation before primary final manifests")
    monkeypatch.setattr(challenge, "make_challenge", forbidden)
    with pytest.raises(ValueError, match="manifest not ready"):
        challenge.evaluate_frozen(output, tmp_path / "reports")
    assert not (output / "selection_manifest.json").exists()


def test_bin_lookup_and_rejection_accounting():
    pools = make_appearance_pools(sizes={"train": 64, "test": 64}, seed=7)
    training = make_controlled_dataset(pools["train"], 256, split="train", palette_mode="shared")
    policy = fit_matching_policy(training, bins=4)
    data = build_matched_subcase_challenge(pools["test"], policy, 32, seed=99,
                                          palette_mode="shared", proposal_pairs=128, max_rounds=40)
    controls = challenge.fit_controls(training, policy)
    for name, control in controls.items():
        predictions = challenge.scalar_predictions(control, data)
        metrics = challenge.classification_metrics(data["y"], predictions, data["cases"],
                                                   intended_cases=challenge.INTENDED_CASES)
        assert metrics["selection_eligible"]
        assert metrics["missing_cases"] == []
        if name.endswith("frozen_hamming_bin"):
            assert metrics["accuracy"] == .5
    audit = data["matching_audit"]
    eligible = sum(audit["candidate_pairs_by_stratum"].values())
    rejected = sum(sum(audit[key].values()) for key in (
        "rejected_bin_mismatch_by_stratum", "rejected_duplicate_images_by_stratum",
        "skipped_completed_quota_by_stratum"))
    assert eligible == audit["retained_pairs"] + rejected
    assert audit["proposed_pairs"] == eligible + audit["nonrequested_case_pairs"]
    assert np.unique(data["pair_ids"]).size == 16


def test_challenge_checks_original_source_fingerprints(tmp_path, config):
    _, sources = challenge.freeze(tmp_path / "challenge", config)
    sources["7"]["training_image_hashes"][0] = "changed"
    with pytest.raises(ValueError, match="pixels changed"):
        challenge.make_challenge(config, 7, sources["7"])
