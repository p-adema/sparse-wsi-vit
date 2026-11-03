"""Curriculum resumes learned weights and rejects incompatible parent runs."""

import argparse
import json

import pytest
import torch

from scripts.probe_mondriaan_controls import build_model, train_one
from scripts.probe_mondriaan_curriculum import validate_parent
from scripts.probe_mondriaan_vit5 import evaluate, make_data


@pytest.mark.parametrize("model_name", ["vit5", "abmil"])
def test_curriculum_starts_from_saved_logits_and_cases(tmp_path, model_name):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model = build_model(model_name)
    data = make_data(8, 42, 1, False, tokenization="object")
    expected = evaluate(model, data, 8)
    checkpoint = tmp_path / "parent.pt"
    torch.save({"settings": {"steps": 2000}, "model_state": model.state_dict()}, checkpoint)
    args = argparse.Namespace(train_size=8, eval_size=8, batch_size=8, steps=1,
                              lr=2e-4, threads=1, output_dir=tmp_path, eval_every=1)
    report = train_one(model_name, "grid", 7, data, {"standard": data}, args,
                       initial_checkpoint=checkpoint, sampler_seed=4007)
    assert report["history"][0]["evaluation"] == expected
    assert report["optimizer_restarted"]
    assert report["settings"]["sampler_seed"] == 4007
    assert report["parent_settings"] == {"steps": 2000}


def test_parent_validation_rejects_incompatible_parent_settings(tmp_path):
    path = tmp_path / "vit5_grid_seed42.pt"
    path.touch()
    report = {"settings": {"model": "vit5", "seed": 42, "train_size": 4096,
                           "eval_size": 256, "random_colors": False, "vocabulary_size": 24,
                           "object_placement": "aligned", "steps": 2000,
                           "control": "pair_only"},
              "tokens_per_sample": 64, "history": [{"step": 2000}]}
    path.with_suffix(".json").write_text(json.dumps(report))
    assert validate_parent(path, "vit5", 42, 4096, 1024, "pair_to_full") == report
    with pytest.raises(ValueError, match="eval_size"):
        validate_parent(path, "vit5", 42, 4096, 1024, "full_to_full")
    with pytest.raises(ValueError, match="seed"):
        validate_parent(path, "vit5", 7, 4096, 1024, "pair_to_full")
