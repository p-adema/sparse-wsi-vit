#!/usr/bin/env bash
set -e

nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=2 dataset.features_name=cls_224x224 dataset.label_col_name=is_tumor \
  net.rope_dynamic_high=448 wandb.name=binary_cls_448 \
  train.iterations=8000 scheduler.total_iterations=8000

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=2 dataset.features_name=patches_224x224 dataset.label_col_name=is_tumor \
  net.rope_dynamic_high=448 wandb.name=binary_patches_448 \
  train.iterations=8000 scheduler.total_iterations=8000

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=4 dataset.features_name=cls_224x224 dataset.label_col_name=is_tumor \
  net.rope_dynamic_high=896 wandb.name=binary_cls_896 \
  train.iterations=8000 scheduler.total_iterations=8000

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=4 dataset.features_name=patches_224x224 dataset.label_col_name=is_tumor \
  net.rope_dynamic_high=896 wandb.name=binary_patches_896\
  train.iterations=8000 scheduler.total_iterations=8000

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=8 dataset.features_name=cls_224x224 dataset.label_col_name=is_tumor \
  net.rope_dynamic_high=1792 wandb.name=binary_cls_1792 \
  train.iterations=8000 scheduler.total_iterations=8000

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=8 dataset.features_name=patches_224x224 dataset.label_col_name=is_tumor \
  net.rope_dynamic_high=1792 wandb.name=binary_patches_1792 \
  train.iterations=8000 scheduler.total_iterations=8000
