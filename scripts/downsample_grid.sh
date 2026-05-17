#!/usr/bin/env bash
set -e

nq /home/peter/reset-ram

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_tcga_tmb.py \
  train.iterations=5000 scheduler.total_iterations=5000 dataset.pin_memory=False \
  net.checkpoint_activations=False scheduler.patience=10 dataset.features_name=cls_224x224

sleep 0.1
nq /home/peter/reset-ram

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=2 dataset.features_name=cls_224x224 dataset.label_col_name=is_tumor \
  train.iterations=5000 scheduler.total_iterations=5000 dataset.pin_memory=False scheduler.patience=30

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=4 dataset.features_name=cls_224x224 dataset.label_col_name=is_tumor \
  train.iterations=5000 scheduler.total_iterations=5000 dataset.pin_memory=False scheduler.patience=30

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=8 dataset.features_name=cls_224x224 dataset.label_col_name=is_tumor \
  train.iterations=5000 scheduler.total_iterations=5000 dataset.pin_memory=False scheduler.patience=30

sleep 0.1
nq /home/peter/reset-ram

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=2 dataset.features_name=patches_224x224 dataset.label_col_name=is_tumor \
  train.iterations=5000 scheduler.total_iterations=5000 dataset.pin_memory=False scheduler.patience=30

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=4 dataset.features_name=patches_224x224 dataset.label_col_name=is_tumor \
  train.iterations=5000 scheduler.total_iterations=5000 dataset.pin_memory=False scheduler.patience=30

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=8 dataset.features_name=patches_224x224 dataset.label_col_name=is_tumor \
  train.iterations=5000 scheduler.total_iterations=5000 dataset.pin_memory=False scheduler.patience=30

nq sh -c 'echo "Downsample grid complete" | mail -s "Jobs done" peter.adema@student.uva.nl'