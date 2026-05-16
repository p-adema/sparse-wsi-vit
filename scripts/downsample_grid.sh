#!/usr/bin/env bash
set -e
# todo: scheduler patience
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=2 dataset.features_name=cls_224x224 dataset.label_col_name=is_tumor \
  train.iterations=8000 scheduler.total_iterations=8000 dataset.pin_memory=False

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=4 dataset.features_name=cls_224x224 dataset.label_col_name=is_tumor \
  train.iterations=8000 scheduler.total_iterations=8000 dataset.pin_memory=False

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=2 dataset.features_name=patches_224x224 dataset.label_col_name=is_tumor \
  train.iterations=8000 scheduler.total_iterations=8000 dataset.pin_memory=False

nq sh -c 'echo "Downsample grid halfway, reset RAM" | mail -s "Reset RAM" peter.adema@student.uva.nl'

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=8 dataset.features_name=cls_224x224 dataset.label_col_name=is_tumor \
  train.iterations=8000 scheduler.total_iterations=8000 dataset.pin_memory=False

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=4 dataset.features_name=patches_224x224 dataset.label_col_name=is_tumor \
  train.iterations=8000 scheduler.total_iterations=8000 dataset.pin_memory=False

sleep 0.1
nq uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py \
  dataset.downscale_block=8 dataset.features_name=patches_224x224 dataset.label_col_name=is_tumor \
  train.iterations=8000 scheduler.total_iterations=8000 dataset.pin_memory=False

nq sh -c 'echo "Downsample grid complete" | mail -s "Jobs done" peter.adema@student.uva.nl'