# Sparse-wsi-vit

This project makes use of [uv](https://docs.astral.sh/uv/getting-started/installation/).

To run the tests, run

```bash
uv run pytest
```

To run the experiments, run

```bash
uv run experiments/run.py --config src/sparse_wsi_vit/configs/baseline_vit5dense_preembed.py
```

Note that the current version requires the zipfile sent by David to be unpacked as `amc-data` in the parent directory,
i.e.

```
parent/
    sparse-wsi-vit/
        this README.md
        ...
        
    amc-data/
        combined_tcga_amc.csv
        RD-GC346-02.h5
        ...
```

Custom embeddings for CHAMELYON16 and TCGA are on their way...