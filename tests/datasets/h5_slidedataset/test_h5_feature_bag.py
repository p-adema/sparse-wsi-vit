import pytest
import torch
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sparse_wsi_vit.datasets.h5_slidedataset.h5_dataset import H5FeatureBagDataset

@pytest.fixture
def mock_fast_extraction_data(tmp_path):
    """Creates dummy h5 files representing FAST Virchow2 extractions and a corresponding CSV."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    
    # Create two dummy H5 files for two slides
    slide1_h5 = features_dir / "slide_001.h5"
    slide2_h5 = features_dir / "slide_002.h5"
    
    # slide_001 will have 10 patches, 1280-dim features
    with h5py.File(slide1_h5, "w") as f:
        f.create_dataset("features", data=np.random.rand(10, 1280).astype(np.float32))
        f.create_dataset("coords", data=np.random.randint(0, 10000, size=(10, 2)).astype(np.int32))
        
    # slide_002 will have 5 patches, 1280-dim features
    with h5py.File(slide2_h5, "w") as f:
        f.create_dataset("features", data=np.random.rand(5, 1280).astype(np.float32))
        f.create_dataset("coords", data=np.random.randint(0, 10000, size=(5, 2)).astype(np.int32))
        
    # Create a CSV that points to these slides, plus one missing slide
    csv_path = tmp_path / "slides.csv"
    df = pd.DataFrame({
        "slidename": ["slide_001", "slide_002", "slide_003_missing"],
        "label": [1, 0, 1]
    })
    df.to_csv(csv_path, index=False)
    
    return str(csv_path), str(features_dir)

def test_h5_feature_bag_dataset_initialization(mock_fast_extraction_data):
    csv_path, features_dir = mock_fast_extraction_data
    
    dataset = H5FeatureBagDataset(
        csv_path=csv_path,
        features_dir=features_dir,
        label_col_name="label"
    )
    
    # Should only load the 2 slides that actually exist in the features_dir
    assert len(dataset) == 2
    assert dataset.slides[0]["slide_name"] == "slide_001"
    assert dataset.slides[1]["slide_name"] == "slide_002"

def test_h5_feature_bag_dataset_getitem(mock_fast_extraction_data):
    csv_path, features_dir = mock_fast_extraction_data
    
    dataset = H5FeatureBagDataset(
        csv_path=csv_path,
        features_dir=features_dir,
        label_col_name="label"
    )
    
    # Get first slide
    sample = dataset[0]
    features = sample["input"]
    label = sample["label"]
    slide_name = sample["slide_name"]
    coords = sample["coords"]
    
    assert slide_name == "slide_001"
    assert label == 1
    assert isinstance(features, torch.Tensor)
    assert isinstance(coords, torch.Tensor)
    assert features.shape == (10, 1280)    # 10 patches, 1280-dim
    assert coords.shape == (10, 2)
    assert features.dtype == torch.float32
    assert coords.dtype == torch.float32
    
    # Get second slide
    sample2 = dataset[1]
    features2 = sample2["input"]
    label2 = sample2["label"]
    slide_name2 = sample2["slide_name"]
    coords2 = sample2["coords"]
    assert slide_name2 == "slide_002"
    assert label2 == 0
    assert features2.shape == (5, 1280)

def test_h5_feature_bag_dataset_transform(mock_fast_extraction_data):
    csv_path, features_dir = mock_fast_extraction_data
    
    # Simple transform that adds 1 to all features
    def mock_transform(x):
        return x + 1.0
        
    dataset = H5FeatureBagDataset(
        csv_path=csv_path,
        features_dir=features_dir,
        label_col_name="label",
        transform=mock_transform
    )
    
    sample = dataset[0]
    features = sample["input"]
    # In the mock, we used rand (0 to 1), so +1 makes all values >= 1.0
    assert torch.all(features >= 1.0)
