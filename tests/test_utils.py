"""
Unit tests for src/utils.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.utils import (
    load_artifact,
    load_config,
    load_json,
    reduce_mem_usage,
    save_artifact,
    save_json,
    setup_logger,
)


def test_reduce_mem_usage():
    # Create sample dataframe with int64 and float64
    df = pd.DataFrame({
        "int_small": np.array([1, 2, 3, 4], dtype=np.int64),
        "int_large": np.array([1000, 2000, 3000, 4000], dtype=np.int64),
        "float_col": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64),
        "text_col": ["a", "b", "c", "d"],
    })
    
    start_mem = df.memory_usage().sum()
    reduced_df = reduce_mem_usage(df, verbose=False)
    end_mem = reduced_df.memory_usage().sum()

    assert end_mem <= start_mem
    assert reduced_df["int_small"].dtype == np.int8
    assert reduced_df["int_large"].dtype == np.int16
    assert reduced_df["float_col"].dtype == np.float32


def test_load_config():
    config = load_config("config/config.yaml")
    assert isinstance(config, dict)
    assert "project" in config
    assert config["project"]["name"] == "IEEE-CIS Fraud Detection"


def test_artifact_persistence(tmp_path):
    sample_obj = {"model_name": "LightGBM", "threshold": 0.35, "scores": [0.88, 0.96]}
    art_path = tmp_path / "test_artifact.pkl"

    save_artifact(sample_obj, str(art_path))
    assert art_path.exists()

    loaded = load_artifact(str(art_path))
    assert loaded == sample_obj


def test_json_persistence(tmp_path):
    data = {"status": "success", "rows": 1000}
    json_path = tmp_path / "metadata.json"

    save_json(data, str(json_path))
    assert json_path.exists()

    loaded = load_json(str(json_path))
    assert loaded == data


def test_setup_logger():
    logger = setup_logger("test_logger")
    assert logger.name == "test_logger"
