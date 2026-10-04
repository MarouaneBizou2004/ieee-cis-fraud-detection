"""
Utility functions for memory optimization, configuration loading, logging, and artifact persistence.
Designed for clarity and efficiency with no unnecessary complexity.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import yaml


def setup_logger(name: str = "fraud_ml", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configures and returns a standard Python logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Optional file handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger


logger = setup_logger("utils")


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads YAML configuration file into a dictionary.
    Falls back gracefully if file is not found.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found at {config_path}. Using default configuration.")
        return {}
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Iterates through all numerical columns of a DataFrame and modifies the data type
    to reduce memory usage. Essential for the large IEEE-CIS dataset.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and not isinstance(col_type, pd.CategoricalDtype):
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        logger.info(
            f"Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB "
            f"({reduction:.1f}% reduction)"
        )
    return df


def save_artifact(obj: Any, filepath: str) -> None:
    """
    Saves a Python object using joblib with compression.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)
    logger.info(f"Saved artifact to {filepath}")


def load_artifact(filepath: str) -> Any:
    """
    Loads a serialized artifact using joblib.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {filepath}")
    return joblib.load(path)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Saves a dictionary as a formatted JSON file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved metadata to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Loads a JSON file into a dictionary.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found at {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
