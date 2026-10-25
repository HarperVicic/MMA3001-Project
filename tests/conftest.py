"""Shared pytest configuration: make ``src/`` importable and provide synthetic data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def make_env_json(unix_time: int, values: dict) -> str:
    """Build one ``jsondata`` string in the environmental sensor's format."""
    return json.dumps([{"values": [{"time": unix_time, "value": v}], "variable": {"name": name, "unit": ""}}
                       for name, v in values.items()])


@pytest.fixture
def synthetic_regression():
    """Training/test data where headcount depends strongly on 'a', weakly on 'b', and not on 'c'."""
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({"a": rng.uniform(0, 10, n), "b": rng.uniform(0, 10, n), "c": rng.uniform(0, 10, n)})
    y = pd.Series(0.8 * X["a"] + 0.1 * X["b"] + rng.normal(0, 0.3, n), name="avg_headcount")
    return X.iloc[:400], X.iloc[400:], y.iloc[:400], y.iloc[400:]
