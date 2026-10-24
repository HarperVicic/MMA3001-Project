"""
Environmental Feature Importance on Occupancy Modelling.

Predicts the headcount of a room from indoor environmental sensor readings
(CO2, temperature, humidity, formaldehyde, pressure, VOCs and particulate
matter) and ranks which environmental variable is the strongest predictor of
occupancy.

Pipeline
--------
1. Load the occupancy events for one room and the environmental readings for
   one sensor (parsed from JSON, timed by measurement time, not upload time).
2. Align the datasets: pair each 10-minute environmental reading with the
   time-weighted average headcount over the preceding 10 minutes.
3. Split the paired readings 70/30 into training and test sets.
4. Train a degree-2 polynomial regression (baseline), a Random Forest and a
   Gradient Boosting model, then tune the Random Forest.
5. Compare the models on accuracy (RMSE, MAE, R^2), run time and model size,
   and select the model with the lowest test RMSE.
6. Run sensitivity analyses on the final model (train/test split size and
   leave-one-feature-out) and rank the environmental variables.
7. Predict headcount for every environmental reading and save all tables and
   figures to the output folder.

Usage
-----
From the project root::

    python src/occupancy_model.py
    python src/occupancy_model.py --data-dir DATA --output-dir outputs

The functions can also be imported, for example from a notebook::

    from occupancy_model import load_occupancy, time_weighted_headcount
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Project root folder (the parent of ``src/``)."""

DEFAULT_DATA_DIR = PROJECT_ROOT / "DATA"
"""Default folder for the input data files."""

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
"""Default folder for output tables and figures."""

OCCUPANCY_FILE = "5occupancySensor_MayToDec2024_9MRows.csv"
"""Occupancy events CSV: one row per headcount event, all rooms."""

ENVIRONMENT_FILE = "5EnvSensor_MayToDec2024_180kRows.csv"
"""Environmental sensor CSV: readings stored as JSON in the ``jsondata`` column."""

REGISTRY_FILE = "Sensor ID and Locations.xlsx"
"""Sensor-room mapping registry (environmental sensors only)."""

ROOM_ID = "941f0352-bb69-47d0-a9c5-ac29eb4f7319"
"""Occupancy room (``floorspaceid``) used for modelling.

Chosen because, after removing time-of-day effects, sensor 869's CO2 tracks
this room's headcount more closely than any other room's.
"""

ENV_SENSOR_ID = 6012002000869
"""Environmental sensor used for modelling (registered in room G.38).

The only sensor whose readings overlap the occupancy data in time: sensors
227, 241 and 777 repeat a frozen value and sensor 326 starts in late 2025.
"""

EXCLUDED_VARIABLES = ("Battery",)
"""Sensor variables that are not environmental measurements and are dropped."""

WINDOW = "10min"
"""Averaging window for headcount; matches sensor 869's 10-minute reporting interval."""

MAX_GAP = "10min"
"""Longest gap between occupancy readings that is bridged; longer gaps count as missing data."""

MIN_COVERAGE = 0.8
"""Minimum fraction of each window that must be covered by occupancy data."""

TEST_SIZE = 0.30
"""Fraction of the modelling dataset held out for testing (70/30 split)."""

RANDOM_STATE = 42
"""Random seed used for the split and all models, for reproducible results."""

SHORT_NAMES = {
    "Carbon dioxide": "CO₂ (ppm)",
    "Temperature": "Temperature (°C)",
    "Humidity": "Humidity (%RH)",
    "Formaldehyde": "Formaldehyde (µg/m³)",
    "Pressure": "Pressure (hPa)",
    "Light Volatile Organic Compounds": "VOC (ppb)",
    "Particulate matter 1": "PM1 (µg/m³)",
    "Particulate matter 2.5": "PM2.5 (µg/m³)",
    "Particulate matter 4": "PM4 (µg/m³)",
    "Particulate matter 10": "PM10 (µg/m³)",
}
"""Display names with units for each environmental variable, used in tables and figures.

The sensor's "Light Volatile Organic Compounds" channel is the project's VOC
variable. The sensor has no light-level (lux) channel.
"""


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    """Raise ValueError if ``df`` is missing any of ``columns``."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_occupancy(path: str | Path, room_id: str = ROOM_ID) -> pd.DataFrame:
    """
    Load the raw occupancy events for a single room.

    Reads the occupancy CSV, keeps only the requested room, converts the
    timestamps to UTC datetimes and the headcount to numbers, and removes
    rows where either cannot be parsed.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the occupancy CSV. Must contain the columns ``floorspaceid``,
        ``headcount`` and ``collecteddate``.
    room_id : str, optional
        ``floorspaceid`` of the room to keep. Defaults to ``ROOM_ID``.

    Returns
    -------
    pandas.DataFrame
        One row per occupancy event, sorted by time, with columns:

        - ``collecteddate`` (datetime64[ns, UTC]): time of the event.
        - ``headcount`` (float): number of people detected (people, >= 0).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If required columns are missing or the room has no valid rows.

    Examples
    --------
    >>> occ = load_occupancy("DATA/5occupancySensor_MayToDec2024_9MRows.csv")  # doctest: +SKIP
    >>> occ.columns.tolist()  # doctest: +SKIP
    ['collecteddate', 'headcount']
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Occupancy file not found: {path}")

    df = pd.read_csv(path, on_bad_lines="skip", low_memory=False)
    _require_columns(df, ["floorspaceid", "headcount", "collecteddate"], "Occupancy file")

    df = df.loc[df["floorspaceid"] == room_id, ["collecteddate", "headcount"]].copy()
    # 'mixed' handles the varying timestamp formats; unparseable values become NaT
    df["collecteddate"] = pd.to_datetime(df["collecteddate"], format="mixed", utc=True, errors="coerce")
    df["headcount"] = pd.to_numeric(df["headcount"], errors="coerce")
    df = df.dropna().sort_values("collecteddate").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid occupancy rows found for room {room_id}")
    logger.info("Loaded %s occupancy events for room %s", f"{len(df):,}", room_id[:8])
    return df


def parse_sensor_json(raw: str, exclude: tuple[str, ...] = EXCLUDED_VARIABLES) -> list[tuple]:
    """
    Extract (measurement time, variable name, value) records from one JSON payload.

    Each environmental record stores its readings as a JSON list. Every item
    has a ``variable`` (with a ``name``) and a ``values`` list whose first
    entry holds the measurement ``time`` (Unix seconds) and ``value``.

    Parameters
    ----------
    raw : str
        JSON string from the ``jsondata`` column.
    exclude : tuple of str, optional
        Variable names to skip. Defaults to ``("Battery",)``.

    Returns
    -------
    list of tuple
        ``(unix_time, variable_name, value)`` for each valid reading. Returns an
        empty list if ``raw`` is not valid JSON, so one corrupt record does not
        stop the whole load.

    Examples
    --------
    >>> raw = ('[{"values": [{"time": 1709186696, "value": 465}], "variable": {"name": "Carbon dioxide"}},'
    ...        ' {"values": [{"time": 1709186696, "value": 4.1}], "variable": {"name": "Battery"}}]')
    >>> parse_sensor_json(raw)
    [(1709186696, 'Carbon dioxide', 465)]
    >>> parse_sensor_json("not json")
    []
    """
    try:
        variables = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []

    records = []
    for var in variables:
        name = (var.get("variable") or {}).get("name")
        values = var.get("values") or []
        if name is None or name in exclude or not values:
            continue
        records.append((values[0].get("time"), name, values[0].get("value")))
    return records


def load_environmental(
    path: str | Path,
    sensor_id: int = ENV_SENSOR_ID,
    exclude: tuple[str, ...] = EXCLUDED_VARIABLES,
) -> pd.DataFrame:
    """
    Load and flatten the environmental readings for a single sensor.

    The ``createdate`` column records when a record was *uploaded*, which can
    be months after it was measured, so the measurement time stored inside the
    JSON payload is used instead. Each measurement is uploaded more than once,
    so duplicates are removed by keeping one value per variable per timestamp.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the environmental CSV. Must contain ``sensorid`` and ``jsondata``.
    sensor_id : int, optional
        Sensor to keep. Defaults to ``ENV_SENSOR_ID`` (6012002000869).
    exclude : tuple of str, optional
        Variable names to drop. Defaults to ``("Battery",)``.

    Returns
    -------
    pandas.DataFrame
        One row per measurement time, sorted by time, with columns:

        - ``measuredate`` (datetime64[ns, UTC]): when the reading was taken.
        - One float column per environmental variable, named as reported by
          the sensor (e.g. ``Carbon dioxide`` in ppm, ``Temperature`` in deg C).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If required columns are missing or the sensor has no readings.

    Examples
    --------
    >>> env = load_environmental("DATA/5EnvSensor_MayToDec2024_180kRows.csv")  # doctest: +SKIP
    >>> env[["measuredate", "Carbon dioxide"]].head(1)  # doctest: +SKIP
                    measuredate  Carbon dioxide
    0 2024-02-29 06:04:56+00:00           465.0
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Environmental file not found: {path}")

    df = pd.read_csv(path, usecols=["sensorid", "jsondata"], on_bad_lines="skip", engine="python")
    df = df[df["sensorid"] == sensor_id]
    if df.empty:
        raise ValueError(f"No readings found for sensor {sensor_id}")

    records = [rec for raw in df["jsondata"] for rec in parse_sensor_json(raw, exclude)]
    long_df = pd.DataFrame(records, columns=["unix_time", "variable", "value"])
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df["measuredate"] = pd.to_datetime(long_df["unix_time"], unit="s", utc=True, errors="coerce")

    # Reshape to one row per measurement time; 'first' removes duplicate uploads
    env = (
        long_df.dropna(subset=["measuredate"])
        .pivot_table(index="measuredate", columns="variable", values="value", aggfunc="first")
        .reset_index()
        .sort_values("measuredate")
        .reset_index(drop=True)
    )
    env.columns.name = None
    logger.info("Loaded %s unique readings for sensor %s", f"{len(env):,}", sensor_id)
    return env


def load_sensor_registry(path: str | Path) -> pd.DataFrame:
    """
    Load the sensor-room mapping registry.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the registry spreadsheet (``.xlsx``) with columns including
        ``Building``, ``Floor/Room``, ``Area``, ``Sensor`` and ``ID``.

    Returns
    -------
    pandas.DataFrame
        The registry with an added ``ID_str`` column (sensor ID as a string),
        used for matching against sensor IDs from the CSV files.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")
    registry = pd.read_excel(path)
    registry["ID_str"] = registry["ID"].astype(str).str.strip()
    return registry


# ---------------------------------------------------------------------------
# 2. Alignment
# ---------------------------------------------------------------------------

def _to_seconds(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Convert UTC datetimes to float seconds since 1970-01-01."""
    times = pd.to_datetime(pd.Series(times), utc=True)
    return (times - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def time_weighted_headcount(
    occ_times: pd.Series,
    headcount: pd.Series,
    window_end: pd.Series,
    window: str = WINDOW,
    max_gap: str = MAX_GAP,
    min_coverage: float = MIN_COVERAGE,
) -> np.ndarray:
    """
    Time-weighted average headcount over the window before each target time.

    Headcount is treated as a step function (zero-order hold): each reading
    holds until the next one, because a count of people stays constant until
    someone enters or leaves. The average over a window [t - window, t] is

        H_avg = (sum_i h_i * dt_i) / T_valid

    where dt_i is how long reading h_i held inside the window and T_valid is
    the total time covered by valid data. Segments longer than ``max_gap`` are
    treated as missing data rather than being carried forward.

    For speed, the cumulative area and cumulative valid time are computed once,
    so each window average is a difference of two cumulative values
    (O((n + m) log n) for n readings and m windows).

    Parameters
    ----------
    occ_times : pandas.Series of datetime
        Times of the occupancy readings (any order; duplicates are averaged).
    headcount : pandas.Series of float
        Headcount at each time in ``occ_times`` (people).
    window_end : pandas.Series of datetime
        End time of each averaging window (here, each environmental reading).
    window : str, optional
        Window length as a pandas time string. Defaults to ``"10min"``.
    max_gap : str, optional
        Longest gap between readings that is bridged. Defaults to ``"10min"``.
    min_coverage : float, optional
        Minimum fraction (0-1) of the window that must be covered by valid
        data. Defaults to 0.8.

    Returns
    -------
    numpy.ndarray of float
        Average headcount for each window, in the same order as
        ``window_end``. ``NaN`` where coverage is below ``min_coverage``.

    Raises
    ------
    ValueError
        If fewer than two occupancy readings are given, the inputs have
        different lengths, or ``min_coverage`` is outside (0, 1].

    Examples
    --------
    Two people for the first 4 minutes, then five people for 6 minutes:
    the average over the 10-minute window is (2*4 + 5*6) / 10 = 3.8.

    >>> times = pd.to_datetime(["2024-03-01 09:00", "2024-03-01 09:04", "2024-03-01 09:10"], utc=True)
    >>> time_weighted_headcount(pd.Series(times), pd.Series([2, 5, 5]),
    ...                         pd.Series(pd.to_datetime(["2024-03-01 09:10"], utc=True)))
    array([3.8])
    """
    if len(occ_times) != len(headcount):
        raise ValueError("occ_times and headcount must be the same length")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")

    # Average any readings that share a timestamp, then sort by time
    occ = (
        pd.DataFrame({"t": pd.to_datetime(pd.Series(occ_times).values, utc=True),
                      "h": pd.to_numeric(pd.Series(headcount).values, errors="coerce")})
        .dropna()
        .groupby("t", as_index=False)["h"].mean()
    )
    if len(occ) < 2:
        raise ValueError("At least two valid occupancy readings are required")

    t = _to_seconds(occ["t"])
    h = occ["h"].to_numpy(dtype=float)
    window_s = pd.Timedelta(window).total_seconds()
    max_gap_s = pd.Timedelta(max_gap).total_seconds()

    seg_len = np.diff(t)  # how long each reading is held
    seg_valid = seg_len <= max_gap_s  # don't bridge long gaps

    # Cumulative area under the step function and cumulative valid time
    cum_area = np.concatenate([[0.0], np.cumsum(np.where(seg_valid, h[:-1] * seg_len, 0.0))])
    cum_time = np.concatenate([[0.0], np.cumsum(np.where(seg_valid, seg_len, 0.0))])

    def cumulative_at(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Area and valid time accumulated from the first reading up to time(s) x."""
        k = np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 2)
        partial = np.clip(x - t[k], 0, seg_len[k])
        partial = np.where(seg_valid[k], partial, 0.0)
        area = cum_area[k] + h[k] * partial
        valid = cum_time[k] + partial
        # Nothing accumulates before the first reading; everything has after the last
        area = np.where(x < t[0], 0.0, np.where(x > t[-1], cum_area[-1], area))
        valid = np.where(x < t[0], 0.0, np.where(x > t[-1], cum_time[-1], valid))
        return area, valid

    end = _to_seconds(window_end)
    area_end, time_end = cumulative_at(end)
    area_start, time_start = cumulative_at(end - window_s)
    covered = time_end - time_start

    result = np.full(len(end), np.nan)
    enough = covered >= min_coverage * window_s
    result[enough] = (area_end - area_start)[enough] / covered[enough]
    return result


def build_modelling_dataset(
    occupancy: pd.DataFrame,
    environment: pd.DataFrame,
    window: str = WINDOW,
    max_gap: str = MAX_GAP,
    min_coverage: float = MIN_COVERAGE,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Pair each environmental reading with its 10-minute average headcount.

    Parameters
    ----------
    occupancy : pandas.DataFrame
        Output of :func:`load_occupancy` (``collecteddate``, ``headcount``).
    environment : pandas.DataFrame
        Output of :func:`load_environmental` (``measuredate`` plus one column
        per environmental variable).
    window, max_gap, min_coverage : optional
        Passed to :func:`time_weighted_headcount`.

    Returns
    -------
    readings : pandas.DataFrame
        Every environmental reading with columns ``time``, ``avg_headcount``
        (``NaN`` where there is not enough occupancy data) and the features.
    dataset : pandas.DataFrame
        The modelling dataset: rows of ``readings`` where the headcount and
        every feature are present, sorted by time.
    features : list of str
        Names of the environmental feature columns.

    Raises
    ------
    ValueError
        If either input is missing its required columns, or if no reading
        ends up with both a headcount and every feature.

    Examples
    --------
    >>> readings, dataset, features = build_modelling_dataset(occ, env)  # doctest: +SKIP
    >>> len(readings), len(dataset)  # doctest: +SKIP
    (13800, 4009)
    """
    _require_columns(occupancy, ["collecteddate", "headcount"], "occupancy")
    _require_columns(environment, ["measuredate"], "environment")
    features = [c for c in environment.columns if c != "measuredate"]
    if not features:
        raise ValueError("environment has no feature columns")
    readings = environment.rename(columns={"measuredate": "time"}).copy()
    readings.insert(1, "avg_headcount", time_weighted_headcount(
        occupancy["collecteddate"], occupancy["headcount"], readings["time"],
        window=window, max_gap=max_gap, min_coverage=min_coverage,
    ))
    dataset = (
        readings.dropna(subset=["avg_headcount"] + features)
        .sort_values("time")
        .reset_index(drop=True)
    )
    if dataset.empty:
        raise ValueError("No environmental reading has enough occupancy data; check the room, sensor and dates")
    logger.info("Modelling dataset: %s of %s readings have a headcount",
                f"{len(dataset):,}", f"{len(readings):,}")
    return readings, dataset, features


def split_dataset(
    dataset: pd.DataFrame,
    features: list[str],
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Randomly split the modelling dataset into training and test sets.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Output of :func:`build_modelling_dataset`.
    features : list of str
        Feature column names.
    test_size : float, optional
        Fraction of readings held out for testing. Defaults to 0.30.
    random_state : int, optional
        Seed for a reproducible split. Defaults to 42.

    Returns
    -------
    X_train, X_test : pandas.DataFrame
        Feature values for the training and test readings.
    y_train, y_test : pandas.Series
        Average headcount for the training and test readings (people).

    Raises
    ------
    ValueError
        If ``avg_headcount`` or any feature column is missing, or
        ``test_size`` is not between 0 and 1.

    Notes
    -----
    The split is random, so readings 10 minutes apart can fall on both sides.
    This can make test scores optimistic compared with predicting new days.

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = split_dataset(dataset, features)  # doctest: +SKIP
    >>> len(X_train), len(X_test)  # doctest: +SKIP
    (2806, 1203)
    """
    _require_columns(dataset, ["avg_headcount"] + list(features), "dataset")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    return train_test_split(dataset[features], dataset["avg_headcount"],
                            test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------

def make_polynomial_model(degree: int = 2) -> Pipeline:
    """
    Build the global polynomial regression baseline.

    The features are standardised (zero mean, unit variance), expanded into
    all polynomial terms up to ``degree`` (including interactions), and fitted
    by ordinary least squares.

    Parameters
    ----------
    degree : int, optional
        Polynomial degree. Defaults to 2 (linear, squared and pairwise terms).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline with steps ``scaler``, ``poly`` and ``linear``.

    Notes
    -----
    With 10 features, degree 2 gives 65 terms: 10 linear, 10 squared and 45
    pairwise interactions. Strongly correlated features (e.g. the four
    particulate-matter channels) make the individual coefficients unstable.

    Examples
    --------
    >>> make_polynomial_model().named_steps["poly"].degree
    2
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linear", LinearRegression()),
    ])


def make_random_forest(n_estimators: int = 100, random_state: int = RANDOM_STATE, **params) -> RandomForestRegressor:
    """
    Build a Random Forest regressor.

    Parameters
    ----------
    n_estimators : int, optional
        Number of trees. Defaults to 100.
    random_state : int, optional
        Seed for reproducibility. Defaults to 42.
    **params
        Any other ``RandomForestRegressor`` settings (e.g. ``max_depth``).

    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        Unfitted model.

    Examples
    --------
    >>> make_random_forest(n_estimators=300, max_depth=20).get_params()["max_depth"]
    20
    """
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, **params)


def make_gradient_boosting(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int = RANDOM_STATE,
) -> GradientBoostingRegressor:
    """
    Build a Gradient Boosting regressor.

    Parameters
    ----------
    n_estimators : int, optional
        Number of boosting stages (trees). Defaults to 100.
    learning_rate : float, optional
        Shrinkage applied to each tree's contribution. Defaults to 0.1.
    max_depth : int, optional
        Maximum depth of each tree. Defaults to 3.
    random_state : int, optional
        Seed for reproducibility. Defaults to 42.

    Returns
    -------
    sklearn.ensemble.GradientBoostingRegressor
        Unfitted model.
    """
    return GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                     max_depth=max_depth, random_state=random_state)


def select_n_trees(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tree_counts: tuple[int, ...] = (25, 50, 100, 150, 200, 300, 400, 500),
    tolerance: float = 0.005,
    random_state: int = RANDOM_STATE,
) -> tuple[int, pd.DataFrame]:
    """
    Choose the number of Random Forest trees from out-of-bag (OOB) convergence.

    Trees are added incrementally (``warm_start``) and the OOB RMSE is recorded
    after each step. Each tree is trained on a bootstrap sample, so the readings
    it never saw act as a built-in validation set. The smallest forest whose
    OOB RMSE is within ``tolerance`` of the best is selected.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training headcount (people).
    tree_counts : tuple of int, optional
        Forest sizes to evaluate, in increasing order.
    tolerance : float, optional
        Relative tolerance on the best OOB RMSE. Defaults to 0.005 (0.5%).
    random_state : int, optional
        Seed for reproducibility. Defaults to 42.

    Returns
    -------
    n_trees : int
        Selected number of trees.
    convergence : pandas.DataFrame
        Columns ``n_trees`` and ``oob_rmse`` for every forest size evaluated.
    """
    forest = RandomForestRegressor(warm_start=True, oob_score=True, random_state=random_state, n_jobs=-1)
    oob_rmse = []
    for n in tree_counts:
        forest.set_params(n_estimators=n)
        forest.fit(X_train, y_train)  # warm_start adds trees to the existing forest
        oob_rmse.append(np.sqrt(mean_squared_error(y_train, forest.oob_prediction_)))

    convergence = pd.DataFrame({"n_trees": tree_counts, "oob_rmse": oob_rmse})
    best = convergence["oob_rmse"].min()
    n_trees = int(convergence.loc[convergence["oob_rmse"] <= best * (1 + tolerance), "n_trees"].iloc[0])
    return n_trees, convergence


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trees: int,
    param_grid: dict | None = None,
    cv_folds: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple[RandomForestRegressor, pd.DataFrame]:
    """
    Tune Random Forest settings by grid search with k-fold cross-validation.

    Only the training data is used, so the test set stays unseen.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training headcount (people).
    n_trees : int
        Number of trees (e.g. from :func:`select_n_trees`).
    param_grid : dict, optional
        Settings to search. Defaults to ``max_depth`` in {None, 20, 10, 6},
        ``min_samples_leaf`` in {1, 3, 5, 10, 20} and ``max_features`` in
        {1.0, 0.5, "sqrt"}.
    cv_folds : int, optional
        Number of cross-validation folds. Defaults to 5.
    random_state : int, optional
        Seed for reproducibility. Defaults to 42.

    Returns
    -------
    best_model : sklearn.ensemble.RandomForestRegressor
        Model with the lowest mean CV RMSE, refitted on all training data.
    cv_results : pandas.DataFrame
        One row per setting with the parameter values and ``cv_rmse``,
        sorted from best to worst.
    """
    if param_grid is None:
        param_grid = {
            "max_depth": [None, 20, 10, 6],
            "min_samples_leaf": [1, 3, 5, 10, 20],
            "max_features": [1.0, 0.5, "sqrt"],
        }
    search = GridSearchCV(
        RandomForestRegressor(n_estimators=n_trees, random_state=random_state, n_jobs=-1),
        param_grid,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results["cv_rmse"] = -cv_results["mean_test_score"]
    keep = [f"param_{p}" for p in param_grid] + ["cv_rmse"]
    cv_results = cv_results[keep].sort_values("cv_rmse").reset_index(drop=True)
    logger.info("Best Random Forest settings: %s (CV RMSE %.4f)", search.best_params_, cv_results["cv_rmse"].iloc[0])
    return search.best_estimator_, cv_results


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute RMSE, MAE and R^2 for a set of predictions.

    Parameters
    ----------
    y_true : array-like of float
        Measured headcount (people).
    y_pred : array-like of float
        Predicted headcount (people).

    Returns
    -------
    dict
        ``{"RMSE": float, "MAE": float, "R2": float}``. RMSE and MAE are in
        people; R^2 is dimensionless (1 = perfect, 0 = no better than the mean).

    Raises
    ------
    ValueError
        If the inputs have different lengths.

    Examples
    --------
    >>> m = regression_metrics([1, 2, 3, 4], [1, 2, 3, 5])
    >>> round(m["RMSE"], 3), round(m["MAE"], 3), round(m["R2"], 3)
    (0.5, 0.25, 0.8)
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def measure_cost(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                 repeats: int = 3) -> dict[str, float]:
    """
    Measure a model's training time, prediction time and size.

    The model is cloned and refitted ``repeats`` times; median times are
    reported to reduce the effect of background load.

    Parameters
    ----------
    model : sklearn estimator
        Model to measure (its settings are copied; it is not modified).
    X_train, y_train : pandas.DataFrame, pandas.Series
        Training data.
    X_test : pandas.DataFrame
        Data used to time prediction.
    repeats : int, optional
        Number of repetitions. Defaults to 3.

    Returns
    -------
    dict
        ``train_time_s`` (seconds), ``predict_ms_per_1k`` (milliseconds per
        1,000 readings) and ``size_mb`` (pickled model size in megabytes).
    """
    fit_times, predict_times = [], []
    for _ in range(repeats):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        fit_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        m.predict(X_test)
        predict_times.append(time.perf_counter() - t0)
    return {
        "train_time_s": float(np.median(fit_times)),
        "predict_ms_per_1k": float(np.median(predict_times) / len(X_test) * 1e6),
        "size_mb": len(pickle.dumps(m)) / 1e6,
    }


def compare_models(
    models: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Fit (optionally) and evaluate several models on the same split.

    Parameters
    ----------
    models : dict
        ``{name: estimator}``.
    X_train, X_test : pandas.DataFrame
        Training and test features.
    y_train, y_test : pandas.Series
        Training and test headcount (people).
    fit : bool, optional
        Fit each model before evaluating. Set False for already-fitted models.

    Returns
    -------
    comparison : pandas.DataFrame
        One row per model with ``Model``, ``RMSE``, ``MAE``, ``R2``,
        ``train_time_s``, ``predict_ms_per_1k`` and ``size_mb``, sorted by RMSE.
    predictions : dict
        ``{name: test-set predictions}``.
    """
    rows, predictions = [], {}
    for name, model in models.items():
        if fit:
            model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        rows.append({"Model": name,
                     **regression_metrics(y_test, predictions[name]),
                     **measure_cost(model, X_train, y_train, X_test)})
    comparison = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return comparison, predictions


def select_final_model(comparison: pd.DataFrame, models: dict) -> tuple[str, object]:
    """
    Pick the model with the lowest test RMSE.

    Parameters
    ----------
    comparison : pandas.DataFrame
        Output of :func:`compare_models`.
    models : dict
        ``{name: fitted estimator}`` with the same names as ``comparison``.

    Returns
    -------
    name : str
        Name of the selected model.
    model : sklearn estimator
        The selected fitted model.
    """
    name = comparison.sort_values("RMSE")["Model"].iloc[0]
    return name, models[name]


def split_sensitivity(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    test_sizes: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5),
    seeds: tuple[int, ...] = (42, 100, 2024, 777, 99),
) -> pd.DataFrame:
    """
    Test how the model's accuracy depends on the train/test split.

    For every combination of test size and random seed, a fresh copy of the
    model is trained and evaluated.

    Parameters
    ----------
    model : sklearn estimator
        Model whose settings are reused (it is cloned, not modified).
    X : pandas.DataFrame
        All features in the modelling dataset.
    y : pandas.Series
        All headcounts in the modelling dataset (people).
    test_sizes : tuple of float, optional
        Fractions of data to hold out.
    seeds : tuple of int, optional
        Random seeds for the split (and the model, where it has one).

    Returns
    -------
    pandas.DataFrame
        One row per (test size, seed) with ``Test Size``, ``Random Seed``,
        ``RMSE``, ``MAE`` and ``R2``.
    """
    rows = []
    for ts in test_sizes:
        for seed in seeds:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=ts, random_state=seed)
            m = clone(model)
            if "random_state" in m.get_params():
                m.set_params(random_state=seed)
            m.fit(X_tr, y_tr)
            rows.append({"Test Size": ts, "Random Seed": seed, **regression_metrics(y_te, m.predict(X_te))})
    return pd.DataFrame(rows)


def leave_one_feature_out(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Sensitivity analysis: remove each feature in turn and measure the loss.

    For each feature, a fresh copy of the model is trained without it and
    evaluated on the test set. A larger increase in error (or decrease in R^2)
    means the feature carried more useful information.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model using all features (cloned for each reduced model).
    X_train, X_test : pandas.DataFrame
        Training and test features.
    y_train, y_test : pandas.Series
        Training and test headcount (people).

    Returns
    -------
    sensitivity : pandas.DataFrame
        One row per removed feature with ``Dropped Feature``, the new
        ``RMSE``/``MAE``/``R2``, and ``RMSE Increase``, ``MAE Increase`` and
        ``R2 Decrease``, sorted by ``RMSE Increase`` (largest first).
    baseline : dict
        Metrics of ``model`` with all features.

    Examples
    --------
    >>> sensitivity, baseline = leave_one_feature_out(final_model, X_train, X_test, y_train, y_test)  # doctest: +SKIP
    >>> sensitivity.loc[0, "Dropped Feature"]  # doctest: +SKIP
    'Carbon dioxide'
    """
    baseline = regression_metrics(y_test, model.predict(X_test))
    rows = []
    for feature in X_train.columns:
        reduced = clone(model).fit(X_train.drop(columns=feature), y_train)
        m = regression_metrics(y_test, reduced.predict(X_test.drop(columns=feature)))
        rows.append({
            "Dropped Feature": feature, **m,
            "RMSE Increase": m["RMSE"] - baseline["RMSE"],
            "MAE Increase": m["MAE"] - baseline["MAE"],
            "R2 Decrease": baseline["R2"] - m["R2"],
        })
    sensitivity = pd.DataFrame(rows).sort_values("RMSE Increase", ascending=False).reset_index(drop=True)
    return sensitivity, baseline


def rank_features(
    model,
    sensitivity: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    short_names: dict[str, str] = SHORT_NAMES,
) -> pd.DataFrame:
    """
    Rank the environmental variables by three measures of importance.

    1. Sensitivity: increase in test RMSE when the variable is removed.
    2. Model importance: the model's built-in ``feature_importances_``
       (tree-based models only; otherwise NaN and ignored in the average).
    3. Correlation: absolute Pearson correlation with headcount (training data).

    The overall rank is the average of the available ranks, with ties broken
    by the sensitivity rank.

    Parameters
    ----------
    model : sklearn estimator
        The fitted final model.
    sensitivity : pandas.DataFrame
        Output of :func:`leave_one_feature_out`.
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training headcount (people).
    short_names : dict, optional
        Display names with units for each feature.

    Returns
    -------
    pandas.DataFrame
        One row per variable, ordered from most to least important, with the
        three measures, their ranks and the ``Overall rank``.

    Examples
    --------
    >>> ranking = rank_features(final_model, sensitivity, X_train, y_train)  # doctest: +SKIP
    >>> ranking.loc[0, "Variable"]  # doctest: +SKIP
    'CO₂ (ppm)'
    """
    ranking = pd.DataFrame({"Feature": X_train.columns})
    ranking["Variable"] = ranking["Feature"].map(short_names).fillna(ranking["Feature"])
    ranking["RMSE increase when removed"] = ranking["Feature"].map(
        sensitivity.set_index("Dropped Feature")["RMSE Increase"])
    ranking["Model importance"] = getattr(model, "feature_importances_", np.full(len(ranking), np.nan))
    ranking["|Correlation| with headcount"] = X_train.corrwith(y_train).abs().to_numpy()

    rank_cols = {"Sensitivity rank": "RMSE increase when removed",
                 "Importance rank": "Model importance",
                 "Correlation rank": "|Correlation| with headcount"}
    for rank_col, value_col in rank_cols.items():
        ranking[rank_col] = ranking[value_col].rank(ascending=False)
    ranking["Average rank"] = ranking[list(rank_cols)].mean(axis=1, skipna=True)

    ranking = ranking.sort_values(["Average rank", "Sensitivity rank"]).reset_index(drop=True)
    ranking.insert(0, "Overall rank", range(1, len(ranking) + 1))
    return ranking.drop(columns="Feature")


def predict_headcount(
    model,
    readings: pd.DataFrame,
    features: list[str],
    train_times: pd.Series,
    test_times: pd.Series,
    train_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Predict headcount for every environmental reading and flag unsupported inputs.

    Tree-based models cannot extrapolate: for inputs beyond the range seen in
    training they return the prediction for the nearest training values. So
    readings outside the training dates or outside the training range of any
    feature are flagged as less reliable rather than rejected.

    Parameters
    ----------
    model : sklearn estimator
        The fitted final model.
    readings : pandas.DataFrame
        Output ``readings`` of :func:`build_modelling_dataset`.
    features : list of str
        Feature columns used by the model.
    train_times, test_times : pandas.Series of datetime
        Times of the training and test readings, used to label each row.
    train_features : pandas.DataFrame, optional
        Training features (``X_train``). If given, readings with any feature
        outside its training minimum-maximum range are flagged.

    Returns
    -------
    pandas.DataFrame
        Columns ``time``, the features, ``measured_headcount`` (NaN where none
        was recorded), ``predicted_headcount`` (people, clipped at 0 because a
        headcount cannot be negative), ``split`` (``train``, ``test`` or
        ``unlabelled``), ``outside_training_period`` (True for readings outside
        the dates the model was trained on), ``outside_training_range`` (True
        if any feature is outside its training range; only when
        ``train_features`` is given) and ``out_of_range_features`` (which
        features were out of range).

    Raises
    ------
    ValueError
        If ``readings`` is missing ``time`` or any feature column.

    Examples
    --------
    >>> preds = predict_headcount(final_model, readings, features, train_times, test_times, X_train)  # doctest: +SKIP
    >>> preds["split"].value_counts()  # doctest: +SKIP
    unlabelled    9791
    train         2806
    test          1203
    """
    _require_columns(readings, ["time"] + list(features), "readings")
    out = readings.dropna(subset=features).rename(columns={"avg_headcount": "measured_headcount"}).copy()
    out["predicted_headcount"] = np.clip(model.predict(out[features]), 0, None)
    out["split"] = np.select([out["time"].isin(set(train_times)), out["time"].isin(set(test_times))],
                             ["train", "test"], default="unlabelled")

    labelled = pd.concat([pd.Series(train_times), pd.Series(test_times)])
    out["outside_training_period"] = (out["time"] < labelled.min()) | (out["time"] > labelled.max())

    if train_features is not None:
        low, high = train_features[features].min(), train_features[features].max()
        beyond = (out[features] < low) | (out[features] > high)
        out["outside_training_range"] = beyond.any(axis=1)
        out["out_of_range_features"] = beyond.apply(lambda row: ", ".join(row.index[row]), axis=1)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Visualisation
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> Path:
    """Save a figure as a PNG, close it and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_time_series(dataset: pd.DataFrame, features: list[str], path: Path,
                     short_names: dict[str, str] = SHORT_NAMES) -> Path:
    """
    Plot headcount and every environmental variable over time.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Modelling dataset with ``time``, ``avg_headcount`` and the features.
    features : list of str
        Feature columns to plot.
    path : pathlib.Path
        Output PNG file.
    short_names : dict, optional
        Axis labels with units.

    Returns
    -------
    pathlib.Path
        Path of the saved figure.
    """
    cols = ["avg_headcount"] + features
    fig, axes = plt.subplots(len(cols), 1, figsize=(16, 2.2 * len(cols)), sharex=True)
    for ax, col in zip(axes, cols):
        ax.plot(dataset["time"], dataset[col], linewidth=0.6,
                color="dodgerblue" if col == "avg_headcount" else "dimgray")
        ax.set_ylabel("Headcount" if col == "avg_headcount" else short_names.get(col, col), fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Time Series: Headcount and Environmental Variables", fontsize=14)
    fig.tight_layout()
    return _save(fig, path)


def plot_correlation_heatmap(X_train: pd.DataFrame, y_train: pd.Series, path: Path,
                             short_names: dict[str, str] = SHORT_NAMES) -> Path:
    """
    Plot the Pearson correlation matrix of the features and headcount.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training headcount.
    path : pathlib.Path
        Output PNG file.
    short_names : dict, optional
        Display names with units.

    Returns
    -------
    pathlib.Path
        Path of the saved figure.
    """
    data = X_train.assign(Headcount=y_train).rename(columns=short_names)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap (Training Data)", fontsize=14)
    fig.tight_layout()
    return _save(fig, path)


def plot_polynomial_fit(X_train: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                        y_pred_poly: np.ndarray, path: Path,
                        short_names: dict[str, str] = SHORT_NAMES) -> Path:
    """
    Plot a quadratic fit of headcount against each variable, plus the
    polynomial baseline's predicted vs actual headcount.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training headcount.
    y_test : pandas.Series
        Test headcount.
    y_pred_poly : numpy.ndarray
        Polynomial baseline's predictions for the test set.
    path : pathlib.Path
        Output PNG file.
    short_names : dict, optional
        Display names with units.

    Returns
    -------
    pathlib.Path
        Path of the saved figure.
    """
    features = list(X_train.columns)
    n_cols = 4
    n_rows = int(np.ceil((len(features) + 1) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.ravel()
    yv = y_train.to_numpy()
    for ax, feature in zip(axes, features):
        xv = X_train[feature].to_numpy()
        coeffs = np.polyfit(xv, yv, deg=2)
        r2_single = 1 - np.sum((yv - np.polyval(coeffs, xv)) ** 2) / np.sum((yv - yv.mean()) ** 2)
        x_line = np.linspace(xv.min(), xv.max(), 200)
        ax.scatter(xv, yv, s=3, alpha=0.2, color="grey")
        ax.plot(x_line, np.polyval(coeffs, x_line), color="crimson", linewidth=2)
        ax.set_title(f"{short_names.get(feature, feature)}  (R² = {r2_single:.3f})", fontsize=10)
        ax.set_ylabel("Headcount")
        ax.grid(True, linestyle="--", alpha=0.5)

    ax = axes[len(features)]
    ax.scatter(y_test, y_pred_poly, s=4, alpha=0.4, color="dodgerblue")
    lim = max(np.max(y_test), np.max(y_pred_poly))
    ax.plot([0, lim], [0, lim], "k--", linewidth=1)
    ax.set_title(f"Polynomial baseline: predicted vs actual\n(test R² = {r2_score(y_test, y_pred_poly):.3f})",
                 fontsize=10)
    ax.set_xlabel("Actual headcount")
    ax.set_ylabel("Predicted headcount")
    ax.grid(True, linestyle="--", alpha=0.5)
    for ax in axes[len(features) + 1:]:
        ax.set_visible(False)
    fig.suptitle("Polynomial Fits: Headcount vs Each Variable, and the Degree-2 Baseline", fontsize=14)
    fig.tight_layout()
    return _save(fig, path)


def plot_predicted_vs_actual(y_test: pd.Series, predictions: dict[str, np.ndarray], path: Path) -> Path:
    """
    Scatter plot of predicted vs measured test headcount for each model.

    Parameters
    ----------
    y_test : pandas.Series
        Measured test headcount.
    predictions : dict
        ``{model name: test predictions}`` (from :func:`compare_models`).
    path : pathlib.Path
        Output PNG file.

    Returns
    -------
    pathlib.Path
        Path of the saved figure.
    """
    fig, axes = plt.subplots(1, len(predictions), figsize=(5.5 * len(predictions), 5.5), sharex=True, sharey=True)
    lim = max(np.max(y_test), max(np.max(p) for p in predictions.values()))
    for ax, (name, pred) in zip(np.atleast_1d(axes), predictions.items()):
        m = regression_metrics(y_test, pred)
        ax.scatter(y_test, pred, s=6, alpha=0.4)
        ax.plot([0, lim], [0, lim], "k--", linewidth=1)
        ax.set_title(f"{name}\nR² = {m['R2']:.3f}, RMSE = {m['RMSE']:.3f}", fontsize=11)
        ax.set_xlabel("Actual headcount")
        ax.grid(True, linestyle="--", alpha=0.6)
    np.atleast_1d(axes)[0].set_ylabel("Predicted headcount")
    fig.suptitle("Predicted vs Actual Headcount (Test Set)", fontsize=14)
    fig.tight_layout()
    return _save(fig, path)


def plot_tree_convergence(convergence: pd.DataFrame, n_trees: int, path: Path) -> Path:
    """
    Plot out-of-bag RMSE against the number of Random Forest trees.

    Parameters
    ----------
    convergence : pandas.DataFrame
        Output ``convergence`` of :func:`select_n_trees`.
    n_trees : int
        Selected number of trees (marked on the plot).
    path : pathlib.Path
        Output PNG file.

    Returns
    -------
    pathlib.Path
        Path of the saved figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(convergence["n_trees"], convergence["oob_rmse"], marker="o", color="crimson")
    ax.axvline(n_trees, color="black", linestyle="--", linewidth=1, label=f"Selected: {n_trees} trees")
    ax.set_title("Random Forest Convergence: Out-of-Bag RMSE vs Number of Trees", fontsize=13)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("OOB RMSE (headcount)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return _save(fig, path)


def plot_sensitivity(sensitivity: pd.DataFrame, model_name: str, path: Path,
                     short_names: dict[str, str] = SHORT_NAMES) -> Path:
    """
    Bar charts of the loss in RMSE, MAE and R^2 when each variable is removed.

    Parameters
    ----------
    sensitivity : pandas.DataFrame
        Output of :func:`leave_one_feature_out`.
    model_name : str
        Name of the model analysed (used in the title).
    path : pathlib.Path
        Output PNG file.
    short_names : dict, optional
        Display names with units.

    Returns
    -------
    pathlib.Path
        Path of the saved figure.
    """
    data = sensitivity.assign(Variable=sensitivity["Dropped Feature"].map(short_names)
                              .fillna(sensitivity["Dropped Feature"]))
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    for ax, metric in zip(axes, ["RMSE Increase", "MAE Increase", "R2 Decrease"]):
        colours = ["crimson" if v > 0 else "lightgrey" for v in data[metric]]
        ax.barh(data["Variable"], data[metric], color=colours)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(metric.replace("R2", "R²"), fontsize=12)
        ax.grid(axis="x", linestyle="--", alpha=0.6)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Variable removed")
    fig.suptitle(f"Sensitivity Analysis ({model_name}): Performance Loss When Each Variable Is Removed",
                 fontsize=14)
    fig.tight_layout()
    return _save(fig, path)


def plot_predictions(predictions: pd.DataFrame, model_name: str, path: Path) -> Path:
    """
    Plot predicted and measured headcount over time.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Output of :func:`predict_headcount`.
    model_name : str
        Name of the model used (for the title).
    path : pathlib.Path
        Output PNG file.

    Returns
    -------
    pathlib.Path
        Path of the saved figure.
    """
    labelled = predictions[predictions["split"] != "unlabelled"]
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(predictions["time"], predictions["predicted_headcount"], color="crimson", linewidth=0.6,
            alpha=0.8, label="Predicted headcount")
    ax.scatter(predictions["time"], predictions["measured_headcount"], s=3, color="black", alpha=0.5,
               label="Measured headcount (10-min average)")
    ax.axvspan(labelled["time"].min(), labelled["time"].max(), color="grey", alpha=0.12,
               label="Period with measured headcount")
    ax.set_title(f"Predicted Headcount from Environmental Data ({model_name})", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Headcount")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------

def main(data_dir: str | Path = DEFAULT_DATA_DIR, output_dir: str | Path = DEFAULT_OUTPUT_DIR,
         tune: bool = True) -> dict:
    """
    Run the full occupancy-modelling pipeline and save every output.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Folder containing the occupancy CSV, environmental CSV and registry.
    output_dir : str or pathlib.Path, optional
        Folder for the output tables (CSV) and figures (``figures/*.png``).
        Created if it does not exist.
    tune : bool, optional
        Tune the Random Forest (takes a few minutes). Defaults to True.

    Returns
    -------
    dict
        Key results: ``final_model_name``, ``final_model``, ``comparison``,
        ``sensitivity``, ``ranking``, ``predictions`` and ``features``.

    Raises
    ------
    FileNotFoundError
        If any input file is missing from ``data_dir``.
    ValueError
        If an input file is missing required columns or contains no usable data.

    Notes
    -----
    Files written to ``output_dir``:

    - ``predicted_headcount.csv``: prediction for every environmental reading,
      with flags for readings outside the training period or range.
    - ``model_performance_metrics.csv``: RMSE, MAE, R^2, run time and size.
    - ``sensitivity_analysis_results.csv``: leave-one-feature-out results.
    - ``split_sensitivity_results.csv``: accuracy vs train/test split.
    - ``feature_importance_ranking.csv``: ranking of the variables.
    - ``rf_tuning_cv_results.csv``: cross-validation results for every
      Random Forest setting (only when ``tune=True``).
    - ``figures/*.png``: time series, correlation heatmap, polynomial fit,
      predicted vs actual, tree convergence (``tune=True``), sensitivity and
      predictions over time.

    Examples
    --------
    From the command line, in the project root::

        python src/occupancy_model.py
        python src/occupancy_model.py --no-tune

    From Python:

    >>> results = main(tune=False)  # doctest: +SKIP
    >>> results["final_model_name"]  # doctest: +SKIP
    'Random Forest'
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    # 1. Load data
    occupancy = load_occupancy(data_dir / OCCUPANCY_FILE)
    environment = load_environmental(data_dir / ENVIRONMENT_FILE)
    registry = load_sensor_registry(data_dir / REGISTRY_FILE)
    location = registry.loc[registry["ID_str"] == str(ENV_SENSOR_ID), ["Building", "Floor/Room", "Area"]]
    if not location.empty:
        logger.info("Sensor %s is registered in: %s", ENV_SENSOR_ID, location.iloc[0].to_dict())

    # 2. Align and split
    readings, dataset, features = build_modelling_dataset(occupancy, environment)
    X_train, X_test, y_train, y_test = split_dataset(dataset, features)
    logger.info("Training readings: %d | test readings: %d", len(X_train), len(X_test))

    # 3. Baseline and alternative models
    models = {
        "Polynomial Regression (Deg 2)": make_polynomial_model(degree=2),
        "Random Forest": make_random_forest(n_estimators=100),
        "Gradient Boosting": make_gradient_boosting(),
    }
    for model in models.values():
        model.fit(X_train, y_train)

    # 4. Tune the Random Forest
    if tune:
        n_trees, convergence = select_n_trees(X_train, y_train)
        plot_tree_convergence(convergence, n_trees, fig_dir / "rf_tree_convergence.png")
        models["Random Forest (Tuned)"], cv_results = tune_random_forest(X_train, y_train, n_trees)
        cv_results.to_csv(output_dir / "rf_tuning_cv_results.csv", index=False)

    # 5. Compare and select the final model
    comparison, test_predictions = compare_models(models, X_train, X_test, y_train, y_test, fit=False)
    final_name, final_model = select_final_model(comparison, models)
    comparison["Role"] = comparison["Model"].map(
        lambda m: "Baseline" if m.startswith("Polynomial") else "Final model" if m == final_name else "Alternative")
    comparison.to_csv(output_dir / "model_performance_metrics.csv", index=False)
    logger.info("Final model: %s\n%s", final_name, comparison.round(4).to_string(index=False))

    # 6. Sensitivity analysis and feature ranking
    split_results = split_sensitivity(final_model, dataset[features], dataset["avg_headcount"])
    split_results.to_csv(output_dir / "split_sensitivity_results.csv", index=False)
    sensitivity, baseline = leave_one_feature_out(final_model, X_train, X_test, y_train, y_test)
    sensitivity.to_csv(output_dir / "sensitivity_analysis_results.csv", index=False)
    ranking = rank_features(final_model, sensitivity, X_train, y_train)
    ranking.to_csv(output_dir / "feature_importance_ranking.csv", index=False)
    logger.info("Feature ranking:\n%s", ranking[["Overall rank", "Variable"]].to_string(index=False))

    # 7. Predict headcount for every reading
    predictions = predict_headcount(final_model, readings, features,
                                    train_times=dataset.loc[X_train.index, "time"],
                                    test_times=dataset.loc[X_test.index, "time"],
                                    train_features=X_train)
    predictions.to_csv(output_dir / "predicted_headcount.csv", index=False)
    unlabelled = predictions[predictions["split"] == "unlabelled"]
    logger.info("Unlabelled readings: %d | outside training period: %d | outside training range: %d",
                len(unlabelled), unlabelled["outside_training_period"].sum(),
                unlabelled["outside_training_range"].sum())

    # 8. Figures
    plot_time_series(dataset, features, fig_dir / "time_series.png")
    plot_correlation_heatmap(X_train, y_train, fig_dir / "correlation_heatmap.png")
    plot_polynomial_fit(X_train, y_train, y_test, test_predictions["Polynomial Regression (Deg 2)"],
                        fig_dir / "polynomial_fit.png")
    plot_predicted_vs_actual(y_test, test_predictions, fig_dir / "predicted_vs_actual.png")
    plot_sensitivity(sensitivity, final_name, fig_dir / "sensitivity_analysis.png")
    plot_predictions(predictions, final_name, fig_dir / "predicted_headcount_time_series.png")

    logger.info("Strongest predictor of occupancy: %s", ranking.loc[0, "Variable"])
    logger.info("Finished in %.0f s. Outputs saved to %s", time.perf_counter() - start, output_dir.resolve())
    return {"final_model_name": final_name, "final_model": final_model, "comparison": comparison,
            "sensitivity": sensitivity, "ranking": ranking, "predictions": predictions, "features": features}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict room headcount from environmental sensor data.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="folder containing the input data files")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="folder for output tables and figures")
    parser.add_argument("--no-tune", action="store_true", help="skip Random Forest tuning (faster)")
    args = parser.parse_args()

    # Render figures straight to files. This is set only when the script is run
    # directly, so importing the module into a notebook keeps inline plots.
    matplotlib.use("Agg")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    main(args.data_dir, args.output_dir, tune=not args.no_tune)
