"""
Tests for src/occupancy_model.py.

Run from the repository root:

    pytest                                         # run all tests
    pytest --html=tests/test_report.html --self-contained-html   # HTML test report (pytest-html)
    pytest --junitxml=tests/test_report.xml        # XML test report (built into pytest)

The tests use small synthetic datasets with known answers, so they run in a few
seconds and do not need the project data files.
"""
import zipfile

import numpy as np
import pandas as pd
import pytest
from conftest import make_env_json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import occupancy_model as om

UTC = "UTC"


def ts(*times):
    """Series of UTC timestamps from strings."""
    return pd.Series(pd.to_datetime(list(times), utc=True))


# ===========================================================================
# 1. Data loading and input handling
# ===========================================================================

class TestResolveDataFile:
    def test_prefers_uncompressed_file(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("a\n1\n")
        (tmp_path / "data.csv.zip").write_text("zip")
        assert om.resolve_data_file(csv) == csv

    def test_falls_back_to_zip(self, tmp_path):
        zipped = tmp_path / "data.csv.zip"
        zipped.write_text("zip")
        assert om.resolve_data_file(tmp_path / "data.csv") == zipped

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            om.resolve_data_file(tmp_path / "missing.csv")


def write_occupancy_csv(path):
    pd.DataFrame({
        "floorspaceid": [om.ROOM_ID, om.ROOM_ID, "other-room", om.ROOM_ID, om.ROOM_ID],
        "headcount": ["3", "1", "9", "not-a-number", "2"],
        "collecteddate": ["2024-03-01 09:10:00+11", "2024-03-01 09:00:00+11", "2024-03-01 09:05:00+11",
                          "2024-03-01 09:20:00+11", "not-a-date"],
    }).to_csv(path, index=False)


class TestLoadOccupancy:
    def test_keeps_one_room_drops_invalid_rows_and_sorts(self, tmp_path):
        path = tmp_path / "occ.csv"
        write_occupancy_csv(path)
        occ = om.load_occupancy(path)
        assert occ.columns.tolist() == ["collecteddate", "headcount"]
        assert occ["headcount"].tolist() == [1.0, 3.0]          # other room and invalid rows removed
        assert occ["collecteddate"].is_monotonic_increasing
        assert str(occ["collecteddate"].dt.tz) == UTC

    def test_reads_zipped_csv(self, tmp_path):
        path = tmp_path / "occ.csv"
        write_occupancy_csv(path)
        with zipfile.ZipFile(tmp_path / "occ.csv.zip", "w") as z:
            z.write(path, arcname="occ.csv")
        path.unlink()
        assert len(om.load_occupancy(path)) == 2

    def test_missing_columns_raise(self, tmp_path):
        path = tmp_path / "occ.csv"
        pd.DataFrame({"headcount": [1]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            om.load_occupancy(path)

    def test_room_without_data_raises(self, tmp_path):
        path = tmp_path / "occ.csv"
        write_occupancy_csv(path)
        with pytest.raises(ValueError, match="No valid occupancy rows"):
            om.load_occupancy(path, room_id="no-such-room")


class TestParseSensorJson:
    def test_extracts_variables_and_excludes_battery(self):
        raw = make_env_json(1709186696, {"Carbon dioxide": 465, "Battery": 4.1, "Temperature": 23.5})
        assert om.parse_sensor_json(raw) == [(1709186696, "Carbon dioxide", 465), (1709186696, "Temperature", 23.5)]

    @pytest.mark.parametrize("bad", ["not json", None, "[]"])
    def test_invalid_or_empty_records_give_no_readings(self, bad):
        assert om.parse_sensor_json(bad) == []


class TestLoadEnvironmental:
    def write(self, path):
        t1, t2 = 1709186696, 1709187296  # 10 minutes apart
        rows = [
            # the same measurement uploaded twice, months later: createdate must be ignored
            {"sensorid": om.ENV_SENSOR_ID, "jsondata": make_env_json(t1, {"Carbon dioxide": 465, "Battery": 4.1}),
             "createdate": "2024-09-01 00:00:00"},
            {"sensorid": om.ENV_SENSOR_ID, "jsondata": make_env_json(t1, {"Carbon dioxide": 465, "Battery": 4.1}),
             "createdate": "2024-10-01 00:00:00"},
            {"sensorid": om.ENV_SENSOR_ID, "jsondata": make_env_json(t2, {"Carbon dioxide": 480, "Battery": 4.1}),
             "createdate": "2024-09-01 00:00:00"},
            {"sensorid": 999, "jsondata": make_env_json(t1, {"Carbon dioxide": 900}), "createdate": "2024-09-01"},
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        return t1, t2

    def test_uses_measurement_time_removes_duplicates_and_battery(self, tmp_path):
        path = tmp_path / "env.csv"
        t1, t2 = self.write(path)
        env = om.load_environmental(path)
        assert env.columns.tolist() == ["measuredate", "Carbon dioxide"]
        assert len(env) == 2                                     # duplicate upload removed, other sensor ignored
        assert env["measuredate"].tolist() == pd.to_datetime([t1, t2], unit="s", utc=True).tolist()
        assert env["Carbon dioxide"].tolist() == [465, 480]

    def test_unknown_sensor_raises(self, tmp_path):
        path = tmp_path / "env.csv"
        self.write(path)
        with pytest.raises(ValueError, match="No readings"):
            om.load_environmental(path, sensor_id=12345)


# ===========================================================================
# 2. Time-weighted averaging (numerical method verification)
# ===========================================================================

class TestTimeWeightedHeadcount:
    def test_hand_calculated_example(self):
        # 2 people for 4 minutes, then 5 people for 6 minutes: (2*4 + 5*6) / 10 = 3.8
        result = om.time_weighted_headcount(ts("2024-03-01 09:00", "2024-03-01 09:04", "2024-03-01 09:10"),
                                            pd.Series([2, 5, 5]), ts("2024-03-01 09:10"))
        assert result.tolist() == pytest.approx([3.8])

    def test_constant_headcount_returns_the_constant(self):
        times = pd.Series(pd.date_range("2024-03-01 09:00", periods=21, freq="1min", tz=UTC))
        result = om.time_weighted_headcount(times, pd.Series([4.0] * 21), ts("2024-03-01 09:10", "2024-03-01 09:20"))
        assert result.tolist() == pytest.approx([4.0, 4.0])

    def test_matches_second_by_second_brute_force(self):
        rng = np.random.default_rng(1)
        start = pd.Timestamp("2024-03-01 09:00", tz=UTC)
        offsets = np.sort(rng.choice(np.arange(0, 3600), size=300, replace=False))
        times = pd.Series(start + pd.to_timedelta(offsets, unit="s"))
        counts = pd.Series(rng.integers(0, 6, size=300).astype(float))
        ends = pd.Series(start + pd.to_timedelta([1200, 2400, 3500], unit="s"))
        fast = om.time_weighted_headcount(times, counts, ends)
        for end, value in zip(ends, fast):
            grid = pd.date_range(end - pd.Timedelta("10min"), end, freq="1s", inclusive="left")
            idx = np.searchsorted(times.values, grid.values, side="right") - 1
            assert value == pytest.approx(counts.values[idx].mean(), abs=1e-9)

    def test_gap_longer_than_max_gap_is_not_bridged(self):
        # a 30-minute gap: the window inside it has no valid data
        result = om.time_weighted_headcount(ts("2024-03-01 09:00", "2024-03-01 09:30", "2024-03-01 09:31"),
                                            pd.Series([3, 1, 1]), ts("2024-03-01 09:20"))
        assert np.isnan(result[0])

    def test_window_with_low_coverage_is_dropped(self):
        # readings cover only the last 5 minutes of the window (50% < 80%)
        times = pd.Series(pd.date_range("2024-03-01 09:05", "2024-03-01 09:15", freq="1min", tz=UTC))
        result = om.time_weighted_headcount(times, pd.Series([2.0] * len(times)), ts("2024-03-01 09:10"))
        assert np.isnan(result[0])

    def test_unsorted_and_duplicate_times_give_same_result(self):
        times = ts("2024-03-01 09:04", "2024-03-01 09:00", "2024-03-01 09:10", "2024-03-01 09:00")
        counts = pd.Series([5, 1, 5, 3])  # two readings at 09:00 are averaged to 2
        result = om.time_weighted_headcount(times, counts, ts("2024-03-01 09:10"))
        assert result.tolist() == pytest.approx([3.8])

    @pytest.mark.parametrize("kwargs", [{"min_coverage": 0}, {"min_coverage": 1.5}])
    def test_invalid_coverage_raises(self, kwargs):
        with pytest.raises(ValueError):
            om.time_weighted_headcount(ts("2024-03-01 09:00", "2024-03-01 09:10"), pd.Series([1, 1]),
                                       ts("2024-03-01 09:10"), **kwargs)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            om.time_weighted_headcount(ts("2024-03-01 09:00", "2024-03-01 09:10"), pd.Series([1]),
                                       ts("2024-03-01 09:10"))

    def test_fewer_than_two_readings_raises(self):
        with pytest.raises(ValueError):
            om.time_weighted_headcount(ts("2024-03-01 09:00"), pd.Series([1]), ts("2024-03-01 09:10"))


class TestBuildDatasetAndSplit:
    def make_inputs(self):
        occ_times = pd.date_range("2024-03-01 09:00", "2024-03-01 11:00", freq="30s", tz=UTC)
        occupancy = pd.DataFrame({"collecteddate": occ_times, "headcount": 2.0})
        env_times = pd.date_range("2024-03-01 09:10", "2024-03-01 12:00", freq="10min", tz=UTC)
        environment = pd.DataFrame({"measuredate": env_times, "Carbon dioxide": 450.0, "Temperature": 22.0})
        return occupancy, environment

    def test_only_readings_with_occupancy_data_are_kept(self):
        occupancy, environment = self.make_inputs()
        readings, dataset, features = om.build_modelling_dataset(occupancy, environment)
        assert features == ["Carbon dioxide", "Temperature"]
        assert len(readings) == len(environment)
        assert len(dataset) == 12                       # 09:10 to 11:00; later readings have no occupancy data
        assert (dataset["avg_headcount"] == 2.0).all()

    def test_missing_columns_raise(self):
        occupancy, environment = self.make_inputs()
        with pytest.raises(ValueError):
            om.build_modelling_dataset(occupancy.drop(columns="headcount"), environment)

    def test_split_is_70_30_and_reproducible(self):
        dataset = pd.DataFrame({"x": np.arange(100.0), "avg_headcount": np.arange(100.0)})
        first = om.split_dataset(dataset, ["x"])
        second = om.split_dataset(dataset, ["x"])
        assert (len(first[0]), len(first[1])) == (70, 30)
        assert first[1].index.equals(second[1].index)

    def test_invalid_test_size_raises(self):
        with pytest.raises(ValueError):
            om.split_dataset(pd.DataFrame({"x": [1.0], "avg_headcount": [1.0]}), ["x"], test_size=1.5)


# ===========================================================================
# 3. Models and evaluation
# ===========================================================================

class TestModels:
    def test_polynomial_has_65_terms_for_10_features(self):
        X = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 10)))
        model = om.make_polynomial_model().fit(X, X.sum(axis=1))
        assert model.named_steps["poly"].n_output_features_ == 65

    def test_polynomial_fits_an_exact_quadratic(self):
        X = pd.DataFrame({"x": np.linspace(-3, 3, 40)})
        y = 1 + 2 * X["x"] - 0.5 * X["x"] ** 2
        assert om.regression_metrics(y, om.make_polynomial_model().fit(X, y).predict(X))["RMSE"] < 1e-9

    def test_random_forest_settings_are_passed_through(self):
        model = om.make_random_forest(n_estimators=7, max_depth=3)
        assert (model.n_estimators, model.max_depth, model.random_state) == (7, 3, om.RANDOM_STATE)


class TestRegressionMetrics:
    def test_known_values(self):
        m = om.regression_metrics([1, 2, 3, 4], [1, 2, 3, 5])
        assert (m["RMSE"], m["MAE"], m["R2"]) == pytest.approx((0.5, 0.25, 0.8))

    def test_perfect_prediction(self):
        m = om.regression_metrics([1, 2, 3], [1, 2, 3])
        assert (m["RMSE"], m["MAE"], m["R2"]) == pytest.approx((0.0, 0.0, 1.0))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            om.regression_metrics([1, 2, 3], [1, 2])


class TestModelSelectionAndSensitivity:
    def test_lowest_rmse_model_is_selected(self, synthetic_regression):
        X_train, X_test, y_train, y_test = synthetic_regression
        models = {"Linear": LinearRegression(), "Mean-like": om.make_random_forest(n_estimators=5, max_depth=1)}
        comparison, predictions = om.compare_models(models, X_train, X_test, y_train, y_test)
        name, model = om.select_final_model(comparison, models)
        assert name == "Linear" and model is models["Linear"]
        assert set(predictions) == set(models)

    def test_leave_one_out_identifies_the_informative_feature(self, synthetic_regression):
        X_train, X_test, y_train, y_test = synthetic_regression
        model = LinearRegression().fit(X_train, y_train)
        sensitivity, baseline = om.leave_one_feature_out(model, X_train, X_test, y_train, y_test)
        assert len(sensitivity) == 3
        assert sensitivity["Dropped Feature"].tolist()[0] == "a"
        assert sensitivity.set_index("Dropped Feature").loc["c", "RMSE Increase"] == pytest.approx(0, abs=0.02)
        assert baseline["R2"] > 0.9

    def test_ranking_puts_the_informative_feature_first(self, synthetic_regression):
        X_train, X_test, y_train, y_test = synthetic_regression
        model = om.make_random_forest(n_estimators=30).fit(X_train, y_train)
        sensitivity, _ = om.leave_one_feature_out(model, X_train, X_test, y_train, y_test)
        ranking = om.rank_features(model, sensitivity, X_train, y_train, short_names={})
        assert ranking["Variable"].tolist()[0] == "a"
        assert ranking["Overall rank"].tolist() == [1, 2, 3]

    def test_split_sensitivity_returns_one_row_per_setting(self, synthetic_regression):
        X_train, _, y_train, _ = synthetic_regression
        result = om.split_sensitivity(LinearRegression(), X_train, y_train, test_sizes=(0.2, 0.4), seeds=(1, 2, 3))
        assert len(result) == 6
        assert {"RMSE", "MAE", "R2"} <= set(result.columns)


class TestPredictHeadcount:
    def test_predictions_are_non_negative_and_flagged(self):
        times = pd.Series(pd.date_range("2024-03-01", periods=6, freq="10min", tz=UTC))
        train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        model = LinearRegression().fit(train, [-1.0, 0.0, 1.0])     # predicts negative values for x < 1
        readings = pd.DataFrame({"time": times, "avg_headcount": [0.0, 1.0, 2.0, np.nan, np.nan, np.nan],
                                 "x": [0.0, 1.0, 2.0, 0.5, 5.0, 1.0]})
        out = om.predict_headcount(model, readings, ["x"], train_times=times[:2], test_times=times[2:3],
                                   train_features=train)
        assert (out["predicted_headcount"] >= 0).all()
        assert out["split"].tolist() == ["train", "train", "test", "unlabelled", "unlabelled", "unlabelled"]
        assert out["outside_training_period"].tolist() == [False, False, False, True, True, True]
        assert out["outside_training_range"].tolist() == [False, False, False, False, True, False]
        assert out.loc[4, "out_of_range_features"] == "x"


# ===========================================================================
# 4. Performance analysis
# ===========================================================================

class TestPerformance:
    def test_original_method_uses_trapezoids(self):
        # a step from 2 to 4 people: trapezoids average to 3, the true time-weighted value is 2
        occ = pd.DataFrame({"collecteddate": pd.to_datetime(["2024-03-01 09:00", "2024-03-01 09:10"], utc=True),
                            "headcount": [2.0, 4.0]})
        assert float(om.original_trapezoid_alignment(occ).iloc[0]) == pytest.approx(3.0)
        step = om.time_weighted_headcount(occ["collecteddate"], occ["headcount"], ts("2024-03-01 09:10"))
        assert step.tolist() == pytest.approx([2.0])

    def test_benchmark_reports_both_methods(self):
        occ_times = pd.date_range("2024-03-01 09:00", "2024-03-01 10:00", freq="30s", tz=UTC)
        occupancy = pd.DataFrame({"collecteddate": occ_times, "headcount": 1.0})
        environment = pd.DataFrame({"measuredate": pd.date_range("2024-03-01 09:10", periods=5, freq="10min", tz=UTC)})
        result = om.benchmark_alignment(occupancy, environment, repeats=1)
        assert len(result) == 2 and (result["median_time_s"] > 0).all()

    def test_flop_estimate(self):
        est = om.alignment_flop_estimate(1024, 100)
        assert est["flops"] == 5 * 1024 + 100 * (2 * 10 + 20)
        assert est["arithmetic_intensity"] == pytest.approx(est["flops"] / est["bytes"])

    def test_flop_estimate_rejects_too_little_data(self):
        with pytest.raises(ValueError):
            om.alignment_flop_estimate(1, 10)

    def test_polynomial_operation_count(self):
        assert om.polynomial_flops_per_prediction(10) == 205

    def test_tree_comparisons_for_stumps(self):
        # depth-1 trees make exactly one comparison each
        X = pd.DataFrame({"x": np.arange(100.0)})
        model = GradientBoostingRegressor(n_estimators=12, max_depth=1, random_state=0).fit(X, X["x"])
        assert om.tree_comparisons_per_prediction(model, X) == pytest.approx(12.0)

    def test_tree_comparisons_rejects_non_tree_models(self):
        with pytest.raises(TypeError):
            om.tree_comparisons_per_prediction(LinearRegression(), pd.DataFrame({"x": [1.0]}))
