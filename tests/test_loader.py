"""
tests/test_loader.py
--------------------
Unit tests for tictactrip.loader.
"""

import numpy as np
import pandas as pd
import pytest

from tictactrip.loader import _parse_stop_count, build_dataset, load_raw


def _make_raw() -> dict:
    tickets = pd.DataFrame({
        "id":             [1, 2, 3],
        "company":        [10, 20, 30],
        "o_station":      [np.nan, 1.0, np.nan],
        "d_station":      [np.nan, 2.0, np.nan],
        "departure_ts":   [
            "2017-10-13 14:00:00+00",
            "2017-10-14 08:00:00+00",
            "2017-10-15 10:00:00+00",
        ],
        "arrival_ts":     [
            "2017-10-13 20:00:00+00",
            "2017-10-14 18:00:00+00",
            "2017-10-15 14:00:00+00",
        ],
        "price_in_cents": [5000, 2500, 1500],
        "search_ts":      [
            "2017-10-01 00:00:00+00",
            "2017-10-10 12:00:00+00",
            "2017-10-14 09:00:00+00",
        ],
        "middle_stations":  [np.nan, "{101,202}", np.nan],
        "other_companies":  [np.nan, "{20}", np.nan],
        "o_city":           [1, 1, 2],
        "d_city":           [2, 3, 1],
    })
    cities = pd.DataFrame({
        "id":          [1, 2, 3],
        "unique_name": ["Paris, Ile-de-France", "Lyon, Auvergne", "Marseille, PACA"],
        "latitude":    [48.8566, 45.7640, 43.2965],
        "longitude":   [2.3522,  4.8357,  5.3698],
        "population":  [2_161_000, 513_000, 861_000],
    })
    stations = pd.DataFrame({
        "id": [1, 2], "unique_name": ["Paris Gare de Lyon", "Lyon Part-Dieu"],
        "latitude": [48.8448, 45.7602], "longitude": [2.3735, 4.8596],
    })
    providers = pd.DataFrame({
        "id":                   [10, 20, 30],
        "company_id":           [1, 2, 3],
        "provider_id":          [np.nan, np.nan, np.nan],
        "name":                 ["tgv", "flixbus", "bbc"],
        "fullname":             ["TGV", "FlixBus", "BlaBlaCar"],
        "has_wifi":             [False, True, False],
        "has_plug":             [True, False, False],
        "has_adjustable_seats": [True, False, False],
        "has_bicycle":          [False, False, False],
        "transport_type":       ["train", "bus", "carpooling"],
    })
    return {"tickets": tickets, "cities": cities,
            "stations": stations, "providers": providers}


@pytest.fixture
def raw():
    return _make_raw()


@pytest.fixture
def df(raw):
    return build_dataset(raw)


def test_load_raw_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="ticket_data.csv"):
        load_raw(tmp_path)


@pytest.mark.parametrize("value,expected", [
    (np.nan,              0),
    ("{}",                0),
    ("",                  0),
    ("{101}",             1),
    ("{101,202}",         2),
    ("{798,798,6794,6246}", 4),
])
def test_parse_stop_count(value, expected):
    assert _parse_stop_count(value) == expected


def test_build_dataset_returns_dataframe(df):
    assert isinstance(df, pd.DataFrame)


def test_build_dataset_has_expected_columns(df):
    required = [
        "price_eur", "distance_km", "duration_h", "duration_min",
        "price_per_km", "speed_kmh", "co2_kg", "days_advance",
        "dep_hour", "dep_dow", "dep_month", "is_weekend",
        "distance_range", "route", "n_stops", "has_stops",
        "o_city_name", "d_city_name", "transport_type",
    ]
    for col in required:
        assert col in df.columns, f"Missing expected column: {col}"


def test_build_dataset_row_count_preserved(df, raw):
    assert len(df) == len(raw["tickets"])


def test_price_eur_conversion(df):
    assert df["price_eur"].iloc[0] == pytest.approx(50.0)
    assert df["price_eur"].iloc[1] == pytest.approx(25.0)
    assert df["price_eur"].iloc[2] == pytest.approx(15.0)


def test_duration_is_positive(df):
    assert (df["duration_h"] > 0).all()


def test_distance_is_positive(df):
    assert (df["distance_km"] > 0).all()


def test_co2_is_non_negative(df):
    assert (df["co2_kg"] >= 0).all()


def test_n_stops_row1(df):
    assert df.iloc[1]["n_stops"] == 2


def test_n_stops_row0(df):
    assert df.iloc[0]["n_stops"] == 0


def test_route_label_format(df):
    assert " -> " in df["route"].iloc[0]


def test_distance_range_categories(df):
    valid = {"0-200 km", "201-800 km", "801-2000 km", "2000+ km"}
    actual = set(df["distance_range"].dropna().astype(str).unique())
    assert actual.issubset(valid)


def test_transport_type_merged(df):
    assert set(df["transport_type"].unique()) == {"train", "bus", "carpooling"}
