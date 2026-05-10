"""
tests/test_quality.py
---------------------
Unit tests for tictactrip.quality.
"""

import numpy as np
import pandas as pd
import pytest

from tictactrip.loader import build_dataset
from tictactrip.quality import QualityReport, audit


def _make_df():
    tickets = pd.DataFrame({
        "id":             [1, 2],
        "company":        [10, 30],
        "o_station":      [np.nan, np.nan],
        "d_station":      [np.nan, np.nan],
        "departure_ts":   ["2017-10-13 14:00:00+00", "2017-10-14 10:00:00+00"],
        "arrival_ts":     ["2017-10-13 20:00:00+00", "2017-10-14 14:00:00+00"],
        "price_in_cents": [5000, 2000],
        "search_ts":      ["2017-10-01 00:00:00+00", "2017-10-13 00:00:00+00"],
        "middle_stations":[np.nan, np.nan],
        "other_companies":[np.nan, np.nan],
        "o_city":         [1, 1],
        "d_city":         [2, 2],
    })
    cities = pd.DataFrame({
        "id":          [1, 2],
        "unique_name": ["Paris, France", "Lyon, France"],
        "latitude":    [48.85, 45.76],
        "longitude":   [2.35, 4.83],
        "population":  [2e6, 5e5],
    })
    stations = pd.DataFrame(
        {"id": [], "unique_name": [], "latitude": [], "longitude": []}
    )
    providers = pd.DataFrame({
        "id":                   [10, 30],
        "company_id":           [1, 3],
        "provider_id":          [np.nan, np.nan],
        "name":                 ["tgv", "bbc"],
        "fullname":             ["TGV", "BlaBlaCar"],
        "has_wifi":             [False, False],
        "has_plug":             [True, False],
        "has_adjustable_seats": [True, False],
        "has_bicycle":          [False, False],
        "transport_type":       ["train", "carpooling"],
    })
    return build_dataset(
        {"tickets": tickets, "cities": cities,
         "stations": stations, "providers": providers}
    )


@pytest.fixture
def df():
    return _make_df()


def test_audit_returns_quality_report(df):
    assert isinstance(audit(df), QualityReport)


def test_audit_violations_keys(df):
    report = audit(df)
    assert "Negative duration (arrival before departure)" in report.violations
    assert "Price equals zero" in report.violations


def test_audit_no_negative_durations(df):
    assert audit(df).violations["Negative duration (arrival before departure)"] == 0


def test_audit_outliers_has_price_key(df):
    assert "price_eur" in audit(df).outliers


def test_summary_is_string(df):
    assert isinstance(audit(df).summary(), str)


def test_summary_contains_section_headers(df):
    summary = audit(df).summary()
    assert "Missing values" in summary
    assert "Business-rule violations" in summary
    assert "IQR outliers" in summary
