"""
tests/test_stats.py
-------------------
Unit tests for tictactrip.stats.
"""

import numpy as np
import pandas as pd
import pytest

from tictactrip.loader import build_dataset
from tictactrip.stats import (
    advance_booking_correlation,
    by_distance_range,
    by_transport,
    global_summary,
    price_variability,
    search_sessions,
    top_routes,
    value_for_money,
)


def _make_raw():
    tickets = pd.DataFrame({
        "id":             range(12),
        "company":        [10, 20, 30] * 4,
        "o_station":      [np.nan] * 12,
        "d_station":      [np.nan] * 12,
        "departure_ts":   ["2017-10-13 14:00:00+00"] * 12,
        "arrival_ts":     ["2017-10-13 20:00:00+00"] * 12,
        "price_in_cents": [5000,3000,2000,8000,4000,1500,
                           6000,3500,2500,7000,2000,1000],
        "search_ts":      ["2017-10-01 00:00:00+00"] * 12,
        "middle_stations":[np.nan] * 12,
        "other_companies":[np.nan] * 12,
        "o_city":         [1] * 12,
        "d_city":         [2] * 12,
    })
    cities = pd.DataFrame({
        "id":          [1, 2],
        "unique_name": ["Paris, Ile-de-France", "Lyon, Auvergne"],
        "latitude":    [48.8566, 45.7640],
        "longitude":   [2.3522,  4.8357],
        "population":  [2_161_000, 513_000],
    })
    stations = pd.DataFrame(
        {"id": [], "unique_name": [], "latitude": [], "longitude": []}
    )
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
def df():
    return build_dataset(_make_raw())


def test_global_summary_returns_series(df):
    assert isinstance(global_summary(df), pd.Series)


def test_global_summary_n_tickets(df):
    assert global_summary(df)["n_tickets"] == len(df)


def test_global_summary_price_min_lte_max(df):
    s = global_summary(df)
    assert s["price_min"] <= s["price_max"]


def test_by_transport_returns_dataframe(df):
    assert isinstance(by_transport(df), pd.DataFrame)


def test_by_transport_market_share_sums_to_100(df):
    result = by_transport(df)
    assert result["market_share_pct"].sum() == pytest.approx(100.0, abs=0.2)


def test_by_transport_has_all_modes(df):
    result = by_transport(df)
    assert set(result.index) == {"train", "bus", "carpooling"}


def test_by_distance_range_multiindex(df):
    result = by_distance_range(df)
    assert result.index.names == ["distance_range", "transport_type"]


def test_search_sessions_potential_saving_non_negative(df):
    assert (search_sessions(df)["potential_saving"] >= 0).all()


def test_search_sessions_n_options_positive(df):
    assert (search_sessions(df)["n_options"] >= 1).all()


def test_top_routes_respects_n(df):
    assert len(top_routes(df, n=2, min_tickets=1)) <= 2


def test_top_routes_min_tickets_filter(df):
    assert len(top_routes(df, min_tickets=999)) == 0


def test_price_variability_cv_non_negative(df):
    assert (price_variability(df, min_tickets=1)["price_cv"] >= 0).all()


def test_advance_booking_correlation_pearson_r_range(df):
    result = advance_booking_correlation(df)
    valid  = result["pearson_r"].dropna()
    if not valid.empty:
        assert (valid.between(-1, 1)).all()


def test_value_for_money_score_range(df):
    result = value_for_money(df, min_tickets=1)
    if not result.empty:
        assert result["vfm_score"].between(-1, 1).all()
