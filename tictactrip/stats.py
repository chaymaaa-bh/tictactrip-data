"""
stats.py
--------
Aggregation and statistical analysis functions.
"""

from __future__ import annotations

from typing import List

import pandas as pd
from scipy import stats as scipy_stats


def global_summary(df: pd.DataFrame) -> pd.Series:
    """Return key-performance indicators for the full dataset."""
    return pd.Series({
        "n_tickets":        len(df),
        "n_routes":         df["route"].nunique(),
        "n_providers":      df["company"].nunique(),
        "price_min":        df["price_eur"].min(),
        "price_median":     df["price_eur"].median(),
        "price_mean":       df["price_eur"].mean(),
        "price_max":        df["price_eur"].max(),
        "price_std":        df["price_eur"].std(),
        "duration_min_h":   df["duration_h"].min(),
        "duration_med_h":   df["duration_h"].median(),
        "duration_mean_h":  df["duration_h"].mean(),
        "duration_max_h":   df["duration_h"].max(),
        "distance_min_km":  df["distance_km"].min(),
        "distance_med_km":  df["distance_km"].median(),
        "distance_mean_km": df["distance_km"].mean(),
        "distance_max_km":  df["distance_km"].max(),
        "price_per_km_mean":df["price_per_km"].mean(),
        "speed_mean_kmh":   df["speed_kmh"].mean(),
        "co2_mean_kg":      df["co2_kg"].mean(),
        "co2_total_tonnes": df["co2_kg"].sum() / 1_000,
    }).round(3)


def by_transport(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price, duration, distance, and CO2 stats grouped by transport mode."""
    agg = (
        df.groupby("transport_type")
        .agg(
            n_tickets         = ("id",            "count"),
            price_min         = ("price_eur",     "min"),
            price_median      = ("price_eur",     "median"),
            price_mean        = ("price_eur",     "mean"),
            price_max         = ("price_eur",     "max"),
            price_std         = ("price_eur",     "std"),
            duration_min_h    = ("duration_h",    "min"),
            duration_median_h = ("duration_h",    "median"),
            duration_mean_h   = ("duration_h",    "mean"),
            duration_max_h    = ("duration_h",    "max"),
            distance_mean_km  = ("distance_km",   "mean"),
            price_per_km      = ("price_per_km",  "mean"),
            speed_mean_kmh    = ("speed_kmh",     "mean"),
            co2_mean_kg       = ("co2_kg",        "mean"),
            pct_weekend       = ("is_weekend",    "mean"),
        )
        .round(3)
        .sort_values("n_tickets", ascending=False)
    )
    agg["market_share_pct"] = (agg["n_tickets"] / len(df) * 100).round(1)
    return agg


def by_distance_range(df: pd.DataFrame) -> pd.DataFrame:
    """Compute stats cross-tabulated by distance range and transport mode."""
    return (
        df.groupby(["distance_range", "transport_type"], observed=True)
        .agg(
            n               = ("id",            "count"),
            price_min       = ("price_eur",     "min"),
            price_median    = ("price_eur",     "median"),
            price_mean      = ("price_eur",     "mean"),
            price_max       = ("price_eur",     "max"),
            duration_min_h  = ("duration_h",    "min"),
            duration_mean_h = ("duration_h",    "mean"),
            duration_max_h  = ("duration_h",    "max"),
            price_per_km    = ("price_per_km",  "mean"),
            speed_mean_kmh  = ("speed_kmh",     "mean"),
            co2_mean_kg     = ("co2_kg",        "mean"),
        )
        .round(3)
    )


def advance_booking_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation between advance booking (days) and price,
    per transport type.
    """
    clean = df[df["days_advance"] >= 0]
    rows = []
    for transport, group in clean.groupby("transport_type"):
        if group["days_advance"].nunique() < 2 or group["price_eur"].nunique() < 2:
            r, p = float("nan"), float("nan")
        else:
            r, p = scipy_stats.pearsonr(group["days_advance"], group["price_eur"])
        rows.append({
            "transport_type": transport,
            "pearson_r":      round(r, 4) if r == r else r,
            "p_value":        round(p, 6) if p == p else p,
            "n":              len(group),
            "significant":    bool(p < 0.001) if p == p else False,
        })
    return pd.DataFrame(rows).set_index("transport_type")


def search_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct search sessions and compute potential savings."""
    return (
        df.groupby(["search_ts", "o_city", "d_city"])
        .agg(
            n_options          = ("id",             "count"),
            price_min          = ("price_eur",      "min"),
            price_max          = ("price_eur",      "max"),
            price_mean         = ("price_eur",      "mean"),
            n_transport_modes  = ("transport_type", "nunique"),
        )
        .assign(potential_saving=lambda d: d["price_max"] - d["price_min"])
        .reset_index()
    )


def top_routes(
    df: pd.DataFrame,
    n: int = 20,
    min_tickets: int = 10,
) -> pd.DataFrame:
    """Return the most-served routes with descriptive price and duration stats."""
    return (
        df.groupby("route")
        .agg(
            n_tickets       = ("id",             "count"),
            price_min       = ("price_eur",      "min"),
            price_median    = ("price_eur",      "median"),
            price_mean      = ("price_eur",      "mean"),
            price_max       = ("price_eur",      "max"),
            duration_mean_h = ("duration_h",     "mean"),
            distance_km     = ("distance_km",    "mean"),
            n_modes         = ("transport_type", "nunique"),
            price_cv        = ("price_eur",      lambda x: x.std() / x.mean()),
        )
        .query(f"n_tickets >= {min_tickets}")
        .sort_values("n_tickets", ascending=False)
        .head(n)
        .round(3)
    )


def price_variability(
    df: pd.DataFrame,
    min_tickets: int = 20,
) -> pd.DataFrame:
    """Rank (route, transport) pairs by price coefficient of variation."""
    return (
        df.groupby(["route", "transport_type"])
        .agg(
            n            = ("id",         "count"),
            price_min    = ("price_eur",  "min"),
            price_median = ("price_eur",  "median"),
            price_max    = ("price_eur",  "max"),
            price_cv     = ("price_eur",  lambda x: x.std() / x.mean()),
        )
        .query(f"n >= {min_tickets}")
        .sort_values("price_cv", ascending=False)
        .round(3)
        .reset_index()
    )


def value_for_money(
    df: pd.DataFrame,
    min_tickets: int = 30,
) -> pd.DataFrame:
    """
    Compute a normalised value-for-money score per (route, transport) pair.
    Score = normalised speed - normalised price. Higher is better.
    """
    grouped = (
        df.groupby(["route", "transport_type"])
        .agg(
            n             = ("id",          "count"),
            price_median  = ("price_eur",   "median"),
            speed_mean    = ("speed_kmh",   "mean"),
            distance_mean = ("distance_km", "mean"),
        )
        .query(f"n >= {min_tickets}")
        .reset_index()
        .dropna(subset=["price_median", "speed_mean"])
    )

    def _normalise(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else series * 0

    grouped["vfm_score"] = (
        _normalise(grouped["speed_mean"]) - _normalise(grouped["price_median"])
    )
    return grouped.sort_values("vfm_score", ascending=False).round(3)
