"""
loader.py
---------
Handles loading raw CSV files and joining them into a single
analysis-ready DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

RAW_FILES: Dict[str, str] = {
    "tickets":   "ticket_data.csv",
    "cities":    "cities.csv",
    "stations":  "stations.csv",
    "providers": "providers.csv",
}


def load_raw(data_dir: Path = DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Load the four raw CSV files into a dictionary of DataFrames."""
    data_dir = Path(data_dir)
    raw: Dict[str, pd.DataFrame] = {}
    for key, filename in RAW_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected data file not found: {path}\n"
                "Place the four CSV files in the data/ directory."
            )
        raw[key] = pd.read_csv(path)
    return raw


def build_dataset(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the four raw tables and compute all derived features."""
    tickets   = raw["tickets"].copy()
    cities    = raw["cities"]
    providers = raw["providers"]

    origin_cities = (
        cities[["id", "unique_name", "latitude", "longitude", "population"]]
        .rename(columns={
            "id": "o_city", "unique_name": "o_city_name",
            "latitude": "o_lat", "longitude": "o_lon", "population": "o_pop",
        })
    )
    tickets = tickets.merge(origin_cities, on="o_city", how="left")

    dest_cities = (
        cities[["id", "unique_name", "latitude", "longitude", "population"]]
        .rename(columns={
            "id": "d_city", "unique_name": "d_city_name",
            "latitude": "d_lat", "longitude": "d_lon", "population": "d_pop",
        })
    )
    tickets = tickets.merge(dest_cities, on="d_city", how="left")

    provider_info = (
        providers[["id", "transport_type", "name", "fullname",
                   "has_wifi", "has_plug", "has_bicycle", "has_adjustable_seats"]]
        .rename(columns={
            "id": "company", "name": "provider_name", "fullname": "provider_fullname",
        })
    )
    tickets = tickets.merge(provider_info, on="company", how="left")

    for col in ("departure_ts", "arrival_ts", "search_ts"):
        tickets[col] = pd.to_datetime(tickets[col], utc=True, format="mixed")

    tickets = _add_geo_features(tickets)
    tickets = _add_time_features(tickets)
    tickets = _add_business_features(tickets)

    return tickets


def _add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    from haversine import haversine
    df["distance_km"] = df.apply(
        lambda r: haversine((r["o_lat"], r["o_lon"]), (r["d_lat"], r["d_lon"])),
        axis=1,
    )
    df["n_stops"] = df["middle_stations"].apply(_parse_stop_count)
    short = lambda s: str(s).split(",")[0].strip() if pd.notna(s) else "?"
    df["route"] = df["o_city_name"].apply(short) + " -> " + df["d_city_name"].apply(short)
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["duration_min"]  = (df["arrival_ts"] - df["departure_ts"]).dt.total_seconds() / 60
    df["duration_h"]    = df["duration_min"] / 60
    df["days_advance"]  = (df["departure_ts"] - df["search_ts"]).dt.total_seconds() / 86_400
    df["dep_hour"]      = df["departure_ts"].dt.hour
    df["dep_dow"]       = df["departure_ts"].dt.dayofweek
    df["dep_dow_name"]  = df["departure_ts"].dt.day_name()
    df["dep_month"]     = df["departure_ts"].dt.month
    df["dep_week"]      = df["departure_ts"].dt.isocalendar().week.astype(int)
    df["dep_date"]      = df["departure_ts"].dt.normalize()
    df["is_weekend"]    = df["dep_dow"] >= 5
    return df


def _add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    df["price_eur"]    = df["price_in_cents"] / 100
    df["price_per_km"] = df["price_eur"] / df["distance_km"]
    df["speed_kmh"]    = df["distance_km"] / df["duration_h"].replace(0, float("nan"))
    co2_factors = {"train": 0.006, "bus": 0.029, "carpooling": 0.028, "car": 0.160}
    df["co2_kg"] = df.apply(
        lambda r: r["distance_km"] * co2_factors.get(r["transport_type"], 0.05),
        axis=1,
    )
    bins   = [0, 200, 800, 2000, float("inf")]
    labels = ["0-200 km", "201-800 km", "801-2000 km", "2000+ km"]
    df["distance_range"] = pd.cut(df["distance_km"], bins=bins, labels=labels)
    df["has_stops"] = df["n_stops"] > 0
    return df


def _parse_stop_count(value: object) -> int:
    import re
    if pd.isna(value) or str(value).strip() in ("", "{}"):
        return 0
    return len(re.findall(r"\d+", str(value)))
