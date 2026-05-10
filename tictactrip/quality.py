"""
quality.py
----------
Data-quality audit utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class QualityReport:
    """Container for all quality-check results."""

    missing:    pd.DataFrame = field(default_factory=pd.DataFrame)
    violations: Dict[str, int] = field(default_factory=dict)
    outliers:   Dict[str, int] = field(default_factory=dict)
    notes:      List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["=" * 60, "DATA QUALITY REPORT", "=" * 60]

        lines.append("\n-- Missing values --")
        if self.missing.empty:
            lines.append("  None.")
        else:
            for col, row in self.missing.iterrows():
                bar = "#" * int(row["pct"] / 5)
                lines.append(f"  {col:<30} {row['count']:>7,}  ({row['pct']:5.1f}%)  {bar}")

        lines.append("\n-- Business-rule violations --")
        for label, count in self.violations.items():
            flag = "WARN" if count > 0 else "  OK"
            lines.append(f"  [{flag}]  {label:<48}  {count:>6,}")

        lines.append("\n-- IQR outliers --")
        for col, n in self.outliers.items():
            lines.append(f"  {col:<25}  {n:>6,}")

        lines.append("\n-- Notes --")
        for note in self.notes:
            lines.append(f"  {note}")

        return "\n".join(lines)


def audit(df: pd.DataFrame) -> QualityReport:
    """Run all quality checks on an enriched DataFrame."""
    report = QualityReport()
    report.missing    = _check_missing(df)
    report.violations = _check_violations(df)
    report.outliers   = _check_outliers(df)
    report.notes      = _generate_notes(df, report)
    return report


def _check_missing(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isnull().sum()
    pct    = counts / len(df) * 100
    return (
        pd.DataFrame({"count": counts, "pct": pct.round(2)})
        .loc[lambda d: d["count"] > 0]
        .sort_values("pct", ascending=False)
    )


def _check_violations(df: pd.DataFrame) -> Dict[str, int]:
    return {
        "Negative duration (arrival before departure)": int((df["duration_min"] < 0).sum()),
        "Duration over 48 h":                           int((df["duration_h"] > 48).sum()),
        "Duration over 24 h (investigate)":             int((df["duration_h"] > 24).sum()),
        "Price equals zero":                            int((df["price_eur"] == 0).sum()),
        "Price over 300 EUR":                           int((df["price_eur"] > 300).sum()),
        "Speed over 350 km/h (physically impossible)":  int((df["speed_kmh"] > 350).sum()),
        "Speed under 10 km/h (suspiciously slow)":      int((df["speed_kmh"] < 10).sum()),
        "Negative advance booking (search after depart)":
                                                        int((df["days_advance"] < 0).sum()),
        "Advance booking over 365 days":                int((df["days_advance"] > 365).sum()),
        "Distance under 5 km (likely same city)":       int((df["distance_km"] < 5).sum()),
    }


def _check_outliers(df: pd.DataFrame) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for col in ("price_eur", "duration_h", "distance_km"):
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr    = q3 - q1
        mask   = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        result[col] = int(mask.sum())
    return result


def _generate_notes(df: pd.DataFrame, report: QualityReport) -> List[str]:
    notes: List[str] = []
    missing_stations = int(df["o_station"].isna().sum())
    n_carpooling     = int((df["transport_type"] == "carpooling").sum())
    match_pct        = missing_stations / max(n_carpooling, 1) * 100
    notes.append(
        f"Missing station IDs ({missing_stations:,}) align with carpooling rows "
        f"({n_carpooling:,}) at {match_pct:.0f}% -- expected, no station for rideshare."
    )
    long_trips = report.violations.get("Duration over 24 h (investigate)", 0)
    if long_trips > 0:
        notes.append(
            f"{long_trips:,} trips exceed 24 h. Likely multi-leg journeys or "
            "timestamp aggregation artefacts. Kept in dataset but flagged."
        )
    negative_advance = report.violations.get(
        "Negative advance booking (search after depart)", 0
    )
    if negative_advance > 0:
        notes.append(
            f"{negative_advance:,} rows have search_ts > departure_ts. "
            "Probably test or import records; excluded from advance-booking analysis."
        )
    return notes
