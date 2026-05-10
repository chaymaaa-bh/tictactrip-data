"""
maps.py
-------
Interactive map generation using Folium.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import folium
    from folium.plugins import HeatMap
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False


def _require_folium() -> None:
    if not _FOLIUM_AVAILABLE:
        raise ImportError("Install folium:  pip install folium")


def origin_heatmap(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> "folium.Map":
    """Heatmap of departure city density across Europe."""
    _require_folium()

    origins = (
        df.groupby(["o_city_name", "o_lat", "o_lon"])
        .size()
        .reset_index(name="count")
        .dropna(subset=["o_lat", "o_lon"])
    )

    m = folium.Map(location=[47.5, 10], zoom_start=5, tiles="CartoDB dark_matter")
    heat_data = origins[["o_lat", "o_lon", "count"]].values.tolist()
    HeatMap(heat_data, min_opacity=0.3, radius=18, blur=15, max_zoom=8).add_to(m)

    if save_path is not None:
        m.save(str(save_path))
    return m


def top_routes_map(
    df: pd.DataFrame,
    top_n: int = 80,
    save_path: Optional[Path] = None,
) -> "folium.Map":
    """Map of the most-served routes as polylines coloured by transport mode."""
    _require_folium()

    COLOR_MAP = {
        "train":      "#4361ee",
        "bus":        "#f77f00",
        "carpooling": "#06d6a0",
    }

    routes = (
        df.groupby(
            ["o_city_name", "d_city_name",
             "o_lat", "o_lon", "d_lat", "d_lon",
             "transport_type"],
            observed=True,
        )
        .agg(
            count      = ("id",         "count"),
            price_mean = ("price_eur",  "mean"),
            dur_mean   = ("duration_h", "mean"),
        )
        .reset_index()
        .dropna(subset=["o_lat", "o_lon", "d_lat", "d_lon"])
        .sort_values("count", ascending=False)
        .head(top_n)
    )

    m = folium.Map(location=[47.5, 10], zoom_start=5, tiles="CartoDB dark_matter")

    for _, row in routes.iterrows():
        weight  = max(1, min(8, row["count"] / 100))
        color   = COLOR_MAP.get(row["transport_type"], "#888888")
        o_short = str(row["o_city_name"]).split(",")[0]
        d_short = str(row["d_city_name"]).split(",")[0]
        tooltip = (
            f"<b>{o_short} -> {d_short}</b><br>"
            f"Mode: {row['transport_type']}<br>"
            f"Tickets: {row['count']:,}<br>"
            f"Mean price: {row['price_mean']:.1f} EUR<br>"
            f"Mean duration: {row['dur_mean']:.1f} h"
        )
        folium.PolyLine(
            locations=[[row["o_lat"], row["o_lon"]], [row["d_lat"], row["d_lon"]]],
            weight=weight, color=color, opacity=0.6,
            tooltip=folium.Tooltip(tooltip),
        ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:rgba(13,17,23,0.9);padding:12px 18px;
                border-radius:8px;border:1px solid #30363d;
                color:white;font-size:13px;">
      <b>Transport mode</b><br>
      <span style="color:#4361ee;">---</span> Train<br>
      <span style="color:#f77f00;">---</span> Bus<br>
      <span style="color:#06d6a0;">---</span> Carpooling<br>
      <i style="font-size:11px;">Line thickness proportional to volume</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    if save_path is not None:
        m.save(str(save_path))
    return m
