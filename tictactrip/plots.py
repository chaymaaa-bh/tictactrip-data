"""
plots.py
--------
Reusable matplotlib/seaborn plotting functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TRANSPORT_COLORS = {
    "train":      "#4361ee",
    "bus":        "#f77f00",
    "carpooling": "#06d6a0",
}

DARK_BG    = "#161b22"
PANEL_BG   = "#0d1117"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#e6edf3"
MUTED      = "#8b949e"


def apply_theme() -> None:
    """Apply the dark-mode global matplotlib style used throughout the project."""
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        "figure.dpi":       130,
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   PANEL_BG,
        "axes.edgecolor":   "#30363d",
        "axes.labelcolor":  TEXT_COLOR,
        "xtick.color":      MUTED,
        "ytick.color":      MUTED,
        "text.color":       TEXT_COLOR,
        "grid.color":       GRID_COLOR,
        "grid.linewidth":   0.7,
        "legend.facecolor": DARK_BG,
        "legend.edgecolor": "#30363d",
    })


def _save(fig: plt.Figure, path: Optional[Path]) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", facecolor=DARK_BG)


def _tc(name: str) -> str:
    return TRANSPORT_COLORS.get(name, MUTED)


def plot_distributions(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Three-panel histogram: price, duration, and distance."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Distribution of key variables", fontsize=14, fontweight="bold")

    configs = [
        ("price_eur",   "Price (EUR)",      300,  "#e94560"),
        ("duration_h",  "Duration (hours)",  50,  TRANSPORT_COLORS["train"]),
        ("distance_km", "Distance (km)",   2000,  TRANSPORT_COLORS["carpooling"]),
    ]
    for ax, (col, label, clip_val, color) in zip(axes, configs):
        data   = df[col].clip(upper=clip_val)
        median = df[col].median()
        mean   = df[col].mean()
        ax.hist(data, bins=80, color=color, alpha=0.85, edgecolor="none")
        ax.axvline(median, color="white",   linestyle="--", linewidth=1.5,
                   label=f"Median: {median:.1f}")
        ax.axvline(mean,   color="#ffd166", linestyle=":",  linewidth=1.5,
                   label=f"Mean:   {mean:.1f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution -- {label}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_boxplots_by_transport(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Side-by-side boxplots of price, duration, and price-per-km by transport mode."""
    order  = ["train", "bus", "carpooling"]
    colors = [_tc(t) for t in order]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Distribution by transport mode", fontsize=14, fontweight="bold")

    metrics = [
        ("price_eur",    "Price (EUR)",      300),
        ("duration_h",   "Duration (h)",      30),
        ("price_per_km", "Price / km (EUR)", 0.5),
    ]
    for ax, (col, ylabel, ylim) in zip(axes, metrics):
        data = [
            df.loc[df["transport_type"] == t, col].clip(upper=ylim * 5).dropna()
            for t in order
        ]
        bp = ax.boxplot(
            data, patch_artist=True, notch=True, labels=order,
            medianprops=dict(color="white",  linewidth=2.5),
            whiskerprops=dict(color=MUTED),
            capprops=dict(color=MUTED),
            flierprops=dict(marker="o", markerfacecolor=MUTED, markersize=2, alpha=0.4),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_ylim(0, ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_transport_overview(
    transport_stats: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Three bar charts: mean price, mean duration, CO2 per transport mode."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Transport mode comparison", fontsize=14, fontweight="bold")

    metrics = [
        ("price_mean",      "Mean price (EUR)"),
        ("duration_mean_h", "Mean duration (h)"),
        ("co2_mean_kg",     "Mean CO2 (kg)"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        labels = transport_stats.index.tolist()
        values = transport_stats[col].tolist()
        colors = [_tc(t) for t in labels]
        bars = ax.bar(labels, values, color=colors, edgecolor=DARK_BG, linewidth=1)
        ax.bar_label(bars, fmt="%.2f", padding=4, color=TEXT_COLOR, fontsize=10)
        ax.set_title(title)
        ax.set_ylabel(title)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_radar(
    transport_stats: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Polar radar chart comparing normalised profiles of each transport mode."""
    categories = ["Mean price", "Mean duration", "Mean speed", "Price/km x100", "CO2 (kg)"]
    n_cats = len(categories)

    radar_df = transport_stats[[
        "price_mean", "duration_mean_h", "speed_mean_kmh", "price_per_km", "co2_mean_kg"
    ]].copy()
    radar_df["price_per_km"] = radar_df["price_per_km"] * 100

    for col in radar_df.columns:
        mn, mx = radar_df[col].min(), radar_df[col].max()
        radar_df[col] = (radar_df[col] - mn) / (mx - mn + 1e-9)

    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    for transport in radar_df.index:
        color  = _tc(transport)
        values = radar_df.loc[transport].tolist() + [radar_df.loc[transport].iloc[0]]
        ax.plot(angles, values, color=color, linewidth=2.5, label=transport)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10, color=TEXT_COLOR)
    ax.set_yticklabels([])
    ax.spines["polar"].set_color("#30363d")
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title(
        "Normalised transport profile (0=min, 1=max per metric)",
        pad=20, fontsize=12, color=TEXT_COLOR, fontweight="bold",
    )

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_distance_heatmap(
    dist_stats: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Heatmap of mean price per km cross-tabulated by distance range and transport mode."""
    pivot_ppkm = dist_stats["price_per_km"].unstack("transport_type").round(3)
    pivot_med  = dist_stats["price_median"].unstack("transport_type")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Price analysis by distance range", fontsize=14, fontweight="bold")

    sns.heatmap(
        pivot_ppkm, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, linecolor="#30363d", ax=axes[0],
        cbar_kws={"label": "EUR/km"},
    )
    axes[0].set_title("Mean price per km\ndarker = more expensive")
    axes[0].set_ylabel("Distance range")
    axes[0].set_xlabel("Transport mode")

    x       = np.arange(len(pivot_med))
    n_modes = len(pivot_med.columns)
    width   = 0.8 / n_modes
    for i, transport in enumerate(pivot_med.columns):
        axes[1].bar(
            x + i * width, pivot_med[transport].fillna(0), width,
            label=transport, color=_tc(transport), edgecolor=DARK_BG, alpha=0.9,
        )
    axes[1].set_xticks(x + width * (n_modes - 1) / 2)
    axes[1].set_xticklabels(pivot_med.index, rotation=15)
    axes[1].set_ylabel("Median price (EUR)")
    axes[1].set_title("Median price by range and transport")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_temporal_overview(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Six-panel temporal analysis."""
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_short = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Temporal analysis", fontsize=15, fontweight="bold", y=1.01)

    hourly = df.groupby("dep_hour")["id"].count()
    axes[0, 0].bar(hourly.index, hourly.values, color=TRANSPORT_COLORS["bus"], alpha=0.9)
    axes[0, 0].set_title("Departure volume by hour")
    axes[0, 0].set_xlabel("Hour of departure")
    axes[0, 0].set_ylabel("Ticket count")
    axes[0, 0].set_xticks(range(0, 24, 2))

    for t in ["train", "bus", "carpooling"]:
        sub = df[df["transport_type"] == t].groupby("dep_hour")["price_eur"].mean()
        axes[0, 1].plot(sub.index, sub.values, marker="o", markersize=3,
                        linewidth=2, color=_tc(t), label=t)
    axes[0, 1].set_title("Mean price by departure hour")
    axes[0, 1].set_xlabel("Hour")
    axes[0, 1].set_ylabel("Mean price (EUR)")
    axes[0, 1].legend()
    axes[0, 1].set_xticks(range(0, 24, 2))

    daily = df.groupby("dep_dow_name")["price_eur"].mean().reindex(dow_order)
    bar_colors = [
        TRANSPORT_COLORS["carpooling"] if d in ("Saturday","Sunday")
        else TRANSPORT_COLORS["train"] for d in dow_order
    ]
    axes[0, 2].bar(dow_short, daily.values, color=bar_colors)
    axes[0, 2].set_title("Mean price by day of week")
    axes[0, 2].set_ylabel("Mean price (EUR)")

    df_copy = df.copy()
    df_copy["dep_week_label"] = df_copy["departure_ts"].dt.to_period("W").dt.start_time
    weekly = df_copy.groupby("dep_week_label")["id"].count()
    axes[1, 0].plot(weekly.index, weekly.values,
                    color=TRANSPORT_COLORS["train"], linewidth=2, marker="o", markersize=4)
    axes[1, 0].fill_between(weekly.index, weekly.values,
                             alpha=0.2, color=TRANSPORT_COLORS["train"])
    axes[1, 0].set_title("Ticket volume by departure week")
    axes[1, 0].set_xlabel("Week")
    axes[1, 0].set_ylabel("Ticket count")
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[1, 0].tick_params(axis="x", rotation=30)

    adv = df[(df["days_advance"] >= 0) & (df["days_advance"] <= 90)].copy()
    adv["adv_bin"] = pd.cut(adv["days_advance"], bins=18)
    adv_curve = adv.groupby("adv_bin", observed=True)["price_eur"].median()
    mids = [b.mid for b in adv_curve.index]
    axes[1, 1].plot(mids, adv_curve.values, color=TRANSPORT_COLORS["bus"],
                    linewidth=2.5, marker="o", markersize=5)
    axes[1, 1].axvline(7,  color="#e94560", linestyle="--", alpha=0.7, linewidth=1.5, label="7 days")
    axes[1, 1].axvline(30, color="#ffd166", linestyle="--", alpha=0.7, linewidth=1.5, label="30 days")
    axes[1, 1].set_title("Median price vs advance booking")
    axes[1, 1].set_xlabel("Days before departure")
    axes[1, 1].set_ylabel("Median price (EUR)")
    axes[1, 1].legend()

    for t in ["train", "bus", "carpooling"]:
        sub = df[(df["transport_type"] == t) &
                 (df["days_advance"] >= 0) & (df["days_advance"] <= 90)]
        axes[1, 2].hist(sub["days_advance"], bins=30, alpha=0.5,
                        color=_tc(t), label=t, density=True)
    axes[1, 2].set_title("Advance booking distribution by transport")
    axes[1, 2].set_xlabel("Days before departure")
    axes[1, 2].set_ylabel("Density")
    axes[1, 2].legend()

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_dow_hour_heatmap(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Day-of-week x hour-of-day heatmaps showing median price per transport mode."""
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_short = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Median price by day of week and hour of departure",
                 fontsize=14, fontweight="bold")

    for ax, transport in zip(axes, ["train", "bus", "carpooling"]):
        sub = df[df["transport_type"] == transport].copy()
        sub["dow_name"] = sub["departure_ts"].dt.day_name()
        pivot = (
            sub.groupby(["dow_name", "dep_hour"])["price_eur"]
            .median()
            .unstack("dep_hour")
            .reindex(dow_order)
        )
        pivot.index = dow_short

        sns.heatmap(
            pivot, ax=ax, cmap="RdYlGn_r", annot=False,
            linewidths=0.3, linecolor=DARK_BG,
            cbar_kws={"label": "Median price (EUR)", "shrink": 0.7},
        )

        if not pivot.empty and not pivot.isnull().all().all():
            best_row, best_col = np.unravel_index(
                np.nanargmin(pivot.values), pivot.shape
            )
            ax.add_patch(plt.Rectangle(
                (best_col, best_row), 1, 1,
                fill=False, edgecolor="white", lw=3
            ))
            best_price = pivot.values[best_row, best_col]
            ax.set_title(
                f"{transport.capitalize()}\n"
                f"Best slot: {dow_short[best_row]} {int(pivot.columns[best_col])}h "
                f"({best_price:.0f} EUR)"
            )
        else:
            ax.set_title(transport.capitalize())

        ax.set_xlabel("Hour of departure")
        ax.set_ylabel("Day of week" if transport == "train" else "")

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_sessions(
    sessions: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Three-panel chart for search-session analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Search session analysis -- potential savings",
                 fontsize=14, fontweight="bold")

    med  = sessions["potential_saving"].median()
    mean = sessions["potential_saving"].mean()
    axes[0].hist(sessions["potential_saving"].clip(upper=100), bins=60,
                 color=TRANSPORT_COLORS["carpooling"], alpha=0.85, edgecolor="none")
    axes[0].axvline(med,  color="white",   linestyle="--", linewidth=2,
                    label=f"Median: {med:.0f} EUR")
    axes[0].axvline(mean, color="#ffd166", linestyle=":",  linewidth=2,
                    label=f"Mean:   {mean:.0f} EUR")
    axes[0].set_xlabel("Potential saving (EUR)")
    axes[0].set_ylabel("Session count")
    axes[0].set_title("Distribution of potential savings")
    axes[0].legend()

    vc = sessions["n_options"].value_counts().sort_index().head(15)
    axes[1].bar(vc.index, vc.values, color=TRANSPORT_COLORS["train"], alpha=0.85)
    axes[1].set_xlabel("Number of options returned")
    axes[1].set_ylabel("Session count")
    axes[1].set_title("Options per search session")

    avg_saving = sessions.groupby("n_options")["potential_saving"].mean().head(15)
    axes[2].plot(avg_saving.index, avg_saving.values,
                 marker="o", color=TRANSPORT_COLORS["bus"], linewidth=2.5, markersize=6)
    axes[2].fill_between(avg_saving.index, avg_saving.values,
                         alpha=0.2, color=TRANSPORT_COLORS["bus"])
    axes[2].set_xlabel("Number of options in session")
    axes[2].set_ylabel("Mean potential saving (EUR)")
    axes[2].set_title("More options -> more savings?")

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_network(
    df: pd.DataFrame,
    min_tickets: int = 50,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Spring-layout network graph of city connections."""
    import networkx as nx

    edges = (
        df.groupby(["o_city_name", "d_city_name"])
        .agg(weight=("id", "count"))
        .reset_index()
        .query(f"weight >= {min_tickets}")
    )
    short = lambda s: str(s).split(",")[0].strip()
    edges["o"] = edges["o_city_name"].apply(short)
    edges["d"] = edges["d_city_name"].apply(short)

    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row["o"], row["d"], weight=row["weight"])

    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    pos         = nx.spring_layout(G, k=2.5, seed=42, weight="weight")

    node_sizes  = [betweenness.get(n, 0.001) * 15_000 + 50 for n in G.nodes()]
    node_colors = [betweenness.get(n, 0) for n in G.nodes()]
    max_w       = max(d["weight"] for _, _, d in G.edges(data=True))
    edge_widths = [d["weight"] / max_w * 3 for _, _, d in G.edges(data=True)]

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(PANEL_BG)
    ax.set_facecolor(PANEL_BG)

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.25,
                           edge_color=TRANSPORT_COLORS["train"], arrows=False)
    sc = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                                 node_color=node_colors, cmap="plasma", alpha=0.9)

    top_nodes = set(sorted(betweenness, key=betweenness.get, reverse=True)[:10])
    nx.draw_networkx_labels(G, pos, labels={n: n for n in top_nodes}, ax=ax,
                             font_size=8, font_color="white", font_weight="bold")

    plt.colorbar(sc, ax=ax, label="Betweenness centrality", shrink=0.6)
    ax.set_title(
        f"City network -- {G.number_of_nodes()} cities, {G.number_of_edges()} routes\n"
        f"(Threshold: >= {min_tickets} tickets | Node size proportional to centrality)",
        fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=15,
    )
    ax.axis("off")

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_ml_results(
    y_test: pd.Series,
    y_pred: np.ndarray,
    feature_names: list,
    feature_importances: np.ndarray,
    mae: float,
    r2: float,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Two-panel figure: feature importance bars and predicted-vs-actual scatter."""
    fi_series = pd.Series(feature_importances, index=feature_names).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("XGBoost -- Model performance", fontsize=14, fontweight="bold")

    colors = [
        TRANSPORT_COLORS["carpooling"] if v == fi_series.max()
        else TRANSPORT_COLORS["train"]
        for v in fi_series
    ]
    axes[0].barh(fi_series.index, fi_series.values, color=colors, edgecolor="none", alpha=0.9)
    axes[0].set_title("Feature importance (gain)")
    axes[0].set_xlabel("Importance")

    axes[1].scatter(y_test, y_pred, alpha=0.25, s=8,
                    color=TRANSPORT_COLORS["bus"], rasterized=True)
    lim = [0, max(float(y_test.max()), float(y_pred.max())) * 1.05]
    axes[1].plot(lim, lim, "r--", linewidth=2, label="Perfect prediction")
    axes[1].set_xlim(lim)
    axes[1].set_ylim(lim)
    axes[1].set_xlabel("Actual price (EUR)")
    axes[1].set_ylabel("Predicted price (EUR)")
    axes[1].set_title(f"Predicted vs Actual\nMAE = {mae:.2f} EUR  |  R2 = {r2:.3f}")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, save_path)
    return fig
