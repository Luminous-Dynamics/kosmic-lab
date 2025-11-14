#!/usr/bin/env python3
"""
🌊 Kosmic Lab Real-Time Dashboard

Live visualization of K-Index across all experimental tracks.
Automatically refreshes every 5 seconds to show latest results.

Usage:
    python scripts/kosmic_dashboard.py --port 8050

Then open: http://localhost:8050
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "🌊 Kosmic Lab Dashboard"

# Global data cache
data_cache = {"track_b": None, "track_c": None, "track_d": None, "last_update": None}


def load_track_b_data():
    """Load Track B (SAC Controller) data."""
    try:
        csv_path = Path("paper2_analyses/results/C_functional_results.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Filter Track B episodes
            track_b = df[df["episode_id"].str.contains("ep0", na=False)].copy()
            # Rename avg_k to K for consistency
            if "avg_k" in track_b.columns:
                track_b["K"] = track_b["avg_k"]
            return track_b
    except Exception as e:
        print(f"Error loading Track B: {e}")
    return pd.DataFrame()


def load_track_c_data():
    """Load Track C (Bioelectric) data."""
    try:
        csv_path = Path("paper2_analyses/results/C_functional_results.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Filter Track C episodes
            track_c = df[df["episode_id"].str.contains("track_c", na=False)].copy()
            # Rename avg_k to K for consistency
            if "avg_k" in track_c.columns:
                track_c["K"] = track_c["avg_k"]
            return track_c
    except Exception as e:
        print(f"Error loading Track C: {e}")
    return pd.DataFrame()


def load_track_d_data():
    """Load Track D (Multi-Agent) data."""
    try:
        # Try to load parameter sweep results first
        sweep_files = list(Path("logs/track_d/parameter_sweep").glob("parameter_sweep_*.csv"))
        if sweep_files:
            # Load most recent sweep
            latest = sorted(sweep_files)[-1]
            df = pd.read_csv(latest)
            # Create a simplified view for dashboard (using ring topology as representative)
            ring_data = (
                df[df["network_topology"] == "ring"].sort_values("communication_cost").copy()
            )
            ring_data["episode"] = range(len(ring_data))
            ring_data["K"] = ring_data["mean_collective_k"]
            ring_data["individual_k"] = ring_data["mean_individual_k"]
            return ring_data

        # Fallback to NPZ files if no sweep data
        npz_files = list(Path("logs/track_d").glob("track_d_*.npz"))
        if npz_files:
            # Load most recent file
            latest = sorted(npz_files)[-1]
            data = np.load(latest)
            df = pd.DataFrame(
                {
                    "episode": np.arange(len(data["collective_k"])),
                    "K": data["collective_k"],
                    "individual_k": data["mean_individual_k"],
                    "reward": data["mean_reward"],
                }
            )
            return df
    except Exception as e:
        print(f"Error loading Track D: {e}")
    return pd.DataFrame()


def update_data_cache():
    """Update all data from disk."""
    data_cache["track_b"] = load_track_b_data()
    data_cache["track_c"] = load_track_c_data()
    data_cache["track_d"] = load_track_d_data()
    data_cache["last_update"] = datetime.now().strftime("%H:%M:%S")


# Layout
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "🌊 Kosmic Lab Real-Time Dashboard",
                    style={"textAlign": "center", "color": "#2c3e50"},
                ),
                html.P(
                    "Live K-Index Monitoring Across All Experimental Tracks",
                    style={"textAlign": "center", "color": "#7f8c8d", "fontSize": 16},
                ),
            ],
            style={"backgroundColor": "#ecf0f1", "padding": "20px", "marginBottom": "20px"},
        ),
        # Auto-refresh interval (5 seconds)
        dcc.Interval(id="interval-component", interval=5000, n_intervals=0),
        # Last update time
        html.Div(
            id="last-update-time",
            style={"textAlign": "right", "marginRight": "20px", "color": "#95a5a6"},
        ),
        # Track selector
        html.Div(
            [
                html.Label("Select Track:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="track-selector",
                    options=[
                        {"label": "🎮 Track B: SAC Controller (56 episodes)", "value": "track_b"},
                        {
                            "label": "⚡ Track C: Bioelectric Rescue (20 episodes)",
                            "value": "track_c",
                        },
                        {
                            "label": "🤝 Track D: Multi-Agent Coordination (Ring Topology)",
                            "value": "track_d",
                        },
                        {"label": "📊 All Tracks Combined", "value": "all"},
                    ],
                    value="all",
                    style={"width": "500px"},
                ),
            ],
            style={"margin": "20px", "display": "flex", "alignItems": "center"},
        ),
        # Main graphs
        html.Div(
            [
                # K-Index Distribution
                html.Div(
                    [dcc.Graph(id="k-distribution")],
                    style={"width": "48%", "display": "inline-block", "padding": "10px"},
                ),
                # K-Index Time Series
                html.Div(
                    [dcc.Graph(id="k-timeseries")],
                    style={"width": "48%", "display": "inline-block", "padding": "10px"},
                ),
            ]
        ),
        html.Div(
            [
                # ℂ Functional Components
                html.Div(
                    [dcc.Graph(id="c-functional")],
                    style={"width": "48%", "display": "inline-block", "padding": "10px"},
                ),
                # Statistics Table
                html.Div(
                    [
                        html.H3("📊 Summary Statistics", style={"textAlign": "center"}),
                        html.Div(id="statistics-table"),
                    ],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "padding": "10px",
                        "verticalAlign": "top",
                    },
                ),
            ]
        ),
        # Footer
        html.Div(
            [
                html.Hr(),
                html.P(
                    "🌊 Kosmic Lab - Revolutionary AI-Accelerated Consciousness Research Platform",
                    style={"textAlign": "center", "color": "#95a5a6", "fontSize": 12},
                ),
            ],
            style={"marginTop": "40px"},
        ),
    ],
    style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#f8f9fa"},
)


@app.callback(
    [
        Output("k-distribution", "figure"),
        Output("k-timeseries", "figure"),
        Output("c-functional", "figure"),
        Output("statistics-table", "children"),
        Output("last-update-time", "children"),
    ],
    [Input("interval-component", "n_intervals"), Input("track-selector", "value")],
)
def update_graphs(n, selected_track):
    """Update all graphs with latest data."""
    # Refresh data from disk
    update_data_cache()

    # Prepare data based on selection
    if selected_track == "track_b":
        data = data_cache["track_b"]
        title_suffix = "Track B (SAC Controller)"
        color = "#3498db"
    elif selected_track == "track_c":
        data = data_cache["track_c"]
        title_suffix = "Track C (Bioelectric)"
        color = "#e74c3c"
    elif selected_track == "track_d":
        data = data_cache["track_d"]
        title_suffix = "Track D (Multi-Agent)"
        color = "#2ecc71"
    else:  # all tracks
        data = None
        title_suffix = "All Tracks"
        color = None

    # K-Index Distribution
    k_dist = go.Figure()

    if selected_track == "all":
        # Combine all tracks
        if not data_cache["track_b"].empty:
            k_dist.add_trace(
                go.Histogram(
                    x=data_cache["track_b"]["K"],
                    name="Track B",
                    opacity=0.7,
                    marker_color="#3498db",
                    nbinsx=30,
                )
            )
        if not data_cache["track_c"].empty:
            k_dist.add_trace(
                go.Histogram(
                    x=data_cache["track_c"]["K"],
                    name="Track C",
                    opacity=0.7,
                    marker_color="#e74c3c",
                    nbinsx=30,
                )
            )
        if not data_cache["track_d"].empty:
            k_dist.add_trace(
                go.Histogram(
                    x=data_cache["track_d"]["K"],
                    name="Track D",
                    opacity=0.7,
                    marker_color="#2ecc71",
                    nbinsx=30,
                )
            )
        k_dist.update_layout(barmode="overlay")
    else:
        if data is not None and not data.empty and "K" in data.columns:
            k_dist.add_trace(go.Histogram(x=data["K"], nbinsx=40, marker_color=color, opacity=0.8))

    k_dist.add_vline(
        x=1.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Corridor Threshold (K=1.5)",
        annotation_position="top right",
    )
    k_dist.update_layout(
        title=f"K-Index Distribution - {title_suffix}",
        xaxis_title="K-Index",
        yaxis_title="Count",
        template="plotly_white",
        height=400,
    )

    # K-Index Time Series
    k_ts = go.Figure()

    if selected_track == "all":
        if not data_cache["track_b"].empty:
            k_ts.add_trace(
                go.Scatter(
                    y=data_cache["track_b"]["K"],
                    mode="lines+markers",
                    name="Track B",
                    marker_color="#3498db",
                    line=dict(width=2),
                )
            )
        if not data_cache["track_c"].empty:
            k_ts.add_trace(
                go.Scatter(
                    y=data_cache["track_c"]["K"],
                    mode="lines+markers",
                    name="Track C",
                    marker_color="#e74c3c",
                    line=dict(width=2),
                )
            )
        if not data_cache["track_d"].empty:
            k_ts.add_trace(
                go.Scatter(
                    y=data_cache["track_d"]["K"],
                    mode="lines+markers",
                    name="Track D",
                    marker_color="#2ecc71",
                    line=dict(width=2),
                )
            )
    else:
        if data is not None and not data.empty and "K" in data.columns:
            k_ts.add_trace(
                go.Scatter(
                    y=data["K"],
                    mode="lines+markers",
                    name="K-Index",
                    marker_color=color,
                    line=dict(width=2),
                )
            )

    k_ts.add_hline(
        y=1.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Corridor",
        annotation_position="right",
    )
    k_ts.update_layout(
        title=f"K-Index Over Episodes - {title_suffix}",
        xaxis_title="Episode",
        yaxis_title="K-Index",
        template="plotly_white",
        height=400,
    )

    # ℂ Functional Components
    c_func = go.Figure()

    if selected_track in ["track_b", "track_c"]:
        if data is not None and not data.empty:
            # Use actual column names from CSV
            components = ["neg_free_energy", "blanket_tightness", "meta_calibration"]
            display_names = ["Free Energy", "Blanket Tightness", "Meta Calibration"]
            for comp, display_name in zip(components, display_names):
                if comp in data.columns:
                    c_func.add_trace(
                        go.Scatter(
                            y=data[comp], mode="lines", name=display_name, line=dict(width=2)
                        )
                    )
    elif selected_track == "track_d":
        if data is not None and not data.empty:
            c_func.add_trace(
                go.Scatter(
                    y=data["K"],
                    mode="lines",
                    name="Collective K",
                    line=dict(width=2, color="#2ecc71"),
                )
            )
            c_func.add_trace(
                go.Scatter(
                    y=data["individual_k"],
                    mode="lines",
                    name="Mean Individual K",
                    line=dict(width=2, color="#e67e22", dash="dash"),
                )
            )

    c_func.update_layout(
        title=f"ℂ Functional Components - {title_suffix}",
        xaxis_title="Episode",
        yaxis_title="Value",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Statistics Table
    stats_html = html.Div(
        [html.P("Loading statistics...", style={"textAlign": "center", "color": "#95a5a6"})]
    )

    if selected_track == "all":
        stats_rows = []
        for track_name, track_data in [
            ("Track B", data_cache["track_b"]),
            ("Track C", data_cache["track_c"]),
            ("Track D", data_cache["track_d"]),
        ]:
            if not track_data.empty and "K" in track_data.columns:
                mean_k = track_data["K"].mean()
                std_k = track_data["K"].std()
                max_k = track_data["K"].max()
                min_k = track_data["K"].min()
                corridor_pct = (track_data["K"] >= 1.5).mean() * 100

                stats_rows.append(
                    html.Tr(
                        [
                            html.Td(track_name, style={"fontWeight": "bold"}),
                            html.Td(f"{mean_k:.3f} ± {std_k:.3f}"),
                            html.Td(f"{max_k:.3f}"),
                            html.Td(f"{corridor_pct:.1f}%"),
                        ]
                    )
                )

        stats_html = html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Track"),
                            html.Th("Mean ± SD"),
                            html.Th("Max K"),
                            html.Th("% Corridor"),
                        ]
                    )
                ),
                html.Tbody(stats_rows),
            ],
            style={"width": "100%", "textAlign": "center", "marginTop": "20px"},
        )
    else:
        if data is not None and not data.empty and "K" in data.columns:
            mean_k = data["K"].mean()
            std_k = data["K"].std()
            max_k = data["K"].max()
            min_k = data["K"].min()
            corridor_pct = (data["K"] >= 1.5).mean() * 100
            n_episodes = len(data)

            stats_html = html.Div(
                [
                    html.Table(
                        [
                            html.Tr(
                                [
                                    html.Td("Episodes:", style={"fontWeight": "bold"}),
                                    html.Td(f"{n_episodes}"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Mean K:", style={"fontWeight": "bold"}),
                                    html.Td(f"{mean_k:.3f}"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Std Dev:", style={"fontWeight": "bold"}),
                                    html.Td(f"{std_k:.3f}"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Max K:", style={"fontWeight": "bold"}),
                                    html.Td(f"{max_k:.3f}"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Min K:", style={"fontWeight": "bold"}),
                                    html.Td(f"{min_k:.3f}"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("% Corridor:", style={"fontWeight": "bold"}),
                                    html.Td(f"{corridor_pct:.1f}%"),
                                ]
                            ),
                        ],
                        style={"width": "100%", "marginTop": "20px"},
                    )
                ]
            )

    # Last update time
    update_time = html.Div(
        [f"🔄 Last updated: {data_cache['last_update']} | Auto-refresh: every 5s"]
    )

    return k_dist, k_ts, c_func, stats_html, update_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kosmic Lab Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to run dashboard")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    print(f"\n🌊 Starting Kosmic Lab Dashboard on http://localhost:{args.port}")
    print("   Press Ctrl+C to stop\n")

    app.run(debug=args.debug, port=args.port, host="0.0.0.0")
