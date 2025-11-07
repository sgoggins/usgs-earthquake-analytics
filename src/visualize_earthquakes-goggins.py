#!/usr/bin/env python3
"""
Earthquake visualization with a rolling window video (no-arg defaults).

Defaults:
- Data: ../data/GeoQuake.json   (relative to this script)
- Out : ../visualizations       (relative to this script)
- Window (days): 7
- FPS: 4
- Frame size: 768x512 (both divisible by 16 to avoid ffmpeg warnings)

Outputs:
- HTML  : heatmap_temporal.html (daily animation, interactive)
- MP4   : heatmap_temporal_rolling.mp4 (rolling window)
- PNGs  : scatter_depth_magnitude.png, frequency_over_time.png
- Frames: frames_rolling/*.png
"""

import json
import os
from pathlib import Path
import sys

import imageio.v2 as imageio
import pandas as pd
import plotly.express as px

# ----------- Defaults (zero-config) -----------
BASE = Path(__file__).resolve().parent
DATA_PATH = (BASE / "../data/GeoQuake.json").resolve()
OUT_DIR = (BASE / "../visualizations").resolve()
FRAMES_DIR = OUT_DIR / "frames_rolling"

WINDOW_DAYS = 7
FPS = 4
WIDTH = 768   # divisible by 16
HEIGHT = 512  # divisible by 16


def require_file(path: Path):
    if not path.exists():
        print(f"[ERROR] Missing data file: {path}")
        print("        Fix: put your GeoJSON there or edit DATA_PATH at top of script.")
        sys.exit(1)


def load_geojson(path: Path) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)

    records = []
    for feat in data.get("features", []):
        try:
            props = feat.get("properties", {})
            lon, lat, depth = feat["geometry"]["coordinates"][:3]
            records.append(
                {
                    "time": pd.to_datetime(props.get("time"), unit="ms", errors="coerce"),
                    "magnitude": props.get("mag", None),
                    "depth": depth,
                    "longitude": lon,
                    "latitude": lat,
                    "place": props.get("place", "Unknown"),
                }
            )
        except Exception:
            # skip malformed rows
            continue

    df = pd.DataFrame.from_records(records)
    # clean
    df = df.dropna(subset=["time", "latitude", "longitude", "magnitude", "depth"])
    df = df[df["depth"] >= 0].copy()
    df["date"] = df["time"].dt.date.astype(str)
    df["depth_scaled"] = (df["depth"] / df["depth"].max()) * 10.0 if len(df) else 0.0
    return df


def write_image(fig, path: Path, width: int, height: int):
    """Write static image with friendly kaleido error."""
    try:
        fig.write_image(str(path), width=width, height=height, scale=1)
    except Exception as e:
        msg = str(e).lower()
        if "kaleido" in msg:
            print("[ERROR] Static image export requires 'kaleido'.")
            print("        Run: pip install kaleido")
        else:
            print(f"[ERROR] Failed to write image {path.name}: {e}")
        sys.exit(1)


def make_html_animation(df: pd.DataFrame, out_dir: Path):
    fig = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        color="magnitude",
        size="depth_scaled",
        hover_name="place",
        animation_frame="date",
        projection="natural earth",
        title="Global Earthquake Events Over Time (Daily Animation)",
        color_continuous_scale="Turbo",
        size_max=10,
    )
    out_path = out_dir / "heatmap_temporal.html"
    fig.write_html(str(out_path))
    print(f"[HTML]  {out_path}")


def write_rolling_frames(df: pd.DataFrame, frames_dir: Path, width: int, height: int, window_days: int):
    frames_dir.mkdir(parents=True, exist_ok=True)

    # continuous daily range
    start_day = df["time"].dt.normalize().min()
    end_day = df["time"].dt.normalize().max()
    if pd.isna(start_day) or pd.isna(end_day):
        print("[ERROR] No valid dates found in the dataset.")
        sys.exit(1)

    all_days = pd.date_range(start_day, end_day, freq="D")

    for idx, day in enumerate(all_days, start=1):
        win_start = (day - pd.Timedelta(days=window_days - 1)).normalize()
        win_end = day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        mask = (df["time"] >= win_start) & (df["time"] <= win_end)
        df_win = df.loc[mask]

        subtitle = f"Window: last {window_days} days up to {day.date()} • n={len(df_win)}"
        fig = px.scatter_geo(
            df_win,
            lat="latitude",
            lon="longitude",
            color="magnitude",
            size="depth_scaled",
            hover_name="place",
            projection="natural earth",
            title="Global Earthquakes (Rolling Window)",
            color_continuous_scale="Turbo",
            size_max=10,
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            coloraxis_colorbar=dict(title="Magnitude"),
            annotations=[
                dict(
                    text=subtitle,
                    x=0.5,
                    xref="paper",
                    y=0.02,
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12),
                )
            ],
        )

        frame_path = frames_dir / f"frame_{idx:04d}.png"
        write_image(fig, frame_path, width, height)
        if idx % 25 == 0 or idx == len(all_days):
            print(f"[FRAME] {frame_path.name} ({idx}/{len(all_days)})")


def frames_to_video(frames_dir: Path, out_mp4: Path, fps: int):
    pngs = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() == ".png"])
    if not pngs:
        print(f"[ERROR] No frames found in {frames_dir} — cannot encode video.")
        sys.exit(1)
    images = [imageio.imread(p) for p in pngs]
    imageio.mimsave(str(out_mp4), images, fps=fps)
    print(f"[MP4]   {out_mp4}")


def make_static_plots(df: pd.DataFrame, out_dir: Path):
    # Magnitude vs Depth
    fig_scatter = px.scatter(
        df,
        x="magnitude",
        y="depth",
        color="magnitude",
        hover_name="place",
        title="Magnitude vs Depth of Earthquakes",
        labels={"magnitude": "Magnitude", "depth": "Depth (km)"},
        color_continuous_scale="Viridis",
    )
    fig_scatter.update_yaxes(autorange="reversed")
    scatter_path = out_dir / "scatter_depth_magnitude.png"
    write_image(fig_scatter, scatter_path, WIDTH, HEIGHT)
    print(f"[PNG]   {scatter_path}")

    # Frequency over time
    df_freq = df.groupby("date").size().reset_index(name="count")
    fig_line = px.line(
        df_freq,
        x="date",
        y="count",
        title="Earthquake Frequency Over Time",
        labels={"date": "Date", "count": "Number of Earthquakes"},
    )
    freq_path = out_dir / "frequency_over_time.png"
    write_image(fig_line, freq_path, WIDTH, HEIGHT)
    print(f"[PNG]   {freq_path}")


def main():
    print(f"[CONF]  DATA_PATH={DATA_PATH}")
    print(f"[CONF]  OUT_DIR  ={OUT_DIR}")
    print(f"[CONF]  WINDOW   ={WINDOW_DAYS} days • FPS={FPS} • SIZE={WIDTH}x{HEIGHT}")

    require_file(DATA_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[LOAD]  Reading data…")
    df = load_geojson(DATA_PATH)
    if df.empty:
        print("[ERROR] Loaded 0 rows from data. Are there valid features in your GeoJSON?")
        sys.exit(1)

    tmin, tmax = df['time'].min(), df['time'].max()
    print(f"[INFO]  Loaded {len(df)} quakes between {tmin.date()} and {tmax.date()}")

    print("[HTML]  Building interactive daily animation…")
    make_html_animation(df, OUT_DIR)

    print("[ROLL]  Writing rolling-window frames…")
    write_rolling_frames(df, FRAMES_DIR, WIDTH, HEIGHT, WINDOW_DAYS)

    print("[VID]   Encoding MP4 from frames…")
    mp4_path = OUT_DIR / "heatmap_temporal_rolling.mp4"
    frames_to_video(FRAMES_DIR, mp4_path, FPS)

    print("[PLOTS] Writing static plots…")
    make_static_plots(df, OUT_DIR)

    print("\nDone.")
    print(f"- HTML : {OUT_DIR / 'heatmap_temporal.html'}")
    print(f"- MP4  : {mp4_path}")
    print(f"- Frames: {FRAMES_DIR}")
    print(f"- PNGs : {OUT_DIR / 'scatter_depth_magnitude.png'}, {OUT_DIR / 'frequency_over_time.png'}")


if __name__ == "__main__":
    main()