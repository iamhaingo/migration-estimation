# %% [markdown]
# # RWI Global Raster Builder
#
# Produces `/content/drive/MyDrive/parquet/rwi_global.tif` consumed by
# `model.py` (Section 4 — Static Socioeconomic Features).
#
# Output: single-band float32 GeoTIFF, EPSG:4326, 0.025° resolution (~2.5 km),
# global extent (-180 → 180 lon, -90 → 90 lat), nodata = -9999.0.
# Each pixel holds the RWI value for that grid cell; NaN-equivalent cells are
# filled with the nodata sentinel.  model.py clips this raster with the event
# AOI and extracts mean + std of valid pixels.
#
# Source:
#   Meta AI for Good — Relative Wealth Index (Chi et al., 2022)
#   https://ai.meta.com/ai-for-good/datasets/relative-wealth-index/
#   Distributed via HDX (Humanitarian Data Exchange), public, no auth required.
#
# Run this notebook ONCE before running model.py.
# Re-run only if a new RWI release is published.

# %%
!pip install requests pandas rasterio numpy -q

# %%
import io
import re
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import requests
from google.colab import drive

warnings.filterwarnings("ignore")

# %%
# ─── Configuration ────────────────────────────────────────────────────────────
OUT_TIF = "/content/drive/MyDrive/parquet/rwi_global.tif"

HDX_API_BASE     = "https://data.humdata.org/api/3/action"
HDX_DATASET_SLUG = "relative-wealth-index"

# Raster grid parameters — 0.025° ≈ 2.5 km at equator, matching RWI native ~2.4 km.
RES    = 0.025
XMIN   = -180.0
YMAX   =   90.0
NCOLS  = round(360 / RES)   # 14 400
NROWS  = round(180 / RES)   # 7 200
NODATA = -9999.0

# %%
drive.mount("/content/drive")
print("Drive mounted.")

# %% [markdown]
# ## 1. Fetch All RWI Resources from HDX

# %%
def _get_hdx_resources(api_base: str, dataset_slug: str) -> list[dict]:
    url = f"{api_base}/package_show"
    resp = requests.get(url, params={"id": dataset_slug}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"HDX API returned success=false: {payload.get('error')}")
    return payload["result"]["resources"]


def _iso3_from_resource_name(name: str) -> str | None:
    stem = re.sub(r"(\.(csv|zip))+$", "", name, flags=re.IGNORECASE)
    m = re.match(r"^([A-Za-z]{3})[\-_]", stem)
    if m:
        return m.group(1).upper()
    if re.fullmatch(r"[A-Za-z]{3}", stem):
        return stem.upper()
    return None


def _download_rwi_csv(url: str) -> pd.DataFrame | None:
    """Download one HDX RWI resource (plain CSV or zipped CSV).

    Returns a DataFrame with columns lat, lon, rwi, or None on failure.
    """
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "zip" in content_type or url.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    return None
                with zf.open(csv_names[0]) as f:
                    df = pd.read_csv(f, low_memory=False)
        else:
            df = pd.read_csv(io.StringIO(resp.text), low_memory=False)

        df.columns = df.columns.str.lower().str.strip()
        if "rwi" not in df.columns:
            return None
        if "latitude" in df.columns:
            df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
        if not {"lat", "lon", "rwi"}.issubset(df.columns):
            return None
        return df[["lat", "lon", "rwi"]]

    except Exception as exc:
        print(f"    [WARN] Download failed: {exc}")
        return None


print("Querying HDX API for RWI resource list...")
hdx_resources = _get_hdx_resources(HDX_API_BASE, HDX_DATASET_SLUG)
print(f"  Found {len(hdx_resources)} resources.")

iso3_to_url: dict[str, str] = {}
for res in hdx_resources:
    name = res.get("name", "")
    url  = res.get("download_url") or res.get("url", "")
    iso3 = _iso3_from_resource_name(name)
    if not iso3 and url:
        iso3 = _iso3_from_resource_name(url.split("/")[-1].split("?")[0])
    if iso3 and url:
        iso3_to_url[iso3] = url

print(f"  Parseable country resources: {len(iso3_to_url)}")

# %% [markdown]
# ## 2. Download All Country Point CSVs

# %%
all_chunks: list[pd.DataFrame] = []
countries = sorted(iso3_to_url)

for i, iso3 in enumerate(countries, 1):
    print(f"  [{i:3d}/{len(countries)}] {iso3} ... ", end="")
    df = _download_rwi_csv(iso3_to_url[iso3])
    if df is not None and not df.empty:
        all_chunks.append(df)
        print(f"{len(df):,} pts")
    else:
        print("SKIPPED")
    if i < len(countries):
        time.sleep(0.15)

points = pd.concat(all_chunks, ignore_index=True)
points["rwi"] = pd.to_numeric(points["rwi"], errors="coerce")
points = points.dropna(subset=["lat", "lon", "rwi"])
points = points[
    (points["lon"] >= XMIN) & (points["lon"] < XMIN + NCOLS * RES) &
    (points["lat"] > YMAX - NROWS * RES) & (points["lat"] <= YMAX)
].copy()

print(f"\nTotal valid points: {len(points):,}")

# %% [markdown]
# ## 3. Rasterize Points → Global GeoTIFF

# %%
# Map each point to its grid cell index.
# Pixel origin is top-left (XMIN, YMAX); rows run south, cols run east.
points["col"] = np.floor((points["lon"] - XMIN) / RES).astype(np.int32)
points["row"] = np.floor((YMAX - points["lat"]) / RES).astype(np.int32)

# Clip to valid range (handles floating-point edge cases)
points = points[
    (points["row"] >= 0) & (points["row"] < NROWS) &
    (points["col"] >= 0) & (points["col"] < NCOLS)
]

# Average cells that receive more than one point (rare, occurs near poles)
cell_means = (
    points.groupby(["row", "col"])["rwi"]
    .mean()
    .reset_index()
)

print(f"Unique raster cells filled: {len(cell_means):,}  "
      f"({100 * len(cell_means) / (NROWS * NCOLS):.2f}% of global grid)")

grid = np.full((NROWS, NCOLS), NODATA, dtype=np.float32)
rows_idx = cell_means["row"].to_numpy()
cols_idx = cell_means["col"].to_numpy()
grid[rows_idx, cols_idx] = cell_means["rwi"].to_numpy().astype(np.float32)

rwi_valid = grid[grid != NODATA]
print(f"RWI range in raster: {rwi_valid.min():.4f} – {rwi_valid.max():.4f}")
print(f"RWI global mean    : {rwi_valid.mean():.4f}")

# %%
transform = from_origin(XMIN, YMAX, RES, RES)

with rasterio.open(
    OUT_TIF, "w",
    driver="GTiff",
    height=NROWS,
    width=NCOLS,
    count=1,
    dtype="float32",
    crs="EPSG:4326",
    transform=transform,
    nodata=NODATA,
    compress="deflate",
    predictor=3,   # floating-point DEFLATE predictor
    zlevel=6,
) as dst:
    dst.write(grid, 1)

import os
size_mb = os.path.getsize(OUT_TIF) / 1e6
print(f"\nSaved to {OUT_TIF}  ({size_mb:.1f} MB on disk)")

# %% [markdown]
# ## 4. Sanity Check — Sample Known Locations

# %%
# Spot-check: read back pixel values at known capital-city coordinates and
# verify they are within the expected LMIC RWI range [−2, +2].
spot_checks = [
    ("Lagos, NGA",    6.45,  3.40),
    ("Dhaka, BGD",   23.81, 90.41),
    ("Nairobi, KEN", -1.29, 36.82),
    ("Kinshasa, COD", -4.32, 15.32),
    ("Addis, ETH",    9.03, 38.74),
]

print(f"{'Location':20} {'Lat':>7} {'Lon':>8} {'RWI pixel':>12} {'In range':>10}")
print("-" * 62)
with rasterio.open(OUT_TIF) as src:
    for label, lat, lon in spot_checks:
        row_px = int((YMAX - lat) / RES)
        col_px = int((lon - XMIN) / RES)
        row_px = max(0, min(NROWS - 1, row_px))
        col_px = max(0, min(NCOLS - 1, col_px))
        val = float(src.read(1)[row_px, col_px])
        in_range = "✓" if val != NODATA and -3.0 <= val <= 3.0 else "✗"
        val_str = f"{val:12.4f}" if val != NODATA else "     nodata"
        print(f"{label:20} {lat:7.2f} {lon:8.2f} {val_str} {in_range:>10}")

# %% [markdown]
# ## 5. Plot the Global RWI Raster

# %%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

with rasterio.open(OUT_TIF) as src:
    data = src.read(1).astype(np.float64)

data[data == NODATA] = np.nan

vmin, vmax = np.nanpercentile(data, [2, 98])  # clip colorscale to 2nd–98th pct

fig, ax = plt.subplots(figsize=(16, 8), dpi=150)

img = ax.imshow(
    data,
    extent=[XMIN, XMIN + NCOLS * RES, YMAX - NROWS * RES, YMAX],
    origin="upper",
    cmap="RdYlGn",
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
    aspect="auto",
)

cbar = fig.colorbar(img, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Relative Wealth Index", fontsize=10)

ax.set_title("Meta AI for Good — Relative Wealth Index (Chi et al., 2022)\n"
             "0.025° global raster  |  green = wealthier, red = poorer", fontsize=11)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
ax.axvline(0, color="gray", linewidth=0.4, linestyle="--")

plt.tight_layout()
plt.show()

# %%
print("\nDone. rwi_global.tif is ready for model.py.")
