# %%
import os

API_TOKEN = os.environ.get("IDMC_API_KEY", "")
DATA_PATH = "../data/"


# %%
import requests
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.lines import Line2D
from pathlib import Path
from typing import List, Optional


# %%
def _fetch(api_token: str, url: str, limit: int = 20) -> dict:
    response = requests.get(
        url,
        params={"client_id": api_token, "limit": limit},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def fetch_disaggregated_data(
    api_token: str,
    limit: int = 20,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> gpd.GeoDataFrame:
    """
    Fetch disaggregated geojson from the API and cache to disk. If a cached
    file exists and force_refresh is False, read from cache instead of fetching.
    """
    url = "https://helix-tools-api.idmcdb.org/external-api/gidd/disaggregations/disaggregation-geojson/"
    cache_path = (
        Path(cache_path)
        if cache_path is not None
        else Path(DATA_PATH) / "disaggregation_geo.parquet"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_refresh:
        gdf = gpd.read_parquet(cache_path)
        print(f"Loaded {len(gdf):,} records from cache: {cache_path}")
        return gdf

    payload = _fetch(api_token, url, limit)
    gdf = gpd.GeoDataFrame.from_features(payload["features"])
    try:
        gdf.to_parquet(cache_path, index=False)
        print(f"Fetched {len(gdf):,} records and saved to: {cache_path}")
    except Exception:
        # fallback: save as GeoJSON if parquet not supported
        geojson_path = cache_path.with_suffix(".geojson")
        gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"Fetched {len(gdf):,} records and saved to: {geojson_path}")

    return gdf


gdf_raw = fetch_disaggregated_data(API_TOKEN)
print(f"Loaded {len(gdf_raw):,} records")


# %% [markdown]
#  # data exploration

# %%
print("=== DISAGGREGATED DATA OVERVIEW ===")
print(f"Shape: {gdf_raw.shape}")
print(f"\nColumns: {gdf_raw.columns.tolist()}")
print(f"\nData types:\n{gdf_raw.dtypes}")
print("\nHazard Categories:")
print(f"  - Category: {gdf_raw['Hazard category'].unique()}")
print(f"  - Sub-category: {gdf_raw['Hazard sub category'].unique()}")
print(f"  - Type: {gdf_raw['Hazard type'].unique()}")
print(f"  - Sub-type: {gdf_raw['Hazard sub type'].unique()}")
print(f"  - Figure cause: {gdf_raw['Figure cause'].unique()}")
print("\nHazard type to sub-type mapping:")
print(
    gdf_raw[["Hazard type", "Hazard sub type"]]
    .dropna()
    .drop_duplicates()
    .sort_values(["Hazard type", "Hazard sub type"])
    .to_string(index=False)
)


# %%
def filter_disaster_rapid_onset(
    gdf_raw: pd.DataFrame, verbose: bool = True
) -> pd.DataFrame:
    """
    Filter dataset for:
    - Figure cause == 'Disaster'
    - AND (Hydrological Flood OR Meteorological Storm)

    Parameters
    ----------
    gdf_raw : pd.DataFrame
        Input dataset containing hazard and cause columns.
    verbose : bool, default True
        If True, prints summary statistics.

    Returns
    -------
    pd.DataFrame
        Filtered copy of the input dataframe.
    """

    # Core masks
    disaster_mask = gdf_raw["Figure cause"] == "Disaster"

    hydro_mask = (gdf_raw["Hazard sub category"] == "Hydrological") & (
        gdf_raw["Hazard type"] == "Flood"
    )

    meteor_mask = (gdf_raw["Hazard sub category"] == "Meteorological") & (
        gdf_raw["Hazard type"] == "Storm"
    )

    # Combined filter
    combined_mask = disaster_mask & (hydro_mask | meteor_mask)
    gdf_work = gdf_raw.loc[combined_mask].copy()

    if verbose:
        print(f"Original data: {len(gdf_raw):,} records")
        print(f"After filtering for Disaster cause: {disaster_mask.sum():,} records")
        print(
            f"  - Hydrological rapid-onset (Floods): {(disaster_mask & hydro_mask).sum():,}"
        )
        print(
            f"  - Meteorological rapid-onset (Storms): {(disaster_mask & meteor_mask).sum():,}"
        )
        print(
            f"\nFinal filtered data (Disaster + rapid-onset hazards): {len(gdf_work):,} records"
        )

        print(f"\nHazard breakdown in filtered data:")
        print(gdf_work[["Hazard sub category", "Hazard type"]].value_counts().head(10))

    return gdf_work


gdf_work = filter_disaster_rapid_onset(gdf_raw)


# %%
(gdf_work["Reported figures"] == 0).sum()


# %%
def explode_multipoints_and_lists(
    gdf: gpd.GeoDataFrame, figure_column: str, list_cols: Optional[List[str]] = None
) -> gpd.GeoDataFrame:
    """
    Explodes MultiPoint geometries and parallel list columns into individual rows.

    Strictly validates that each specified list column has the exact same length as
    the number of points in the corresponding MultiPoint geometry. Distributes the
    numeric figure evenly across the newly separated points.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The input GeoDataFrame containing MultiPoints and list columns.
    figure_column : str
        The column name of the numeric figure to be distributed.
    list_cols : list of str, optional
        Columns containing lists parallel to the MultiPoint geometries. Defaults to
        ['Locations type', 'Locations accuracy', 'Locations name'].

    Returns
    -------
    gpd.GeoDataFrame
        The exploded GeoDataFrame with a new 'distributed_figure' column.
    """
    # 1. Setup & Input Normalization
    df = gdf.copy()
    df[figure_column] = pd.to_numeric(df[figure_column], errors="coerce")

    list_cols = list_cols or ["Locations type", "Locations accuracy", "Locations name"]
    valid_cols = [col for col in list_cols if col in df.columns]

    # 2. Normalize Geometries to lists of individual Points
    df["geom_list"] = df.geometry.apply(
        lambda g: list(g.geoms) if hasattr(g, "geoms") else [g]
    )
    df["geom_count"] = df["geom_list"].str.len()

    # 3. Normalize & Strictly Validate List Columns (Vectorized)
    for col in valid_cols:
        # Convert scalars/NaNs to lists, and ensure existing iterables are pure lists
        df[col] = df[col].apply(
            lambda x: (
                list(x)
                if isinstance(x, (list, tuple, np.ndarray))
                else ([] if pd.isna(x) else [x])
            )
        )

        # Vectorized length check (much faster than row-by-row validation)
        col_lengths = df[col].str.len()
        mismatches = df[col_lengths != df["geom_count"]]

        if not mismatches.empty:
            err_idx = mismatches.index[0]
            geom_len = mismatches.loc[err_idx, "geom_count"]
            val_len = col_lengths.loc[err_idx]
            val = mismatches.loc[err_idx, col]

            raise ValueError(
                f"Length mismatch at row index {err_idx} for column '{col}'. "
                f"Geometry has {geom_len} point(s), but list has {val_len} item(s). "
                f"Values: {val}"
            )

    # 4. Explode Data
    # Drop original geometry to prevent GeoPandas dtype conflicts during explode
    df = df.drop(columns=["geometry"])
    exploded_df = df.explode(column=["geom_list"] + valid_cols, ignore_index=True)

    # 5. Reconstruct GeoDataFrame & Distribute Figures
    exploded_gdf = gpd.GeoDataFrame(
        exploded_df, geometry="geom_list", crs=gdf.crs
    ).rename_geometry("geometry")

    exploded_gdf["distributed_figure"] = (
        exploded_gdf[figure_column] / exploded_gdf["geom_count"]
    )

    return exploded_gdf.drop(columns=["geom_count"])


# %%
gdf_work = explode_multipoints_and_lists(
    gdf=gdf_work,
    figure_column="Total figures",
    list_cols=[
        "Locations type",
        "Locations accuracy",
        "Locations name",
    ],  # Specify the list columns to be exploded
)
gdf_work


# %%
def filter_accuracy_dates(gdf_work: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Filter dataset to keep only records where both:
    - Start date accuracy == 'Day'
    - End date accuracy == 'Day'

    Parameters
    ----------
    gdf_work : pd.DataFrame
        Input dataframe to filter.
    verbose : bool, default True
        If True, prints record count after filtering.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with valid day-level date accuracy.
    """

    date_accuracy_mask = (gdf_work["Start date accuracy"] == "Day") & (
        gdf_work["End date accuracy"] == "Day"
    )

    gdf_filtered = gdf_work.loc[date_accuracy_mask].copy()

    if verbose:
        print(
            f"\nAfter filtering for valid start/end dates: {len(gdf_filtered):,} records"
        )

    return gdf_filtered


gdf_work = filter_accuracy_dates(gdf_work)


# %%
print(gdf_work["Locations accuracy"].value_counts().to_string())


# %%
def filter_by_location_accuracy(gdf_work, keep_levels=None):
    if keep_levels is None:
        keep_levels = {"County/City/town/Village/Woreda (ADM3)", "Point"}

    filtered = gdf_work[gdf_work["Locations accuracy"].isin(keep_levels)].copy()

    print(f"\nAfter filtering for location accuracy: {len(filtered):,} records")
    return filtered


gdf_work = filter_by_location_accuracy(gdf_work)


# %%
location_counts = gdf_work["Locations type"].value_counts()
print("\nLocation type counts:")
print(location_counts.head(10))


# %%
gdf_work = gdf_work[gdf_work["Locations type"] == "Origin"].copy()
print(f"\nAfter filtering for location type 'Origin': {len(gdf_work):,} records")


# %%
event_counts = gdf_work["Event ID"].value_counts()
print("\nEvent ID counts:")
print(event_counts.head(10))


# %%
len(event_counts), len(gdf_work)


# %%
gdf_work


# %%
cols_to_keep = [
    "Event ID",
    "geometry",
    "ISO3",
    "Start date",
    "End date",
    "Hazard type",
    "Hazard sub category",
    "distributed_figure",
]
gdf_satellite_input = gdf_work[cols_to_keep].copy()
gdf_satellite_input


# %%
gdf_satellite_input["Start date"] = pd.to_datetime(
    gdf_satellite_input["Start date"], errors="coerce"
)
gdf_satellite_input["End date"] = pd.to_datetime(
    gdf_satellite_input["End date"], errors="coerce"
)


# %%
output_path = Path(DATA_PATH) / "disaster_rapid_onset_disaggregated.parquet"
output_path.parent.mkdir(parents=True, exist_ok=True)

# %%
gdf_satellite_input.to_parquet(output_path, index=False)

# %%
print(output_path.resolve())

# %%
# read back in to verify
gdf_check = gpd.read_parquet(output_path)

# %% [markdown]
# - Plot

# %%
figure_column = "distributed_figure"

figure_values = pd.to_numeric(gdf_check[figure_column], errors="coerce").dropna()
missing_count = gdf_check[figure_column].isna().sum()

if figure_values.empty:
    raise ValueError(f"No numeric values found in {figure_column}")

quantiles = figure_values.quantile(
    [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
)
q1 = quantiles.loc[0.25]
q3 = quantiles.loc[0.75]
iqr = q3 - q1
outlier_threshold = q3 + 1.5 * iqr
outlier_share = (figure_values > outlier_threshold).mean()
zero_share = (figure_values == 0).mean()
positive_values = figure_values[figure_values > 0]
skewness = figure_values.skew()
mean_val = figure_values.mean()
std_val = figure_values.std()

print(f"Using figure column: {figure_column}")
print(f"Rows in dataset: {len(gdf_check):,}")
print(f"Non-missing numeric values: {len(figure_values):,}")
print(f"Missing values in {figure_column}: {missing_count:,}")
print(f"Distinct numeric values: {figure_values.nunique():,}")
print(f"Zero share: {zero_share:.2%}")
print(
    f"Outlier share above IQR threshold ({outlier_threshold:,.2f}): {outlier_share:.2%}"
)
print(f"Skewness: {skewness:.2f}")

print("\nSummary statistics")
summary = pd.DataFrame(
    {
        "count": [figure_values.count()],
        "mean": [mean_val],
        "std": [std_val],
        "min": [figure_values.min()],
        "1%": [quantiles.loc[0.01]],
        "5%": [quantiles.loc[0.05]],
        "10%": [quantiles.loc[0.10]],
        "25%": [q1],
        "50%": [quantiles.loc[0.50]],
        "75%": [q3],
        "90%": [quantiles.loc[0.90]],
        "95%": [quantiles.loc[0.95]],
        "99%": [quantiles.loc[0.99]],
        "max": [figure_values.max()],
        "skewness": [skewness],
        "missing": [missing_count],
        "zero_share": [zero_share],
        "outlier_share": [outlier_share],
    }
)
print(summary.round(2).to_string(index=False))

print("\nMost common values")
common_values = (
    figure_values.round()
    .astype("Int64")
    .value_counts()
    .head(10)
    .rename_axis("figure")
    .reset_index(name="count")
)
print(common_values.to_string(index=False))

print("\nInterpretation")
if skewness > 1:
    shape_note = "strongly right-skewed"
elif skewness > 0.5:
    shape_note = "moderately right-skewed"
elif skewness < -0.5:
    shape_note = "left-skewed"
else:
    shape_note = "roughly symmetric"
print(f"- The distribution is {shape_note}, with a long upper tail.")
print(
    f"- The median ({quantiles.loc[0.50]:,.2f}) is much more informative than the mean ({mean_val:,.2f}) if skewness is high."
)
print(
    f"- About {outlier_share:.2%} of observations sit above the classic IQR outlier threshold."
)
if len(positive_values) > 0:
    print(
        "- A log scale is likely to help if you model this variable directly or use it as a target."
    )
else:
    print(
        "- The values are too sparse for a log-scale readout; inspect the raw measurement first."
    )

# Publication-grade visualization
plt.style.use("seaborn-v0_8-darkgrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=100)
fig.suptitle(
    f"Distribution Analysis: {figure_column} ({len(gdf_check):,} records)",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

# ===== Top-left: Raw histogram with mean/median lines =====
ax0 = axes[0, 0]
sns.histplot(
    figure_values,
    bins=50,
    kde=True,
    ax=ax0,
    color="#2E86AB",
    edgecolor="white",
    linewidth=0.5,
)
median_val = quantiles.loc[0.50]
mean_val = figure_values.mean()
ax0.axvline(
    median_val,
    color="#A23B72",
    linestyle="--",
    linewidth=2.5,
    label=f"Median: {median_val:,.0f}",
)
ax0.axvline(
    mean_val,
    color="#F18F01",
    linestyle="--",
    linewidth=2.5,
    label=f"Mean: {mean_val:,.0f}",
)
ax0.set_title("Distribution (Raw Scale)", fontsize=12, fontweight="bold", pad=10)
ax0.set_xlabel(f"{figure_column}", fontsize=11)
ax0.set_ylabel("Frequency", fontsize=11)
ax0.legend(fontsize=10, loc="upper right")
ax0.grid(axis="y", alpha=0.3)

# ===== Top-right: Box plot with quartile annotations =====
ax1 = axes[0, 1]
bp = ax1.boxplot(
    figure_values,
    vert=False,
    widths=0.4,
    patch_artist=True,
    boxprops=dict(facecolor="#2E86AB", alpha=0.7),
    medianprops=dict(color="#A23B72", linewidth=2.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
)
ax1.set_xlabel(f"{figure_column}", fontsize=11)
ax1.set_title(
    "Boxplot (Raw Scale) with Quartiles", fontsize=12, fontweight="bold", pad=10
)
ax1.set_yticklabels([])
# Add quartile annotations
ax1.text(
    q1, 1.25, f"Q1\n{q1:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
)
ax1.text(
    median_val,
    1.25,
    f"Median\n{median_val:,.0f}",
    ha="center",
    va="bottom",
    fontsize=9,
    fontweight="bold",
)
ax1.text(
    q3, 1.25, f"Q3\n{q3:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
)
ax1.grid(axis="x", alpha=0.3)

# ===== Bottom-left: Log-transformed histogram =====
ax2 = axes[1, 0]
if len(positive_values) > 0:
    log_vals = np.log1p(positive_values)
    sns.histplot(
        log_vals,
        bins=50,
        kde=True,
        ax=ax2,
        color="#06A77D",
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.axvline(
        log_vals.median(),
        color="#A23B72",
        linestyle="--",
        linewidth=2.5,
        label=f"Median (log1p): {log_vals.median():.2f}",
    )
    ax2.set_xlabel("log₁ₚ(value)", fontsize=11)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.set_title(
        "Distribution (Log₁ₚ-Transformed Scale)", fontsize=12, fontweight="bold", pad=10
    )
else:
    ax2.text(
        0.5,
        0.5,
        "No positive values to log-transform",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax2.transAxes,
    )
    ax2.set_title("Log-Transformed (N/A)", fontsize=12, fontweight="bold", pad=10)
    ax2.axis("off")
ax2.set_ylabel("Frequency", fontsize=11)
ax2.grid(axis="y", alpha=0.3)

# ===== Bottom-right: Summary statistics table =====
ax3 = axes[1, 1]
ax3.axis("off")
summary_text = f"""
DISTRIBUTION SUMMARY STATISTICS

Shape & Skewness:
  • Distribution: {shape_note.upper()}
  • Skewness coefficient: {skewness:.2f}
  • Zero values: {zero_share:.2%} ({(zero_share * len(figure_values)):,.0f} obs)
  • Outliers (> Q3 + 1.5·IQR): {outlier_share:.2%} ({(outlier_share * len(figure_values)):,.0f} obs)

Central Tendency:
  • Mean: {mean_val:,.2f}
  • Median: {median_val:,.0f}
  • Std Dev: {std_val:,.2f}

Quartiles:
  • Q1 (25%): {q1:,.0f}
  • Q2 (50%): {median_val:,.0f}
  • Q3 (75%): {q3:,.0f}
  • IQR: {iqr:,.0f}

Range:
  • Min: {figure_values.min():,.0f}
  • 1st %ile: {quantiles.loc[0.01]:,.0f}
  • 5th %ile: {quantiles.loc[0.05]:,.0f}
  • 10th %ile: {quantiles.loc[0.10]:,.0f}
  • 90th %ile: {quantiles.loc[0.90]:,.0f}
  • 95th %ile: {quantiles.loc[0.95]:,.0f}
  • 99th %ile: {quantiles.loc[0.99]:,.0f}
  • Max: {figure_values.max():,.0f}

Modeling Recommendations:
  ✓ Log-transform the target for regression (reduces skew)
  ✓ Use robust metrics (median, IQR) for evaluation
  ✓ Consider stratified sampling by magnitude bins
  ✓ Robust regression or quantile regression may outperform OLS
"""
ax3.text(
    0.05,
    0.95,
    summary_text,
    transform=ax3.transAxes,
    fontsize=10,
    verticalalignment="top",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="#F0F0F0", alpha=0.8, pad=1),
)

plt.tight_layout()
plt.show()


# %%
gdf_disaster = gdf_raw[gdf_raw["Figure cause"] == "Disaster"].copy()

# %%
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from matplotlib.cm import ScalarMappable
import pycountry

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

# Perceptually-distinct qualitative palette (Okabe-Ito + extensions),
# chosen to remain distinguishable under common forms of colour-vision
# deficiency and to print well in greyscale.
HAZARD_COLORS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # teal green
    "#CC79A7",  # mauve
    "#0072B2",  # deep blue
    "#D55E00",  # vermilion
    "#F0E442",  # yellow  (use last — low contrast on white)
    "#999999",  # mid-grey
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _fmt(x, _=None):
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{int(x)}"


def _trunc(text, n=18):
    return text if len(text) <= n else text[: n - 1] + "…"


def _resolve(code, ne_lookup):
    """Return a human-readable country name for *code*.

    Priority:
    1. Natural Earth display name (already present in the shapefile).
    2. pycountry alpha-3 lookup.
    3. pycountry alpha-2 lookup.
    4. pycountry fuzzy search.
    5. Fall back to the raw code.
    """
    if not isinstance(code, str):
        return str(code)

    # Natural Earth name takes precedence (most display-friendly)
    if code in ne_lookup:
        return ne_lookup[code]

    # Try pycountry alpha-3
    country = pycountry.countries.get(alpha_3=code.upper())
    if country:
        return country.common_name if hasattr(country, "common_name") else country.name

    # Try pycountry alpha-2
    country = pycountry.countries.get(alpha_2=code.upper())
    if country:
        return country.common_name if hasattr(country, "common_name") else country.name

    # Fuzzy search as a last resort (pycountry >= 22.x)
    try:
        results = pycountry.countries.search_fuzzy(code)
        if results:
            c = results[0]
            return c.common_name if hasattr(c, "common_name") else c.name
    except LookupError:
        pass

    return code


# -----------------------------------------------------------------------------
# Main figure
# -----------------------------------------------------------------------------


def plot_hazards(gdf, figure_column="Total figures", hazard_col="Hazard type"):

    df = gdf.copy()

    df[figure_column] = pd.to_numeric(df[figure_column], errors="coerce")

    df = df[(df[figure_column] > 0) & df.geometry.is_valid].dropna(
        subset=[figure_column, hazard_col, "geometry"]
    )

    if (df.geometry.geom_type == "MultiPoint").any():
        df = df.explode(index_parts=False)
        df = df[df.geometry.geom_type == "Point"]

    country_col = (
        next(
            (
                c
                for c in df.columns
                if c.lower() in ("iso3", "iso", "country_iso", "iso3_code")
            ),
            None,
        )
        or next((c for c in df.columns if "iso" in c.lower()), None)
        or next((c for c in df.columns if "countr" in c.lower()), None)
    )

    # -------------------------------------------------------------------------
    # Hazard summaries
    # -------------------------------------------------------------------------

    hazard_totals = (
        df.groupby(hazard_col)[figure_column].sum().sort_values(ascending=True)
    )

    hazard_order = (
        df.groupby(hazard_col)[figure_column]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    hcmap = {
        h: HAZARD_COLORS[i % len(HAZARD_COLORS)] for i, h in enumerate(hazard_order)
    }

    # -------------------------------------------------------------------------
    # Country polygons
    # -------------------------------------------------------------------------

    ne_path = shpreader.natural_earth(
        resolution="110m", category="cultural", name="admin_0_countries"
    )

    ne_gdf = gpd.read_file(ne_path)[["ADM0_A3", "NAME", "geometry"]].rename(
        columns={"ADM0_A3": "ISO3"}
    )

    # Build lookup used by _resolve()
    iso_lookup = dict(zip(ne_gdf["ISO3"], ne_gdf["NAME"]))

    if country_col:
        ne_gdf = ne_gdf.merge(
            df.groupby(country_col).size().rename("n_events"),
            left_on="ISO3",
            right_index=True,
            how="left",
        )
    else:
        joined = gpd.sjoin(
            df[["geometry"]],
            ne_gdf.to_crs(4326),
            how="left",
            predicate="within",
        )
        ne_gdf["n_events"] = ne_gdf.index.map(joined.groupby("index_right").size())

    ne_gdf["n_events"] = ne_gdf["n_events"].fillna(0)
    ne_gdf["log_events"] = np.log10(ne_gdf["n_events"] + 1)

    # -------------------------------------------------------------------------
    # Top countries
    # -------------------------------------------------------------------------

    if country_col:
        top_raw = (
            df.groupby(country_col)[figure_column].sum().nlargest(10).sort_values()
        )

        top = top_raw.rename(index=lambda c: _trunc(_resolve(c, iso_lookup)))

    else:
        joined = gpd.sjoin(
            df[[figure_column, hazard_col, "geometry"]],
            ne_gdf[["NAME", "geometry"]],
            how="left",
            predicate="within",
        )

        top = joined.groupby("NAME")[figure_column].sum().nlargest(10).sort_values()

    COUNTRY_BAR_COLOR = "#0072B2"  # single uniform colour for Panel C

    # -------------------------------------------------------------------------
    # Figure layout
    # -------------------------------------------------------------------------

    fig = plt.figure(figsize=(7, 9), constrained_layout=True)

    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.8, 1],
        width_ratios=[1.3, 1],
    )

    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.EqualEarth())
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # -------------------------------------------------------------------------
    # Panel A: map
    # -------------------------------------------------------------------------

    ax_map.set_global()

    ax_map.add_feature(cfeature.LAND, facecolor="#f0ede8", edgecolor="none")
    ax_map.add_feature(cfeature.OCEAN, facecolor="#dce8f0")
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.3, color="0.45")

    # Choropleth: "Cividis" is perceptually uniform and print-safe.
    cmap = plt.get_cmap("cividis")
    norm = mcolors.Normalize(vmin=0, vmax=ne_gdf["log_events"].max())

    for _, row in ne_gdf.iterrows():
        if row.geometry is None:
            continue

        if row["n_events"] == 0:
            fc = "#d8d4cc"  # light warm grey for no-data countries
            ec = "#b8b4ac"
            lw = 0.15
        else:
            fc = cmap(norm(row["log_events"]))
            ec = "#7a7a7a"
            lw = 0.25

        ax_map.add_geometries(
            [row.geometry],
            crs=ccrs.PlateCarree(),
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cb = fig.colorbar(sm, ax=ax_map, orientation="horizontal", shrink=0.55, pad=0.02)
    cb.set_label("Events per country (log₁₀ scale)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax_map.set_title("A. Event frequency by country", loc="left", fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel B: hazards
    # -------------------------------------------------------------------------

    colors = [hcmap[h] for h in hazard_totals.index]

    bars = ax_b.barh(
        hazard_totals.index,
        hazard_totals.values,
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        height=0.65,
    )

    ax_b.set_xscale("log")
    ax_b.set_xlabel("Total displaced")
    ax_b.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))

    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.tick_params(axis="y", labelsize=8)

    for bar, val in zip(bars, hazard_totals.values):
        ax_b.text(
            val * 1.06,
            bar.get_y() + bar.get_height() / 2,
            _fmt(val),
            va="center",
            fontsize=7,
            color="#444444",
        )

    ax_b.set_title("B. Displacement by hazard", loc="left", fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel C: countries
    # -------------------------------------------------------------------------

    bars = ax_c.barh(
        top.index,
        top.values,
        color=COUNTRY_BAR_COLOR,
        edgecolor="white",
        linewidth=0.4,
        height=0.65,
    )

    ax_c.set_xscale("log")
    ax_c.set_xlabel("Displaced persons")
    ax_c.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))

    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.tick_params(axis="y", labelsize=8)

    for bar, val in zip(bars, top.values):
        ax_c.text(
            val * 1.06,
            bar.get_y() + bar.get_height() / 2,
            _fmt(val),
            va="center",
            fontsize=7,
            color="#444444",
        )

    ax_c.set_title("C. Most affected countries", loc="left", fontweight="bold")

    # No figure-level suptitle.

    return fig


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

fig = plot_hazards(
    gdf_disaster, figure_column="Total figures", hazard_col="Hazard type"
)

fig.savefig("hazards_figure.pdf", dpi=300, bbox_inches="tight")
fig.savefig("hazards_figure.png", dpi=300, bbox_inches="tight")

plt.show()
print("✓ Saved: hazards_figure.pdf / .png")
