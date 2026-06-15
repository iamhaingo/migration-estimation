# %% [markdown]
# # IMR + HDI Lookup Table Builder
#
# Produces `/content/drive/MyDrive/parquet/imr_hdi_lookup.csv` consumed by
# `idp_sar1_tree.py` (Section 4 — Static Socioeconomic Features).
#
# Output schema (long format, one row per ISO3 × year):
#   iso3                  — ISO 3166-1 alpha-3, uppercase
#   year                  — integer
#   infant_mortality_rate — deaths per 1,000 live births  (World Bank SP.DYN.IMRT.IN)
#   hdi                   — Human Development Index 0–1   (UNDP HDR composite series)
#
# Sources:
#   IMR : World Bank Data API  (free, no auth)
#   HDI : UNDP Human Development Report composite indices CSV (public download)
#
# Run this notebook ONCE before running idp_sar1_tree.py.
# The output file is stable — re-run only if new IDMC events fall outside the
# covered year range or new HDR report is published.

# %%
!pip install requests pandas geopandas pyarrow -q

# %%
import datetime
import io
import time
import warnings

import geopandas as gpd
import pandas as pd
import requests
from google.colab import drive

warnings.filterwarnings("ignore")

# %%
# ─── Configuration ────────────────────────────────────────────────────────────
PARQUET_PATH = "/content/drive/MyDrive/parquet/disaster_rapid_onset_disaggregated.parquet"
OUT_CSV      = "/content/drive/MyDrive/parquet/imr_hdi_lookup.csv"

# World Bank: annual IMR 1990–present (end year resolved at runtime)
WB_INDICATOR = "SP.DYN.IMRT.IN"
WB_DATE_RANGE = f"1990:{datetime.date.today().year}"
WB_PER_PAGE   = 20000   # enough to get all countries × years in one call

# UNDP HDR composite indices — try newest report first, fall back to older editions.
# URL pattern: https://hdr.undp.org/sites/default/files/<REPORT_DIR>/<FILE>
# Last verified: 2026-05-27.
#   HDR25 (2025 report) is the latest published; covers HDI through 2023.
#   HDR26 URL returns 404 as of May 2026 — add it here once UNDP publishes it.
# Update this list when UNDP publishes a new HDR edition.
UNDP_HDR_CSV_URLS = [
    # 2025 report — latest confirmed available (covers 1990–2023)
    "https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Composite_indices_complete_time_series.csv",
    # 2023-24 report (fallback)
    "https://hdr.undp.org/sites/default/files/2023-24_HDR/HDR23-24_Composite_indices_complete_time_series.csv",
]
UNDP_HDR_CSV_FALLBACK = (
    "https://hdr.undp.org/sites/default/files/2025_HDR/"
    "HDR25_Statistical_Annex_HDI_Trends_Table.xlsx"
)

# %%
drive.mount("/content/drive")
print("Drive mounted.")

# %% [markdown]
# ## 1. Determine Required ISO3 Codes and Year Range
#
# We only fetch data for countries and years that actually appear in the IDMC
# dataset. This keeps the lookup table small and makes coverage gaps obvious.

# %%
gdf = gpd.read_parquet(PARQUET_PATH)
gdf["Start date"] = pd.to_datetime(gdf["Start date"])
gdf["event_year"] = gdf["Start date"].dt.year

required_iso3  = sorted(gdf["ISO3"].dropna().unique().tolist())
required_years = sorted(gdf["event_year"].dropna().astype(int).unique().tolist())
year_min, year_max = required_years[0], required_years[-1]

print(f"Required ISO3 codes : {len(required_iso3)}")
print(f"Year range          : {year_min} – {year_max}")
print("Sample ISO3s        :", required_iso3[:10])

# %% [markdown]
# ## 2. Fetch Infant Mortality Rate — World Bank API
#
# Indicator SP.DYN.IMRT.IN: Mortality rate, infant, per 1,000 live births.
# Fetched for ALL countries to support future events; filtered to required set later.

# %%
def _fetch_wb_indicator(indicator: str, date_range: str, per_page: int) -> pd.DataFrame:
    """
    Pull a World Bank indicator for all countries, all years in date_range.
    Returns long-format DataFrame with columns: iso3, year, value.
    Handles WB pagination automatically.
    """
    base = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
    rows, page, total_pages = [], 1, 1

    while page <= total_pages:
        resp = requests.get(base, params={
            "format": "json",
            "date": date_range,
            "per_page": per_page,
            "page": page,
        }, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        if not isinstance(payload, list) or len(payload) < 2:
            print(f"  [WARN] Unexpected WB response on page {page}; stopping.")
            break

        meta, data = payload[0], payload[1]
        total_pages = meta.get("pages", 1)

        if data:
            for rec in data:
                iso3  = rec.get("countryiso3code", "")
                year  = rec.get("date", "")
                value = rec.get("value")
                if iso3 and len(iso3) == 3 and year.isdigit() and value is not None:
                    rows.append({"iso3": iso3.upper(), "year": int(year), "value": float(value)})

        print(f"  Page {page}/{total_pages} — {len(rows):,} records so far")
        page += 1
        if page <= total_pages:
            time.sleep(0.3)  # be polite to WB API

    return pd.DataFrame(rows, columns=["iso3", "year", "value"])


print("Fetching IMR from World Bank API...")
imr_raw = _fetch_wb_indicator(WB_INDICATOR, WB_DATE_RANGE, WB_PER_PAGE)
imr_raw = imr_raw.rename(columns={"value": "infant_mortality_rate"})
print(f"IMR records fetched: {len(imr_raw):,}")
print(f"Countries covered  : {imr_raw['iso3'].nunique()}")
print(f"Year range         : {imr_raw['year'].min()} – {imr_raw['year'].max()}")

# %% [markdown]
# ## 3. Fetch HDI — UNDP Human Development Report
#
# The UNDP publishes a composite-indices CSV with HDI time series back to 1990.
# The file is in wide format (one column per year); we melt it to long format.

# %%
def _fetch_undp_hdi_csv(url: str) -> pd.DataFrame | None:
    """
    Download UNDP HDR composite indices CSV and return long-format HDI series.
    Returns None if the download fails.
    Expected wide-format columns: iso3, country, hdi_1990, hdi_1991, ..., hdi_YYYY
    """
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        df.columns = df.columns.str.lower().str.strip()
        print(f"  Downloaded: {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"  Columns sample: {list(df.columns[:8])}")
        return df
    except Exception as exc:
        print(f"  Failed: {exc}")
        return None


def _parse_undp_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt wide UNDP HDR DataFrame into long format.
    Handles two column naming conventions:
      - 'hdi_YYYY'  (standard composite indices CSV)
      - 'hdi (YYYY)' (older annex format)
    """
    # Identify iso3 column
    iso3_col = next((c for c in df.columns if "iso" in c and "3" in c), None)
    if iso3_col is None:
        iso3_col = "iso3" if "iso3" in df.columns else df.columns[0]

    # Find HDI year columns
    hdi_cols = [c for c in df.columns if c.startswith("hdi_") and c[4:].isdigit()]
    if not hdi_cols:
        # Try alternate pattern: 'hdi (1990)'
        import re
        hdi_cols = [c for c in df.columns if re.match(r"hdi\s*\(\d{4}\)", c)]

    if not hdi_cols:
        raise ValueError(f"Cannot find HDI year columns. Available: {list(df.columns)}")

    sub = df[[iso3_col] + hdi_cols].copy()
    sub = sub.rename(columns={iso3_col: "iso3"})
    sub["iso3"] = sub["iso3"].astype(str).str.upper().str.strip()

    long = sub.melt(id_vars="iso3", var_name="year_col", value_name="hdi")
    # Extract 4-digit year from column name
    long["year"] = long["year_col"].str.extract(r"(\d{4})").astype(int)
    long = long.drop(columns="year_col")
    long["hdi"] = pd.to_numeric(long["hdi"], errors="coerce")
    long = long.dropna(subset=["hdi"])
    long = long[long["iso3"].str.len() == 3]
    return long[["iso3", "year", "hdi"]].sort_values(["iso3", "year"]).reset_index(drop=True)


print("Fetching HDI from UNDP HDR...")
hdi_raw_df = None
for _url in UNDP_HDR_CSV_URLS:
    print(f"  Trying: {_url}")
    hdi_raw_df = _fetch_undp_hdi_csv(_url)
    if hdi_raw_df is not None:
        break

if hdi_raw_df is None:
    print("All CSV URLs failed. Trying fallback Excel file...")
    try:
        resp = requests.get(UNDP_HDR_CSV_FALLBACK, timeout=120)
        resp.raise_for_status()
        hdi_raw_df = pd.read_excel(io.BytesIO(resp.content), sheet_name=0, header=None)
        # Excel annex has a multi-row header; find the row with "iso3" or country codes
        # Heuristic: look for row where cell contains "iso3" case-insensitive
        for i, row in hdi_raw_df.iterrows():
            if any("iso" in str(v).lower() for v in row.values):
                hdi_raw_df.columns = [str(c).strip().lower() for c in hdi_raw_df.iloc[i]]
                hdi_raw_df = hdi_raw_df.iloc[i+1:].reset_index(drop=True)
                break
        print(f"  Fallback Excel loaded: {hdi_raw_df.shape}")
    except Exception as exc:
        print(f"  Fallback also failed: {exc}")
        hdi_raw_df = None

if hdi_raw_df is None:
    raise RuntimeError(
        "Could not download UNDP HDI data automatically.\n"
        "Manual fix: download the HDR composite indices CSV from\n"
        "  https://hdr.undp.org/data-center/human-development-index\n"
        "and upload it to Colab, then set hdi_raw_df = pd.read_csv('<path>')."
    )

hdi_long = _parse_undp_wide(hdi_raw_df)
print(f"HDI records parsed : {len(hdi_long):,}")
print(f"Countries covered  : {hdi_long['iso3'].nunique()}")
print(f"Year range         : {hdi_long['year'].min()} – {hdi_long['year'].max()}")

# %% [markdown]
# ## 3b. Last Year Covered by Each Dataset
#
# For each source, reports the most recent year that has at least one non-null
# value across all countries — this is the effective data horizon before
# nearest-year fallback kicks in.

# %%
imr_last = int(imr_raw.dropna(subset=["infant_mortality_rate"])["year"].max())
hdi_last = int(hdi_long.dropna(subset=["hdi"])["year"].max())

print("Last year with data in each source:")
print(f"  World Bank IMR  (SP.DYN.IMRT.IN) : {imr_last}")
print(f"  UNDP HDI                          : {hdi_last}")
print()

# Per-country last year — useful to spot countries that stop early
imr_last_by_country = (
    imr_raw.dropna(subset=["infant_mortality_rate"])
    .groupby("iso3")["year"].max()
    .rename("imr_last_year")
)
hdi_last_by_country = (
    hdi_long.dropna(subset=["hdi"])
    .groupby("iso3")["year"].max()
    .rename("hdi_last_year")
)
last_years = pd.concat([imr_last_by_country, hdi_last_by_country], axis=1)

# Flag countries whose last year lags behind the global maximum
imr_laggards = last_years[last_years["imr_last_year"] < imr_last].sort_values("imr_last_year")
hdi_laggards = last_years[last_years["hdi_last_year"] < hdi_last].sort_values("hdi_last_year")

if not imr_laggards.empty:
    print(f"Countries with IMR data ending before {imr_last}:")
    print(imr_laggards["imr_last_year"].to_string())
    print()
if not hdi_laggards.empty:
    print(f"Countries with HDI data ending before {hdi_last}:")
    print(hdi_laggards["hdi_last_year"].to_string())

# %% [markdown]
# ## 4. Merge IMR and HDI into a Single Lookup Table

# %%
# Full outer join so that rows with one source missing are retained with NaN
lut = pd.merge(
    imr_raw[["iso3", "year", "infant_mortality_rate"]],
    hdi_long[["iso3", "year", "hdi"]],
    on=["iso3", "year"],
    how="outer",
)
lut = lut.sort_values(["iso3", "year"]).reset_index(drop=True)
print(f"Merged lookup table: {len(lut):,} rows")
print(f"  infant_mortality_rate missing: {lut['infant_mortality_rate'].isna().sum():,}")
print(f"  hdi missing                  : {lut['hdi'].isna().sum():,}")

# %% [markdown]
# ## 5. Gap-Fill Within Each Country
#
# For each ISO3, linearly interpolate across years where one source has values
# but the other is missing. Extrapolation is intentionally NOT applied — the
# lookup function in idp_sar1_tree.py handles nearest-year fallback at query time.

# %%
def _interpolate_country(grp: pd.DataFrame) -> pd.DataFrame:
    """Linear interpolation within a country group; nearest-year fill at boundaries."""
    grp = grp.sort_values("year").copy()
    grp["infant_mortality_rate"] = (
        grp["infant_mortality_rate"]
        .interpolate(method="linear", limit_direction="both", limit_area="inside")
        .ffill().bfill()
    )
    grp["hdi"] = (
        grp["hdi"]
        .interpolate(method="linear", limit_direction="both", limit_area="inside")
        .ffill().bfill()
    )
    return grp


# Expand to a full year grid before interpolating so that years beyond the last
# published data point (e.g. 2025 when sources only reach 2023–2024) exist as
# NaN rows that ffill() can then fill with the nearest known value.
_target_years = list(range(max(1990, year_min - 5), year_max + 6))
_grid = pd.MultiIndex.from_product(
    [lut["iso3"].unique(), _target_years], names=["iso3", "year"]
)
lut = lut.set_index(["iso3", "year"]).reindex(_grid).reset_index()

lut = lut.groupby("iso3", group_keys=False).apply(_interpolate_country)
lut["year"] = lut["year"].astype(int)

print("After within-country linear interpolation:")
print(f"  infant_mortality_rate missing: {lut['infant_mortality_rate'].isna().sum():,}")
print(f"  hdi missing                  : {lut['hdi'].isna().sum():,}")

# %% [markdown]
# ## 6. Coverage Check Against Required IDMC Events

# %%
# Build the set of (ISO3, year) pairs needed by idp_sar1_tree.py
required_pairs = (
    gdf[["ISO3", "event_year"]]
    .dropna()
    .drop_duplicates()
    .rename(columns={"ISO3": "iso3", "event_year": "year"})
)
required_pairs["year"] = required_pairs["year"].astype(int)

lut_pairs = lut[lut[["iso3", "year"]].apply(tuple, axis=1).isin(
    set(zip(required_pairs["iso3"], required_pairs["year"]))
)]

# For each required pair, check what is available
coverage = required_pairs.merge(lut, on=["iso3", "year"], how="left")
n_required = len(coverage)
n_imr_ok   = coverage["infant_mortality_rate"].notna().sum()
n_hdi_ok   = coverage["hdi"].notna().sum()

print(f"\nCoverage over required (ISO3 × year) pairs:")
print(f"  Total required        : {n_required}")
print(f"  IMR available (exact) : {n_imr_ok}  ({100*n_imr_ok/n_required:.1f}%)")
print(f"  HDI available (exact) : {n_hdi_ok}  ({100*n_hdi_ok/n_required:.1f}%)")

# Report pairs with both features missing — nearest-year fallback will be used
both_missing = coverage[coverage["infant_mortality_rate"].isna() & coverage["hdi"].isna()]
if len(both_missing):
    print(f"\n  Pairs with BOTH features missing (nearest-year fallback will apply):")
    print(both_missing[["iso3", "year"]].to_string(index=False))

# Report pairs with only one feature missing
imr_only_missing = coverage[coverage["infant_mortality_rate"].isna() & coverage["hdi"].notna()]
hdi_only_missing = coverage[coverage["hdi"].isna() & coverage["infant_mortality_rate"].notna()]
if len(imr_only_missing):
    print(f"\n  Pairs with IMR missing (HDI present):")
    print(imr_only_missing[["iso3", "year"]].to_string(index=False))
if len(hdi_only_missing):
    print(f"\n  Pairs with HDI missing (IMR present):")
    print(hdi_only_missing[["iso3", "year"]].to_string(index=False))

# %% [markdown]
# ## 7. Trim and Save
#
# Keep only ISO3 codes that appear in the IDMC dataset (plus a ±5-year buffer
# for nearest-year fallback), to keep the file small.

# %%
# ISO3 filter: only countries in the IDMC data
lut_filtered = lut[lut["iso3"].isin(required_iso3)].copy()

# Year filter: year_min-5 to year_max+5 (buffer for interpolation at boundaries)
lut_filtered = lut_filtered[
    (lut_filtered["year"] >= max(1990, year_min - 5)) &
    (lut_filtered["year"] <= year_max + 5)
].copy()

lut_filtered = lut_filtered.sort_values(["iso3", "year"]).reset_index(drop=True)

print(f"Final lookup table: {len(lut_filtered):,} rows  |  {lut_filtered['iso3'].nunique()} countries")
print(lut_filtered.head(10).to_string(index=False))

lut_filtered.to_csv(OUT_CSV, index=False, float_format="%.6f")
print(f"\nSaved to {OUT_CSV}")

# %% [markdown]
# ## 8. Sanity Check — Spot-Check Known Values

# %%
# Spot-check a handful of well-known values against published figures.
# IMR values from World Bank; HDI values from UNDP 2025 report.
spot_checks = [
    ("NOR", 2022, "infant_mortality_rate", 1.8),   # Norway 2022 ≈ 1.8
    ("SOM", 2022, "infant_mortality_rate", 78.0),  # Somalia 2022 ≈ 78
    ("NOR", 2022, "hdi", 0.966),                   # Norway 2022 ≈ 0.966
    ("NER", 2022, "hdi", 0.394),                   # Niger 2022 ≈ 0.394
    ("BGD", 2010, "hdi", 0.554),                   # Bangladesh 2010 ≈ 0.554
]

print("Spot-check (tolerance ±10%):")
print(f"{'ISO3':6} {'Year':6} {'Feature':25} {'Expected':>10} {'Actual':>10} {'OK':>5}")
print("-" * 62)
for iso3, year, feat, expected in spot_checks:
    row = lut_filtered[(lut_filtered["iso3"] == iso3) & (lut_filtered["year"] == year)]
    actual = float(row[feat].iloc[0]) if not row.empty and not row[feat].isna().all() else float("nan")
    ok = abs(actual - expected) / expected < 0.10 if not pd.isna(actual) else False
    print(f"{iso3:6} {year:6} {feat:25} {expected:10.3f} {actual:10.3f} {'✓' if ok else '✗':>5}")

# %%
print("\nDone. imr_hdi_lookup.csv is ready for idp_sar1_tree.py.")
