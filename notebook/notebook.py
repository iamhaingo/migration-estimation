# %%
!pip install xgboost shap scikit-learn geopandas pyarrow geemap earthengine-api optuna optuna-integration cartopy rasterio statsmodels pycountry pycountry-convert -q

# %%
import os
import random
import glob
import hashlib
import time
import warnings
import datetime
import dataclasses
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import ee
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.special import boxcox as sp_boxcox, inv_boxcox
from scipy.stats import boxcox as boxcox_fit, gaussian_kde
from scipy.ndimage import gaussian_filter
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pycountry
import pycountry_convert as pc
import shap
import optuna
from optuna.integration import XGBoostPruningCallback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Patch
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
@contextmanager
def _banner(title: str = "", w: int = 55):
    print(f"\n{'═' * w}")
    if title:
        for ln in title.splitlines():
            print(f"  {ln}")
        print(f"{'─' * w}")
    try:
        yield
    finally:
        print(f"{'═' * w}")

# %% [markdown]
#  %% [markdown]

# %%
@dataclass(frozen=True)
class Config:
    gee_project:   str = "haingo-498815"
    parquet_path:  str = "/content/drive/MyDrive/parquet/disaster_rapid_onset_disaggregated.parquet"
    imr_hdi_csv:   str = "/content/drive/MyDrive/parquet/imr_hdi_lookup.csv"
    rwi_tif: str = "/content/drive/MyDrive/parquet/rwi_global.tif"
    gee_cache_dir:  str = "/content/drive/MyDrive/parquet"
    gee_part_prefix: str = "gee_features_part"
    gee_legacy_cache_csv: str = "/content/drive/MyDrive/parquet/gee_features_cache.csv"
    target_transform: str = "log1p"
    test_size:        float = 0.10
    n_target_bins:    int   = 10
    n_folds:         int   = 10
    random_state:    int   = 12061998
    n_boot:          int   = 500
    n_perm:          int   = 30
    n_optuna_trials: int   = 100
    shap_rf_trees:   int   = 200
    corr_threshold: float = 0.7
    vif_cutoff:     float = 5.0
    ntl_baseline_days:        int   = 30
    ntl_baseline_gap_days:    int   = 3
    ntl_acute_window_days:    int   = 10
    ntl_persistent_start_day: int   = 30
    ntl_persistent_end_day:   int   = 60
    ntl_lit_threshold:        float = 0.5
    ntl_outage_threshold:     float = 0.8
    sar_baseline_days:             int   = 365
    sar_match_orbit:               bool  = True
    sar_min_baseline_scenes:       int   = 10
    sar_event_buffer_days_flood:   int   = 3
    sar_event_buffer_days_cyclone: int   = 2
    sar_z_threshold:               float = -2
    sar_scale_m:                   int   = 30
    output_dir: str = "/content/drive/MyDrive/idp_pipeline_runs"
    show_figure_titles: bool = True  # in-figure titles on; set False for journal figures that rely on the caption
    gadm_gpkg_path:          str   = "/content/drive/MyDrive/parquet/gadm_410.gpkg"
    admin3_cache_parquet:    str   = "/content/drive/MyDrive/parquet/admin3_geometry_cache.parquet"
    gadm_simplify_tolerance: float = 0.005
    aoi_buffer_m:              int   = 20_000
    aoi_buffer_m_by_region: Dict[str, int] = dataclasses.field(default_factory=lambda: {
        "North America": 39_000,
        "South America": 31_000,
        "Africa":        26_000,
        "Asia":          20_000,
        "Europe":        13_000,
        "Oceania":       10_000,
    })
    ntl_band:                  str   = "Gap_Filled_DNB_BRDF_Corrected_NTL"
    ntl_scale_f:               float = 0.1
    permanent_water_threshold: int   = 10
    gee_tile_scale:            int   = 16
    gee_tile_scale_coarse:     int   = 1
    gee_batch_size:            int   = 25
    features: Tuple[str, ...] = (
        "flood_area_km2", "pop_exposed",
        "ntl_outage_pop_acute", "ntl_outage_pop_persistent", "disaster_type",
        "infant_mortality_rate", "hdi",
        "peak_precip_event", "accum_precip_event", "antecedent_precip_30d",
        "peak_wind_event",
        "mean_slope_deg", "mean_twi", "built_surface_m2",
        "rwi_mean", "rwi_std",
    )
    features_no_satellite: Tuple[str, ...] = (
        "disaster_type", "infant_mortality_rate", "hdi",
        "rwi_mean", "rwi_std",
    )

# %% [markdown]
#  %% [markdown]

# %%
@dataclass
class RawDataset:
    gdf: gpd.GeoDataFrame
@dataclass
class SplitDataset:
    df_train:          pd.DataFrame
    X_train:           np.ndarray
    y_train:           np.ndarray
    strat_labels:      pd.Series
    df_test:           pd.DataFrame
    X_test:            np.ndarray
    y_test:            np.ndarray
    groups:            np.ndarray = None
    feature_scaler:    StandardScaler = None
@dataclass
class ModelResults:
    name:          str
    cv_scores:     pd.DataFrame
    oof_preds:     np.ndarray
    fitted_models: list
    feature_names: List[str]
    shap_values:             Optional[np.ndarray] = None
    X_test:                  Optional[np.ndarray] = None
    X_train_imp:             Optional[np.ndarray] = None
    best_params:             Optional[dict]       = None
    test_pred:               Optional[np.ndarray] = None
    family:                  Optional[str]        = None

# %% [markdown]
#  %% [markdown]

# %%
_PERSON_CAP: float = float(np.expm1(20.0))
@dataclass(frozen=True)
class TargetTransform:
    name:        str
    short_label: str
    axis_label:  str
    lam:         Optional[float] = None
    def forward(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.lam is None:
            return np.log1p(y)
        return sp_boxcox(1.0 + y, self.lam)
    def inverse(self, z, *, clip: bool = True) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self.lam is None:
            if clip:
                z = np.clip(z, None, 20.0)
            return np.expm1(z)
        if clip:
            z = np.clip(z, None, sp_boxcox(1.0 + _PERSON_CAP, self.lam))
        return inv_boxcox(z, self.lam) - 1.0
def make_log1p_transform() -> TargetTransform:
    return TargetTransform(
        name="log1p", short_label="log1p",
        axis_label="log1p(displaced persons)", lam=None,
    )
def make_target_transform(cfg: Config, y_full) -> TargetTransform:
    kind = cfg.target_transform.lower()
    if kind == "log1p":
        return make_log1p_transform()
    if kind == "boxcox":
        y = np.asarray(y_full, dtype=float)
        y = y[np.isfinite(y)]
        _, lam = boxcox_fit(1.0 + y)
        lam = float(lam)
        print(f"  Target transform : Box-Cox  (MLE λ = {lam:.4f}, fit on 1 + target)")
        return TargetTransform(
            name="box-cox", short_label="box-cox",
            axis_label=f"box-cox(displaced persons, λ={lam:.3f})", lam=lam,
        )
    raise ValueError(
        f"Config.target_transform must be 'log1p' or 'boxcox', got {cfg.target_transform!r}"
    )
_TARGET_TRANSFORM: TargetTransform = make_log1p_transform()
def set_target_transform(t: TargetTransform) -> None:
    global _TARGET_TRANSFORM
    _TARGET_TRANSFORM = t
def target_transform() -> TargetTransform:
    return _TARGET_TRANSFORM

# %% [markdown]
#  %% [markdown]

# %%
_CONTINENT_CODE_TO_NAME: dict[str, str] = {
    "AF": "Africa", "AS": "Asia", "EU": "Europe",
    "NA": "North America", "SA": "South America", "OC": "Oceania",
}
def _build_iso3_region_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for country in pycountry.countries:
        try:
            code = pc.country_alpha2_to_continent_code(country.alpha_2)
            mapping[country.alpha_3] = _CONTINENT_CODE_TO_NAME.get(code, "Other")
        except (KeyError, AttributeError):
            pass
    return mapping
_ISO3_TO_REGION: dict[str, str] = _build_iso3_region_map()
def _aoi_buffer_m_for_iso3(iso3: str, cfg: Config) -> int:
    region = _ISO3_TO_REGION.get(str(iso3).upper().strip(), "Other")
    return cfg.aoi_buffer_m_by_region.get(region, cfg.aoi_buffer_m)
def _geometry_to_ee(geom, aoi_buffer_m: int):
    if geom.geom_type == "Point":
        return ee.Geometry.Point([geom.x, geom.y]).buffer(aoi_buffer_m)
    return ee.Geometry(mapping(geom))
GEE_EXPORT_COLUMNS = [
    "aoi_id", "event_id", "ISO3", "disaster_type", "start_date", "end_date",
    "flood_area_km2", "pop_exposed", "ntl_outage_pop_acute",
    "ntl_outage_pop_persistent", "peak_precip_event",
    "accum_precip_event", "antecedent_precip_30d", "peak_wind_event",
    "mean_slope_deg", "mean_twi", "built_surface_m2",
]
_GEE_AUDIT_COLS = frozenset({"event_id", "ISO3", "disaster_type", "start_date", "end_date"})
def _build_ee_feature(row, aoi_buffer_m: int) -> ee.Feature:
    return ee.Feature(
        _geometry_to_ee(row["geometry"], aoi_buffer_m),
        {
            "aoi_id":       str(row["aoi_id"]),
            "event_id":     str(row["event_id"]),
            "ISO3":         str(row["ISO3"]),
            "disaster_type": int(row["disaster_type"]),
            "start_date":   row["Start date"].strftime("%Y-%m-%d"),
            "end_date":     row["End date"].strftime("%Y-%m-%d"),
        },
    )
def _make_server_side_mapper(cfg: Config):
    def _mapper(feat):
        aoi        = feat.geometry()
        start_date = ee.Date(feat.getString("start_date"))
        end_date   = ee.Date(feat.getString("end_date"))
        end_date = ee.Date(ee.Algorithms.If(
            end_date.difference(start_date, "day").lte(0),
            start_date.advance(1, "day"),
            end_date,
        ))
        is_cyclone_sar = feat.getNumber("disaster_type").eq(1)
        sar_event_buffer = ee.Number(ee.Algorithms.If(
            is_cyclone_sar, cfg.sar_event_buffer_days_cyclone, cfg.sar_event_buffer_days_flood,
        ))
        def _s1_col(d_start, d_end):
            return (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(aoi).filterDate(d_start, d_end)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .select("VV")
            )
        sar_baseline_end   = start_date.advance(sar_event_buffer.multiply(-1), "day")
        baseline_start     = sar_baseline_end.advance(-cfg.sar_baseline_days, "day")
        event_all          = _s1_col(start_date, end_date.advance(sar_event_buffer, "day"))
        baseline_all       = _s1_col(baseline_start, sar_baseline_end)
        if cfg.sar_match_orbit:
            event_orbit = ee.Number(ee.Algorithms.If(
                event_all.size().gt(0),
                ee.Image(event_all.first()).getNumber("relativeOrbitNumber_start"),
                -1,
            ))
            event_m    = event_all.filter(ee.Filter.eq("relativeOrbitNumber_start", event_orbit))
            baseline_m = baseline_all.filter(ee.Filter.eq("relativeOrbitNumber_start", event_orbit))
            use_matched = baseline_m.size().gte(cfg.sar_min_baseline_scenes).And(event_m.size().gt(0))
            event_col        = ee.ImageCollection(ee.Algorithms.If(use_matched, event_m, event_all))
            sar_baseline_col = ee.ImageCollection(ee.Algorithms.If(use_matched, baseline_m, baseline_all))
        else:
            event_col, sar_baseline_col = event_all, baseline_all
        has_s1 = event_col.size().gt(0).And(sar_baseline_col.size().gt(0))
        baseline_mean    = sar_baseline_col.mean()
        baseline_mean_sq = sar_baseline_col.map(lambda img: img.pow(2)).mean()
        baseline_std     = baseline_mean_sq.subtract(baseline_mean.pow(2)).max(1e-6).sqrt()
        flood = (
            event_col.mean().subtract(baseline_mean).divide(baseline_std)
            .lt(cfg.sar_z_threshold).unmask(0)
            .where(
                ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
                .select("seasonality").gte(cfg.permanent_water_threshold),
                0,
            )
        )
        pop = (
            ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP")
            .filter(ee.Filter.date("2025-01-01", "2025-12-31"))
            .first().select("population_count")
        )
        def _safe(d, key):
            raw = d.get(key)
            return ee.Number(ee.Algorithms.If(raw, raw, 0))
        slope     = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
        flow_acc  = ee.Image("WWF/HydroSHEDS/15ACC").select("b1").max(1).rename("twi")
        tan_slope = slope.multiply(np.pi / 180).tan().max(1e-3).rename("twi")
        twi_img   = flow_acc.log().subtract(tan_slope.log())
        pop_density = pop.divide(100 * 100).max(0)
        sar_dict = (
            flood.multiply(ee.Image.pixelArea())
            .addBands(pop_density.resample("bilinear")
                      .multiply(ee.Image.pixelArea()).updateMask(flood))
            .reduceRegion(ee.Reducer.sum(), aoi, cfg.sar_scale_m, maxPixels=1e10, tileScale=cfg.gee_tile_scale)
        )
        flood_area_km2 = ee.Algorithms.If(has_s1, _safe(sar_dict, "VV").divide(1e6), None)
        pop_exposed    = ee.Algorithms.If(has_s1, _safe(sar_dict, "population_count"), None)
        def _ntl_col(d_start, d_end):
            def _mask_q(img):
                return img.updateMask(img.select("Mandatory_Quality_Flag").lte(1))
            return (
                ee.ImageCollection("NASA/VIIRS/002/VNP46A2")
                .filterDate(d_start, d_end)
                .select([cfg.ntl_band, "Mandatory_Quality_Flag"])
                .map(_mask_q).select(cfg.ntl_band)
                .map(lambda img: img.multiply(cfg.ntl_scale_f).copyProperties(img, ["system:time_start"]))
            )
        def _ntl_median(col):
            return ee.Image(ee.Algorithms.If(
                col.size().gt(0), col.median(),
                ee.Image.constant(0).rename(cfg.ntl_band).updateMask(ee.Image.constant(0)),
            ))
        ntl_baseline_end = start_date.advance(-cfg.ntl_baseline_gap_days, "day")
        ntl_baseline_col = _ntl_col(ntl_baseline_end.advance(-cfg.ntl_baseline_days, "day"), ntl_baseline_end)
        acute_col    = _ntl_col(start_date, start_date.advance(cfg.ntl_acute_window_days, "day"))
        persist_col  = _ntl_col(start_date.advance(cfg.ntl_persistent_start_day, "day"),
                                start_date.advance(cfg.ntl_persistent_end_day, "day"))
        ntl_base = _ntl_median(ntl_baseline_col)
        lit_mask = ntl_base.gte(cfg.ntl_lit_threshold)
        def _outage_pop(post_col):
            pct_normal = _ntl_median(post_col).divide(ntl_base.max(1e-6)).updateMask(lit_mask)
            outage     = pct_normal.lt(cfg.ntl_outage_threshold)
            return pop_density.multiply(ee.Image.pixelArea()).updateMask(outage)
        m500_dict = (
            _outage_pop(acute_col).rename("ntl_outage_pop_acute")
            .addBands(_outage_pop(persist_col).rename("ntl_outage_pop_persistent"))
            .addBands(twi_img)
            .reduceRegion(
                ee.Reducer.sum().combine(ee.Reducer.mean(), sharedInputs=True),
                aoi, 500, maxPixels=1e10, tileScale=cfg.gee_tile_scale_coarse)
        )
        has_ntl_base = ntl_baseline_col.size().gt(0)
        ntl_outage_pop_acute = ee.Algorithms.If(
            has_ntl_base.And(acute_col.size().gt(0)),
            _safe(m500_dict, "ntl_outage_pop_acute_sum"), None,
        )
        ntl_outage_pop_persistent = ee.Algorithms.If(
            has_ntl_base.And(persist_col.size().gt(0)),
            _safe(m500_dict, "ntl_outage_pop_persistent_sum"), None,
        )
        mean_twi = m500_dict.get("twi_mean")
        imerg_event = (
            ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
            .filterDate(start_date, end_date)
            .select("precipitation")
        )
        has_imerg = imerg_event.size().gt(0)
        era5_wind = (
            ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
            .filterDate(start_date, end_date)
            .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
        )
        has_era5 = era5_wind.size().gt(0)
        imerg_ante = (
            ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
            .filterDate(start_date.advance(-30, "day"), start_date).select("precipitation")
        )
        has_ante = imerg_ante.size().gt(0)
        def _band_or_masked(has, img, name):
            return ee.Image(ee.Algorithms.If(
                has, img,
                ee.Image.constant(0).rename(name).updateMask(ee.Image.constant(0)),
            ))
        wind_speed_max = era5_wind.map(
            lambda img: img.select("u_component_of_wind_10m")
            .hypot(img.select("v_component_of_wind_10m"))
            .rename("wind_speed_10m")
        ).max()
        km11_dict = (
            _band_or_masked(has_imerg, imerg_event.reduce(ee.Reducer.max()), "precipitation_max")
            .addBands(_band_or_masked(has_imerg, imerg_event.reduce(ee.Reducer.sum()), "precipitation_sum"))
            .addBands(_band_or_masked(has_era5, wind_speed_max, "wind_speed_10m"))
            .addBands(_band_or_masked(
                has_ante,
                imerg_ante.reduce(ee.Reducer.sum()).rename("antecedent_precip_30d"),
                "antecedent_precip_30d"))
            .reduceRegion(
                ee.Reducer.max().combine(ee.Reducer.mean(), sharedInputs=True),
                aoi, 11000, maxPixels=1e9, tileScale=cfg.gee_tile_scale_coarse)
        )
        peak_precip_event     = ee.Algorithms.If(has_imerg, km11_dict.get("precipitation_max_max"), None)
        accum_precip_event    = ee.Algorithms.If(has_imerg, km11_dict.get("precipitation_sum_max"), None)
        peak_wind_event       = ee.Algorithms.If(has_era5, km11_dict.get("wind_speed_10m_max"), None)
        antecedent_precip_30d = ee.Algorithms.If(has_ante, km11_dict.get("antecedent_precip_30d_mean"), None)
        mean_slope_deg = slope.reduceRegion(ee.Reducer.mean(), aoi, 30, maxPixels=1e9, tileScale=cfg.gee_tile_scale_coarse).get("slope")
        built_col  = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S").filter(
            ee.Filter.date("2025-01-01", "2025-12-31"))
        built_surface_m2 = ee.Algorithms.If(
            built_col.size().gt(0),
            built_col.first().select("built_surface")
            .reduceRegion(ee.Reducer.mean(), aoi, 100, maxPixels=1e9, tileScale=cfg.gee_tile_scale_coarse).get("built_surface"),
            None,
        )
        return feat.set({
            "flood_area_km2": flood_area_km2, "pop_exposed": pop_exposed,
            "ntl_outage_pop_acute": ntl_outage_pop_acute,
            "ntl_outage_pop_persistent": ntl_outage_pop_persistent,
            "peak_precip_event": peak_precip_event,
            "accum_precip_event": accum_precip_event, "antecedent_precip_30d": antecedent_precip_30d,
            "peak_wind_event": peak_wind_event,
            "mean_slope_deg": mean_slope_deg, "mean_twi": mean_twi, "built_surface_m2": built_surface_m2,
        })
    return _mapper
def _read_gee_parts(cfg: Config) -> pd.DataFrame:
    paths = sorted(glob.glob(
        os.path.join(cfg.gee_cache_dir, f"{cfg.gee_part_prefix}*.csv")))
    frames = []
    for path in paths:
        part = pd.read_csv(path)
        missing = [c for c in GEE_EXPORT_COLUMNS if c not in part.columns]
        if missing:
            print(f"  Skipping stale part {os.path.basename(path)} — "
                  f"missing column(s) {missing}")
            continue
        frames.append(part[GEE_EXPORT_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=GEE_EXPORT_COLUMNS)
    cache = pd.concat(frames, ignore_index=True)
    cache["aoi_id"] = cache["aoi_id"].astype(str)
    n_before = len(cache)
    cache = cache.drop_duplicates(subset="aoi_id", keep="first")
    if len(cache) < n_before:
        print(f"  Dropped {n_before - len(cache):,} duplicate cache rows "
              f"(resubmitted chunks)")
    return cache
def _backfill_legacy_gee_cache(gdf: gpd.GeoDataFrame, cfg: Config) -> None:
    out_path = os.path.join(cfg.gee_cache_dir, f"{cfg.gee_part_prefix}_legacy.csv")
    if os.path.exists(out_path) or not os.path.exists(cfg.gee_legacy_cache_csv):
        return
    legacy = pd.read_csv(cfg.gee_legacy_cache_csv)
    required = [c for c in GEE_EXPORT_COLUMNS if c != "aoi_id"] + ["row_index"]
    missing = [c for c in required if c not in legacy.columns]
    if missing:
        print(f"  Legacy cache missing column(s) {missing} — not backfilling")
        return
    legacy["row_index"] = legacy["row_index"].astype(gdf.index.dtype)
    legacy = legacy.set_index("row_index")
    legacy = legacy.loc[~legacy.index.duplicated(keep="first")]
    common = legacy.index.intersection(gdf.index)
    legacy = legacy.loc[common]
    ok = legacy["event_id"].astype(str).eq(gdf.loc[common, "event_id"].astype(str))
    if (~ok).any():
        print(f"  Legacy cache: {int((~ok).sum()):,} rows fail the event_id "
              f"check (upstream data changed) — dropped, will re-extract")
    part = legacy.loc[ok].copy()
    part.insert(0, "aoi_id", gdf.loc[part.index, "aoi_id"])
    part[GEE_EXPORT_COLUMNS].to_csv(out_path, index=False)
    print(f"  Legacy cache backfilled: {len(part):,} rows → "
          f"{os.path.basename(out_path)}")
def _wait_for_gee_tasks(tasks: List[Tuple[str, "ee.batch.Task"]]) -> None:
    pending = dict(tasks)
    last_state: Dict[str, str] = {}
    failed: Dict[str, str] = {}
    delay = 15.0
    while pending:
        for key, task in list(pending.items()):
            status = task.status()
            state = status["state"]
            if state != last_state.get(key):
                print(f"  [{time.strftime('%H:%M:%S')}] chunk {key}: {state}")
                last_state[key] = state
            if state == "COMPLETED":
                pending.pop(key)
            elif state in ("FAILED", "CANCELLED"):
                failed[key] = status.get("error_message", state)
                pending.pop(key)
        if pending:
            time.sleep(delay)
            delay = min(delay * 1.5, 60.0)
    if failed:
        raise RuntimeError(
            f"{len(failed)} GEE export chunk(s) failed: {failed}. "
            f"Completed chunks are cached — re-run to export only the rest."
        )
def extract_gee_features(gdf: gpd.GeoDataFrame, cfg: Config, export_folder: str = "parquet") -> pd.DataFrame:
    _backfill_legacy_gee_cache(gdf, cfg)
    cache = _read_gee_parts(cfg)
    todo = gdf.loc[~gdf["aoi_id"].isin(set(cache["aoi_id"]))]
    print(f"GEE cache: {len(gdf) - len(todo):,} / {len(gdf):,} AOIs cached, "
          f"{len(todo):,} to extract")
    if todo.empty:
        return cache
    mapper = _make_server_side_mapper(cfg)
    tasks: List[Tuple[str, "ee.batch.Task"]] = []
    expected_paths: List[str] = []
    for lo in range(0, len(todo), cfg.gee_batch_size):
        chunk = todo.iloc[lo:lo + cfg.gee_batch_size]
        chunk_key = hashlib.sha1(
            "|".join(sorted(chunk["aoi_id"])).encode()).hexdigest()[:12]
        fc = ee.FeatureCollection([
            _build_ee_feature(row, _aoi_buffer_m_for_iso3(row["ISO3"], cfg))
            for _, row in chunk.iterrows()
        ])
        task = ee.batch.Export.table.toDrive(
            collection=fc.map(mapper),
            description=f"gee_features_{chunk_key}",
            folder=export_folder,
            fileNamePrefix=f"{cfg.gee_part_prefix}_{chunk_key}",
            fileFormat="CSV", selectors=GEE_EXPORT_COLUMNS,
        )
        task.start()
        tasks.append((chunk_key, task))
        expected_paths.append(os.path.join(
            cfg.gee_cache_dir, f"{cfg.gee_part_prefix}_{chunk_key}.csv"))
        print(f"  Chunk {chunk_key}: {len(chunk):,} AOIs submitted "
              f"(task {task.id})")
    _wait_for_gee_tasks(tasks)
    deadline = time.time() + 180
    while time.time() < deadline:
        if all(os.path.exists(p) for p in expected_paths):
            break
        time.sleep(10)
    still_absent = [p for p in expected_paths if not os.path.exists(p)]
    if still_absent:
        print(f"  Warning: {len(still_absent)} part file(s) not visible yet "
              f"on the Drive mount — their rows will read as missing")
    cache = _read_gee_parts(cfg)
    n_missing = int((~gdf["aoi_id"].isin(set(cache["aoi_id"]))).sum())
    if n_missing:
        print(f"  Warning: {n_missing:,} AOIs still missing after export "
              f"(GEE returned no row for them)")
    return cache

# %% [markdown]
#  %% [markdown]

# %%
def load_raw_data(cfg: Config) -> RawDataset:
    gdf = gpd.read_parquet(cfg.parquet_path)
    with _banner():
        print(f"  Data loaded: {cfg.parquet_path.split('/')[-1]}")
        print(f"  Rows        : {len(gdf):,}")
        print(f"  Events      : {gdf['Event ID'].nunique():,} unique")
        print(f"  Countries   : {gdf['ISO3'].nunique():,}")
        print(f"  Date range  : {gdf['Start date'].min().date()} → {gdf['Start date'].max().date()}")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = (
        gdf
        .assign(
            **{"Start date": pd.to_datetime(gdf["Start date"]),
               "End date":   pd.to_datetime(gdf["End date"])},
            event_id   = gdf["Event ID"].astype(str),
            event_year = pd.to_datetime(gdf["Start date"]).dt.year,
        )
        .dropna(subset=["geometry", "Start date", "End date", "distributed_figure"])
        .copy()
    )
    print(f"After dropping rows with missing geometry / dates / target: {len(gdf):,} rows retained")
    return RawDataset(gdf=gdf)
def encode_disaster_type(ds: RawDataset) -> RawDataset:
    hazard_lower = ds.gdf["Hazard type"].str.lower().str.strip()
    gdf = ds.gdf.assign(
        disaster_type=np.where(
            hazard_lower.str.contains(r"cyclone|storm|typhoon|hurricane", regex=True), 1, 0
        ).astype(int)
    )
    print("disaster_type distribution:")
    print(gdf["disaster_type"].value_counts())
    return RawDataset(gdf=gdf)
def attach_admin3_geometry(ds: RawDataset, cfg: Config) -> RawDataset:
    with _banner("Attaching GADM Admin3 polygons to event geometries"):
        if os.path.exists(cfg.admin3_cache_parquet):
            print(f"  Cache found — loading from {cfg.admin3_cache_parquet.split('/')[-1]}")
            cache = gpd.read_parquet(cfg.admin3_cache_parquet)
            cache.index = cache.index.astype(ds.gdf.index.dtype)
            gdf = ds.gdf.copy()
            aligned = cache.reindex(gdf.index)
            gdf["admin3_gid"]  = aligned["admin3_gid"]
            gdf["admin3_name"] = aligned["admin3_name"]
            poly_mask = aligned.geometry.notna() & (aligned.geometry.geom_type != "Point")
            gdf.loc[poly_mask[poly_mask].index, "geometry"] = aligned.loc[poly_mask, "geometry"]
            print(f"  Loaded {poly_mask.sum():,} polygon geometries from cache")
            return RawDataset(gdf=gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326"))
        layers = gpd.list_layers(cfg.gadm_gpkg_path)
        layer_name = layers["name"].iloc[0]
        print(f"  Layer: '{layer_name}'  |  file: {cfg.gadm_gpkg_path.split('/')[-1]}")
        admin3 = gpd.read_file(cfg.gadm_gpkg_path, layer=layer_name)
        admin3 = (
            admin3[admin3["GID_3"].fillna("").str.strip().ne("")]
            .drop_duplicates(subset="GID_3")
            [["GID_3", "NAME_3", "geometry"]]
            .copy()
        )
        if admin3.crs is None:
            admin3 = admin3.set_crs("EPSG:4326")
        elif admin3.crs.to_epsg() != 4326:
            admin3 = admin3.to_crs("EPSG:4326")
        if cfg.gadm_simplify_tolerance > 0:
            admin3["geometry"] = admin3.geometry.simplify(
                cfg.gadm_simplify_tolerance, preserve_topology=True
            )
        print(f"  Admin3 polygons loaded: {len(admin3):,}")
        gdf = ds.gdf.copy()
        point_mask = gdf.geometry.geom_type == "Point"
        n_pts = int(point_mask.sum())
        gdf["admin3_gid"]  = None
        gdf["admin3_name"] = None
        if n_pts == 0:
            print("  No Point geometries — nothing to attach")
            return RawDataset(gdf=gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326"))
        events_pts = gpd.GeoDataFrame(
            geometry=gdf.geometry[point_mask].values,
            crs="EPSG:4326",
            index=gdf.index[point_mask],
        )
        joined = gpd.sjoin(
            events_pts,
            admin3[["GID_3", "NAME_3", "geometry"]],
            how="left",
            predicate="within",
        )
        joined = joined.loc[~joined.index.duplicated(keep="first")]
        poly_lookup = admin3.set_index("GID_3")["geometry"].to_dict()
        matched_mask = joined["GID_3"].notna()
        new_geom = gdf.geometry.copy()
        new_geom.loc[joined.index[matched_mask]] = (
            joined.loc[matched_mask, "GID_3"].map(poly_lookup)
        )
        gdf["geometry"] = new_geom
        gdf.loc[joined.index, "admin3_gid"]  = joined["GID_3"]
        gdf.loc[joined.index, "admin3_name"] = joined["NAME_3"]
        n_matched = int(matched_mask.sum())
        print(f"  Point events        : {n_pts:,}")
        print(f"  Admin3 matched      : {n_matched:,}  ({100 * n_matched / n_pts:.1f}%)")
        print(f"  Region-sized buffer : {n_pts - n_matched:,}")
        cache_gdf = gpd.GeoDataFrame(
            {"admin3_gid": gdf["admin3_gid"], "admin3_name": gdf["admin3_name"]},
            geometry=gdf.geometry,
            crs="EPSG:4326",
            index=gdf.index,
        )
        cache_gdf.to_parquet(cfg.admin3_cache_parquet, index=True)
        print(f"  Cache saved → {cfg.admin3_cache_parquet.split('/')[-1]}")
        return RawDataset(gdf=gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326"))
def stamp_aoi_ids(ds: RawDataset, cfg: Config) -> RawDataset:
    gdf = ds.gdf.copy()
    def _aoi_id(row) -> str:
        buffer_m = (_aoi_buffer_m_for_iso3(row["ISO3"], cfg)
                    if row["geometry"].geom_type == "Point" else 0)
        key = (f"{row['event_id']}|{row['Start date']:%Y-%m-%d}|"
               f"{row['End date']:%Y-%m-%d}|{buffer_m}").encode()
        return hashlib.sha1(key + row["geometry"].wkb).hexdigest()[:16]
    gdf["aoi_id"] = [_aoi_id(row) for _, row in gdf.iterrows()]
    dup = gdf.groupby("aoi_id").cumcount()
    if (dup > 0).any():
        print(f"  {int((dup > 0).sum()):,} duplicate AOI rows suffixed")
        gdf.loc[dup > 0, "aoi_id"] += "-" + dup[dup > 0].astype(str)
    assert gdf["aoi_id"].is_unique
    print(f"AOI ids stamped: {len(gdf):,} rows")
    return RawDataset(gdf=gdf)
def merge_gee_features(ds: RawDataset, gee_df: pd.DataFrame) -> RawDataset:
    gee_df = gee_df[[c for c in GEE_EXPORT_COLUMNS if c not in _GEE_AUDIT_COLS]].copy()
    gee_df["aoi_id"] = gee_df["aoi_id"].astype(str)
    assert gee_df["aoi_id"].is_unique, \
        "duplicate aoi_id in GEE cache (overlapping chunks?)"
    gdf = ds.gdf.merge(gee_df, on="aoi_id", how="left", validate="one_to_one")
    assert len(gdf) == len(ds.gdf)
    n_matched = int(ds.gdf["aoi_id"].isin(set(gee_df["aoi_id"])).sum())
    print(f"  Cache coverage: {n_matched:,} / {len(gdf):,} AOIs matched")
    for col in ["flood_area_km2", "pop_exposed",
                "ntl_outage_pop_acute", "ntl_outage_pop_persistent"]:
        n_zeros = int((gdf[col] == 0).sum())
        n_nan   = int(gdf[col].isna().sum())
        print(f"  {col}: {n_zeros:,} true zeros kept, {n_nan:,} NaN (no imagery)")
    print(f"After GEE merge: {len(gdf):,} rows")
    return RawDataset(gdf=gdf)
def merge_imr_hdi(ds: RawDataset, cfg: Config) -> RawDataset:
    imr_hdi = pd.read_csv(cfg.imr_hdi_csv)
    imr_hdi.columns = imr_hdi.columns.str.strip().str.lower()
    imr_hdi = (
        imr_hdi
        .rename(columns={"iso3": "ISO3_key", "year": "year_key"})
        .assign(ISO3_key=lambda d: d["ISO3_key"].str.upper().str.strip(),
                year_key=lambda d: d["year_key"].astype(int))
    )
    gdf = (
        ds.gdf
        .assign(ISO3_upper=lambda d: d["ISO3"].str.upper().str.strip(),
                event_year_int=lambda d: d["event_year"].astype(int))
        .merge(
            imr_hdi[["ISO3_key", "year_key", "infant_mortality_rate", "hdi"]],
            left_on=["ISO3_upper", "event_year_int"],
            right_on=["ISO3_key", "year_key"],
            how="left",
            validate="many_to_one",
        )
        .drop(columns=["ISO3_key", "year_key", "ISO3_upper", "event_year_int"])
    )
    assert len(gdf) == len(ds.gdf)
    print(f"Rows missing IMR: {gdf['infant_mortality_rate'].isna().sum():,}  "
          f"HDI: {gdf['hdi'].isna().sum():,}")
    return RawDataset(gdf=gdf)
def _rwi_stats(src, geom, nodata) -> Tuple[float, float]:
    try:
        data, _ = rio_mask(
            src, [mapping(geom)], crop=True, all_touched=True, nodata=nodata,
        )
    except (ValueError, rasterio.errors.RasterioError):
        return np.nan, np.nan
    vals = data[0].astype(np.float64)
    if nodata is not None:
        vals[vals == nodata] = np.nan
    vals = vals[np.isfinite(vals)]
    return (float(np.mean(vals))        if len(vals) > 0 else np.nan,
            float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan)
def merge_rwi(ds: RawDataset, cfg: Config) -> RawDataset:
    gdf = ds.gdf.copy()
    stats_rows: list[Tuple[float, float]] = []
    with rasterio.open(cfg.rwi_tif) as src:
        for _, row in gdf.iterrows():
            geom = row["geometry"]
            if geom is None or geom.is_empty:
                stats_rows.append((np.nan, np.nan))
                continue
            if geom.geom_type == "Point":
                geom = geom.buffer(_aoi_buffer_m_for_iso3(row["ISO3"], cfg) / 111_000)
            stats_rows.append(_rwi_stats(src, geom, src.nodata))
    gdf[["rwi_mean", "rwi_std"]] = np.array(stats_rows, dtype=float)
    print(f"Rows missing RWI mean: {gdf['rwi_mean'].isna().sum():,}  "
          f"std: {gdf['rwi_std'].isna().sum():,}")
    return RawDataset(gdf=gdf)
def engineer_features(ds: RawDataset, cfg: Config) -> RawDataset:
    gdf = ds.gdf.copy()
    for col in ["ntl_outage_pop_acute", "ntl_outage_pop_persistent"]:
        gdf[col] = np.log1p(gdf[col])
    for col in ["accum_precip_event", "antecedent_precip_30d"]:
        gdf[col] = gdf[col] * 0.5
    transform = make_target_transform(cfg, gdf["distributed_figure"].values)
    set_target_transform(transform)
    gdf["target"] = transform.forward(gdf["distributed_figure"].values)
    print("Feature set ready:")
    print(gdf[list(cfg.features) + ["target"]].describe().T[["count", "mean", "std", "min", "max"]])
    return RawDataset(gdf=gdf)
def split_data(ds: RawDataset, cfg: Config) -> SplitDataset:
    gdf = ds.gdf.reset_index(drop=True)
    groups_all = pd.to_datetime(gdf["Start date"]).dt.strftime("%Y-%m")
    q_bins_all = pd.qcut(gdf["target"], q=cfg.n_target_bins,
                         labels=False, duplicates="drop")
    train_idx, test_idx = train_test_split(
        np.arange(len(gdf)), test_size=cfg.test_size,
        stratify=q_bins_all, random_state=cfg.random_state)
    df_train = gdf.iloc[train_idx].copy().reset_index(drop=True)
    df_test  = gdf.iloc[test_idx].copy().reset_index(drop=True)
    groups_train = groups_all.iloc[train_idx].to_numpy()
    features = list(cfg.features)
    y_train  = df_train["target"].values
    y_test   = df_test["target"].values
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(df_train[features].values)
    X_test  = feature_scaler.transform(df_test[features].values)
    strat_labels = pd.qcut(df_train["target"], q=5, labels=False, duplicates="drop")
    train_grp, test_grp = set(groups_train), set(groups_all.iloc[test_idx])
    with _banner():
        print("  Split method     : target-stratified random hold-out (ungrouped)")
        print(f"  Target bins      : {cfg.n_target_bins}  |  test fraction: {cfg.test_size:.0%}")
        print(f"  CV grouping      : month-year (month-level) — "
              f"{groups_all.nunique():,} groups, applied to CV folds only")
        print(f"  Train∩test groups: {len(train_grp & test_grp)}  "
              f"(shared by design — test is independent of grouping)")
        print(f"  Training set     : {len(df_train):,} rows")
        print(f"  Test set         : {len(df_test):,} rows")
    return SplitDataset(
        df_train=df_train, X_train=X_train, y_train=y_train,
        strat_labels=strat_labels,
        df_test=df_test, X_test=X_test, y_test=y_test,
        groups=groups_train,
        feature_scaler=feature_scaler,
    )
def _collinearity_screen(
    X: np.ndarray, feature_names: List[str],
    corr_threshold: float, vif_cutoff: float,
) -> Tuple[List[str], Dict]:
    names: List[str] = list(feature_names)
    corr = pd.DataFrame(X, columns=names).corr(method="spearman").abs()
    flagged = sorted(
        [(names[i], names[j], float(corr.iloc[i, j]))
         for i in range(len(names))
         for j in range(i + 1, len(names))
         if corr.iloc[i, j] > corr_threshold],
        key=lambda t: -t[2])
    dropped: List[Tuple[str, float]] = []
    while True:
        vifs  = {names[i]: variance_inflation_factor(X, i) for i in range(len(names))}
        worst = max(vifs, key=vifs.get)
        if vifs[worst] < vif_cutoff:
            break
        dropped.append((worst, float(vifs[worst])))
        idx = names.index(worst)
        names.pop(idx)
        X = np.delete(X, idx, axis=1)
    info = {"flagged_pairs": flagged, "dropped": dropped,
            "final_vifs": {k: float(v) for k, v in vifs.items()}}
    return names, info
def drop_correlated_features(
    df_train: pd.DataFrame,
    feature_names: List[str],
    corr_threshold: float = 0.7,
    vif_cutoff: float = 5.0,
) -> Tuple[List[str], Dict]:
    X = SimpleImputer(strategy="median").fit_transform(
        df_train[feature_names].values).astype(float)
    names, info = _collinearity_screen(X, feature_names, corr_threshold, vif_cutoff)
    W = 55
    with _banner(f"Collinearity screen  |Spearman ρ| > {corr_threshold}", w=W):
        if info["flagged_pairs"]:
            for a, b, r in info["flagged_pairs"]:
                print(f"  {a:<30s} ↔  {b:<24s}  ρ = {r:.3f}")
        else:
            print("  No pairs exceed threshold.")
        print(f"{'─' * W}")
        print(f"  VIF iterative removal  (cutoff = {vif_cutoff})")
        print(f"{'─' * W}")
        if not info["dropped"]:
            print("  No features dropped — all VIF below cutoff.")
        else:
            for d, v in info["dropped"]:
                print(f"  Drop  '{d}'  VIF = {v:.2f}")
            print(f"  Dropped : {[d for d, _ in info['dropped']]}")
        print(f"  Retained ({len(names)}): {names}")
    return names, info

# %% [markdown]
#  %% [markdown]

# %%
def _median_impute(X: np.ndarray) -> np.ndarray:
    return SimpleImputer(strategy="median").fit_transform(X)
def _get_cv_splits(split: SplitDataset, cfg: Config) -> list:
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state)
    return list(sgkf.split(split.X_train, split.strat_labels, groups=split.groups))
def _rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5
def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae":  mean_absolute_error(y_true, y_pred),
        "r2":   r2_score(y_true, y_pred),
    }
def _fit_fold(est, X_tr, y_tr, X_va, y_va):
    if isinstance(est, xgb.XGBRegressor):
        est.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        est.fit(X_tr, y_tr)
    return est
def _run_cv(make_est: Callable, X: np.ndarray, y: np.ndarray, folds: list,
            verbose: bool = True, w: int = 55) -> Tuple[pd.DataFrame, np.ndarray, list]:
    oof          = np.full(len(y), np.nan)
    scores: list = []
    fitted: list = []
    for fold_idx, (tr, va) in enumerate(folds):
        est     = _fit_fold(make_est(), X[tr], y[tr], X[va], y[va])
        pred    = est.predict(X[va])
        oof[va] = pred
        m       = _fold_metrics(y[va], pred)
        scores.append(m)
        fitted.append(est)
        if verbose:
            print(f"  Fold {fold_idx + 1}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")
    cv = pd.DataFrame(scores)
    if verbose:
        print(f"{'─' * w}")
        print(f"  Mean    RMSE={cv['rmse'].mean():.4f}  MAE={cv['mae'].mean():.4f}  R²={cv['r2'].mean():.4f}")
    return cv, oof, fitted
def ensemble_test_pred(r: ModelResults, split: SplitDataset) -> np.ndarray:
    if r.test_pred is None:
        X_eval = r.X_test if r.X_test is not None else split.X_test
        r.test_pred = np.mean([m.predict(X_eval) for m in r.fitted_models], axis=0)
    return r.test_pred
def _get_iso3_cv_splits(split: SplitDataset, cfg: Config) -> list:
    iso3     = split.df_train["ISO3"].astype(str).to_numpy()
    n_splits = min(cfg.n_folds, int(pd.unique(iso3).size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=cfg.random_state)
    return list(sgkf.split(split.X_train, split.strat_labels, groups=iso3))
def cross_validate_iso3(
    r: ModelResults, split: SplitDataset, cfg: Config, folds: list,
) -> Tuple[pd.DataFrame, dict]:
    X = StandardScaler().fit_transform(split.df_train[r.feature_names].values)
    y = split.y_train
    cv, oof, _ = _run_cv(lambda: clone(r.fitted_models[0]), X, y, folds, verbose=False)
    mask = np.isfinite(oof)
    return cv, _fold_metrics(y[mask], oof[mask])
def _design_matrices(
    split: SplitDataset, cfg: Config, feature_names: Optional[List[str]],
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    if feature_names is None:
        return split.X_train, None, list(cfg.features)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(split.df_train[feature_names].values)
    X_te   = scaler.transform(split.df_test[feature_names].values)
    return X_tr, X_te, list(feature_names)
def train_ridge_baseline(
    split: SplitDataset, cfg: Config,
    feature_names: Optional[List[str]] = None,
    model_name: Optional[str] = None,
) -> ModelResults:
    name = model_name if model_name is not None else (
        "Ridge (no satellite)" if feature_names is not None else "Ridge")
    X_tr, X_te, feature_names = _design_matrices(split, cfg, feature_names)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("ridge",   Ridge(alpha=1.0)),
    ])
    folds = _get_cv_splits(split, cfg)
    with _banner(f"{name} — {cfg.n_folds}-fold StratifiedGroupKFold"):
        cv, oof, fitted = _run_cv(lambda: clone(pipe), X_tr, split.y_train, folds)
    return ModelResults(name, cv, oof, fitted, feature_names,
                        X_test=X_te, X_train_imp=_median_impute(X_tr),
                        best_params={"alpha": 1.0}, family="ridge")
def train_xgboost(
    split: SplitDataset, cfg: Config,
    feature_names: Optional[List[str]] = None,
    model_name: Optional[str] = None,
) -> ModelResults:
    name = model_name if model_name is not None else (
        "XGBoost (no satellite)" if feature_names is not None else "XGBoost")
    X_tr, X_te, feature_names = _design_matrices(split, cfg, feature_names)
    folds = _get_cv_splits(split, cfg)
    with _banner(f"{name} — Optuna tuning ({cfg.n_optuna_trials} trials, "
                 f"{cfg.n_folds}-fold CV)"):
        def objective(trial):
            params = dict(
                objective        = "reg:squarederror",
                max_depth        = trial.suggest_int("max_depth", 3, 8),
                learning_rate    = trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
                subsample        = trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.3, 1.0),
                min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
                gamma            = trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                reg_lambda       = trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
                eval_metric      = "rmse",
                tree_method      = "hist",
            )
            dtrain = xgb.DMatrix(X_tr, label=split.y_train)
            pruning_callback = XGBoostPruningCallback(trial, "test-rmse")
            cv_results = xgb.cv(
                params, dtrain, num_boost_round=1000, folds=folds,
                early_stopping_rounds=30, callbacks=[pruning_callback],
                seed=cfg.random_state, verbose_eval=False,
            )
            return float(cv_results["test-rmse-mean"].iloc[-1])
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                multivariate=True, n_startup_trials=20, seed=cfg.random_state,
            ),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        )
        study.optimize(objective, n_trials=cfg.n_optuna_trials, show_progress_bar=True, n_jobs=1)
        best = {
            "objective":    "reg:squarederror",
            "n_estimators": 1000,
            **study.best_params,
        }
        print(f"  Best params: {best}")
        print(f"  Best mean CV RMSE: {study.best_value:.4f}")
        print(f"\n  {name} — {cfg.n_folds}-fold CV with best params")
        print(f"{'─' * 55}")
        cv, oof, fitted = _run_cv(
            lambda: xgb.XGBRegressor(**best, early_stopping_rounds=30,
                                     random_state=cfg.random_state,
                                     tree_method="hist"),
            X_tr, split.y_train, folds)
    best_n      = int(np.mean([m.best_iteration + 1 for m in fitted]))
    shap_params = {**best, "n_estimators": best_n}
    return ModelResults(name, cv, oof, fitted, feature_names,
                        X_test=X_te, X_train_imp=_median_impute(X_tr),
                        best_params=shap_params, family="xgb")
def train_random_forest(
    split: SplitDataset, cfg: Config,
    feature_names: Optional[List[str]] = None,
    model_name: Optional[str] = None,
) -> ModelResults:
    name = model_name if model_name is not None else (
        "Random Forest (no satellite)" if feature_names is not None else "Random Forest")
    X_tr, X_te, feature_names = _design_matrices(split, cfg, feature_names)
    folds = _get_cv_splits(split, cfg)
    def _make_pipe(params: dict) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(
                **params, random_state=cfg.random_state, n_jobs=-1)),
        ])
    with _banner(f"{name} — Fast Optuna tuning ({cfg.n_optuna_trials} trials, "
                 "warm_start, Hyperband)"):
        inner_tr, inner_va = next(
            StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
            .split(X_tr, split.strat_labels, groups=split.groups))
        X_t, X_v = X_tr[inner_tr], X_tr[inner_va]
        y_t, y_v = split.y_train[inner_tr], split.y_train[inner_va]
        imp = SimpleImputer(strategy="median")
        X_t_imp = imp.fit_transform(X_t)
        X_v_imp = imp.transform(X_v)
        def objective(trial):
            params = dict(
                max_depth         = trial.suggest_int("max_depth", 3, 20),
                max_features      = trial.suggest_float("max_features", 0.3, 1.0),
                min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
            )
            max_est = 200
            step_size = 25
            rf = RandomForestRegressor(
                **params,
                n_estimators=step_size,
                warm_start=True,
                random_state=cfg.random_state,
                n_jobs=-1,
            )
            for step in range(step_size, max_est + 1, step_size):
                rf.set_params(n_estimators=step)
                rf.fit(X_t_imp, y_t)
                pred = rf.predict(X_v_imp)
                rmse = _rmse(y_v, pred)
                trial.report(rmse, step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return rmse
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=20, seed=cfg.random_state),
            pruner=optuna.pruners.HyperbandPruner(min_resource=25, max_resource=200, reduction_factor=2),
        )
        study.optimize(objective, n_trials=cfg.n_optuna_trials, show_progress_bar=True, n_jobs=1)
        best = dict(study.best_params)
        best["n_estimators"] = 500
        print(f"  Best params: {best}")
        print(f"  Best inner validation RMSE: {study.best_value:.4f}")
        print(f"\n  {name} — {cfg.n_folds}-fold CV with best params")
        print(f"{'─' * 55}")
        cv, oof, fitted = _run_cv(lambda: _make_pipe(best), X_tr, split.y_train, folds)
    return ModelResults(name, cv, oof, fitted, feature_names,
                        X_test=X_te, X_train_imp=_median_impute(X_tr),
                        best_params=best, family="rf")
_FAMILIES: Dict[str, Tuple[str, Callable[..., ModelResults]]] = {
    "ridge": ("Ridge",         train_ridge_baseline),
    "rf":    ("Random Forest", train_random_forest),
    "xgb":   ("XGBoost",       train_xgboost),
}
def _suite(models: Dict[str, ModelResults]) -> Tuple[ModelResults, ...]:
    return tuple(models[k] for fam in _FAMILIES for k in (f"{fam}_ns", fam))
def evaluate_models(*results: ModelResults, split: SplitDataset) -> None:
    rows = []
    for r in results:
        tm = _fold_metrics(split.y_test, ensemble_test_pred(r, split))
        rows.append({
            "Model":  r.name,
            "RMSE":   round(tm["rmse"], 4),
            "MAE":    round(tm["mae"],  4),
            "R²":     round(tm["r2"],   4),
        })
    with _banner(f"Hold-out Test Results  ({target_transform().short_label} scale)"):
        print(pd.DataFrame(rows).to_string(index=False))
def compute_bootstrap_distributions(*results: ModelResults, split: SplitDataset,
                                    cfg: Config) -> Dict[str, Dict[str, np.ndarray]]:
    rng      = np.random.default_rng(cfg.random_state)
    n        = len(split.y_test)
    boot_idx = rng.integers(0, n, size=(cfg.n_boot, n))
    y_res  = split.y_test[boot_idx]
    ss_tot = ((y_res - y_res.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    boot: Dict[str, Dict[str, np.ndarray]] = {}
    for r in results:
        err = ensemble_test_pred(r, split) - split.y_test
        sq  = (err ** 2)[boot_idx]
        boot[r.name] = {
            "rmse": np.sqrt(sq.mean(axis=1)),
            "mae":  np.abs(err)[boot_idx].mean(axis=1),
            "r2":   1.0 - sq.sum(axis=1) / ss_tot,
        }
    return boot
def bootstrap_test_uncertainty(*results: ModelResults, split: SplitDataset,
                               cfg: Config) -> Dict[str, Dict[str, np.ndarray]]:
    boot = compute_bootstrap_distributions(*results, split=split, cfg=cfg)
    rows = []
    for r in results:
        row: dict = {"Model": r.name}
        for metric, arr in boot[r.name].items():
            lo, hi = np.percentile(arr, [2.5, 97.5])
            row[f"{metric.upper()} mean"] = f"{arr.mean():.4f}"
            row[f"{metric.upper()} std"]  = f"{arr.std():.4f}"
            row[f"{metric.upper()} 95% CI"] = f"[{lo:.4f}, {hi:.4f}]"
        rows.append(row)
    with _banner(f"Bootstrap Uncertainty  (n={cfg.n_boot}, 95% CI, test set, "
                 f"{target_transform().short_label} scale)\n"
                 "Paired resamples shared across models (and with Fig. 05 / results.md)",
                 w=97):
        print(pd.DataFrame(rows).to_string(index=False))
    return boot
_MODALITY_BLOCKS: Dict[str, List[str]] = {
    "SAR (flood extent, exposed pop.)": ["flood_area_km2", "pop_exposed"],
    "NTL (population without power)":   ["ntl_outage_pop_acute",
                                         "ntl_outage_pop_persistent"],
    "Precipitation (IMERG)":          ["peak_precip_event", "accum_precip_event",
                                       "antecedent_precip_30d"],
    "Wind (ERA5-Land)":               ["peak_wind_event"],
    "Terrain (slope, TWI, built-up)": ["mean_slope_deg", "mean_twi", "built_surface_m2"],
    "Vulnerability (IMR, HDI, RWI)":  ["infant_mortality_rate", "hdi",
                                       "rwi_mean", "rwi_std"],
    "Hazard type":                    ["disaster_type"],
}
_SATELLITE_BLOCKS = {
    "SAR (flood extent, exposed pop.)", "NTL (population without power)",
    "Precipitation (IMERG)", "Wind (ERA5-Land)",
    "Terrain (slope, TWI, built-up)",
}
def compute_block_permutation(res: ModelResults, split: SplitDataset,
                              cfg: Config) -> Dict:
    rng    = np.random.default_rng(cfg.random_state)
    fn     = res.feature_names
    X_eval = res.X_test if res.X_test is not None else split.X_test
    baseline_rmse = _rmse(split.y_test, ensemble_test_pred(res, split))
    deltas: Dict[str, np.ndarray] = {}
    for block_name, block_feats in _MODALITY_BLOCKS.items():
        col_idx = [fn.index(f) for f in block_feats if f in fn]
        if not col_idx:
            continue
        X_perm = X_eval.copy()
        ds = np.empty(cfg.n_perm)
        for k in range(cfg.n_perm):
            X_perm[:, col_idx] = X_eval[rng.permutation(len(X_eval))][:, col_idx]
            ds[k] = _rmse(
                split.y_test,
                np.mean([m.predict(X_perm) for m in res.fitted_models], axis=0),
            ) - baseline_rmse
        deltas[block_name] = ds
    return {"model": res.name, "baseline_rmse": baseline_rmse, "deltas": deltas}
def evaluate_iso3_cv(*results: ModelResults, split: SplitDataset, cfg: Config) -> dict:
    folds  = _get_iso3_cv_splits(split, cfg)
    n_iso3 = int(split.df_train["ISO3"].nunique())
    per_model: dict = {}
    rows = []
    for r in results:
        cv, oof = cross_validate_iso3(r, split, cfg, folds)
        per_model[r.name] = {"cv": cv, "oof": oof}
        rows.append({
            "Model":  r.name,
            "RMSE":   f"{cv['rmse'].mean():.4f}±{cv['rmse'].std():.4f}",
            "MAE":    f"{cv['mae'].mean():.4f}±{cv['mae'].std():.4f}",
            "R²":     f"{cv['r2'].mean():.4f}±{cv['r2'].std():.4f}",
            "OOF R²": f"{oof['r2']:.4f}",
        })
    with _banner(f"ISO3-grouped CV — leave-countries-out  "
                 f"({len(folds)} folds, {n_iso3} countries, {target_transform().short_label} scale)\n"
                 "Hyper-parameters reused from month-level tuning (cloned, no re-tuning)",
                 w=79):
        print(pd.DataFrame(rows).to_string(index=False))
    return {"folds_n": len(folds), "n_iso3": n_iso3, "per_model": per_model}
def select_best_model(candidates: List[Tuple[str, ModelResults]],
                      split: SplitDataset) -> Tuple[str, ModelResults]:
    r2s = {tag: _fold_metrics(split.y_test, ensemble_test_pred(r, split))["r2"]
           for tag, r in candidates}
    best_tag, best_res = max(candidates, key=lambda c: r2s[c[0]])
    with _banner("Plot-model selection — best hold-out test R²"):
        for tag, r in candidates:
            print(f"  {r.name:<28} test R²={r2s[tag]:.4f}")
        print(f"{'─' * 55}")
        print(f"  → selected for plots: {best_res.name}  (R²={r2s[best_tag]:.4f})")
    return best_tag, best_res
def ensure_shap(res: ModelResults, split: SplitDataset, cfg: Config) -> ModelResults:
    if res.shap_values is not None:
        return res
    if res.X_train_imp is None or res.best_params is None:
        print(f"  ensure_shap: no refit artefacts for {res.name} — SHAP figures will skip")
        return res
    X_imp = res.X_train_imp
    print(f"\n  Computing SHAP for plot-selected model: {res.name} ...")
    if res.family == "ridge":
        final = Ridge(**res.best_params)
        final.fit(X_imp, split.y_train)
        res.shap_values = shap.LinearExplainer(final, X_imp).shap_values(X_imp)
    elif res.family == "rf":
        shap_params = dict(res.best_params)
        n_full = shap_params.get("n_estimators", 500)
        shap_params["n_estimators"] = min(n_full, cfg.shap_rf_trees)
        if shap_params["n_estimators"] < n_full:
            print(f"  SHAP-only RF refit: {n_full} → {shap_params['n_estimators']} trees")
        final = RandomForestRegressor(**shap_params,
                                      random_state=cfg.random_state, n_jobs=-1)
        final.fit(X_imp, split.y_train)
        res.shap_values = shap.TreeExplainer(final).shap_values(X_imp)
    else:
        final = xgb.XGBRegressor(**res.best_params, random_state=cfg.random_state,
                                 tree_method="hist")
        final.fit(X_imp, split.y_train)
        res.shap_values = shap.TreeExplainer(final).shap_values(X_imp)
    print(f"  SHAP values shape: {res.shap_values.shape}")
    return res

# %% [markdown]
#  %% [markdown]

# %%
def _pin_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
def _init_environment(cfg: Config, *, init_gee: bool = True) -> None:
    """Mount Drive (and optionally init GEE).

    Drive must be mounted even when only loading cached outputs, since the
    run artifacts live under cfg.output_dir on Drive; otherwise the cache
    lookup fails. GEE auth is only needed when (re)computing features.
    """
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        if init_gee:
            ee.Authenticate(force=True)
            ee.Initialize(project=cfg.gee_project)
            print("GEE initialized  |  Drive mounted")
        else:
            print("Drive mounted (GEE not initialized)")
    except ImportError:
        if init_gee:
            ee.Initialize(project=cfg.gee_project)
            print("GEE initialized (no Colab drive)")
        else:
            print("No Colab drive — using local paths")
def run_pipeline(cfg: Config) -> Dict:
    _pin_global_seeds(cfg.random_state)
    _init_environment(cfg, init_gee=True)
    run_tag    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.output_dir, run_tag)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Artifacts will be saved to: {output_dir}")
    ds = load_raw_data(cfg)
    ds = encode_disaster_type(ds)
    ds = attach_admin3_geometry(ds, cfg)
    ds = stamp_aoi_ids(ds, cfg)
    gee_df = extract_gee_features(ds.gdf, cfg)
    ds     = merge_gee_features(ds, gee_df)
    ds     = merge_imr_hdi(ds, cfg)
    ds     = merge_rwi(ds, cfg)
    ds     = engineer_features(ds, cfg)
    split = split_data(ds, cfg)
    sat_features, collinearity = drop_correlated_features(
        split.df_train, list(cfg.features),
        corr_threshold=cfg.corr_threshold, vif_cutoff=cfg.vif_cutoff)
    ns_features = [f for f in sat_features if f in cfg.features_no_satellite]
    models: Dict[str, ModelResults] = {}
    for fam, (label, trainer) in _FAMILIES.items():
        models[fam]         = trainer(split, cfg, feature_names=sat_features,
                                      model_name=label)
        models[f"{fam}_ns"] = trainer(split, cfg, feature_names=ns_features)
    suite = _suite(models)
    evaluate_models(*suite, split=split)
    boot    = bootstrap_test_uncertainty(*suite, split=split, cfg=cfg)
    iso3_cv = evaluate_iso3_cv(*suite, split=split, cfg=cfg)
    best_tag, best_res = select_best_model(
        [(fam, models[fam]) for fam in _FAMILIES], split)
    ensure_shap(best_res, split, cfg)
    block_perm = compute_block_permutation(best_res, split, cfg)
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    for tag, res in models.items():
        joblib.dump(res.fitted_models, os.path.join(models_dir, f"{tag}_models.pkl"))
        res.cv_scores.to_csv(os.path.join(models_dir, f"{tag}_cv_scores.csv"), index=False)
        if res.shap_values is not None:
            np.save(os.path.join(models_dir, f"{tag}_shap_values.npy"), res.shap_values)
        if res.X_train_imp is not None:
            np.save(os.path.join(models_dir, f"{tag}_X_train_imp.npy"), res.X_train_imp)
    np.save(os.path.join(models_dir, "X_train.npy"), split.X_train)
    np.save(os.path.join(models_dir, "X_test.npy"),  split.X_test)
    np.save(os.path.join(models_dir, "y_train.npy"), split.y_train)
    np.save(os.path.join(models_dir, "y_test.npy"),  split.y_test)
    feat_cols = list(cfg.features) + ["target", "distributed_figure",
                                       "Start date", "event_year", "ISO3"]
    split.df_train[[c for c in feat_cols if c in split.df_train.columns]].to_parquet(
        os.path.join(models_dir, "df_train.parquet"), index=True)
    split.df_test[[c for c in feat_cols if c in split.df_test.columns]].to_parquet(
        os.path.join(models_dir, "df_test.parquet"), index=True)
    print(f"\nModels + artefacts saved → {models_dir}")
    result = {"ds": ds, "split": split, "models": models,
              "best": best_res, "best_tag": best_tag,
              "sat_features": sat_features, "collinearity": collinearity,
              "iso3_cv": iso3_cv, "boot": boot, "block_perm": block_perm,
              "run_dir": output_dir}
    joblib.dump(result, os.path.join(output_dir, "outputs.pkl"))
    print(f"Full outputs dict cached → {os.path.join(output_dir, 'outputs.pkl')}")
    return result

# %% [markdown]
#  %% [markdown]

# %%
# Okabe-Ito colour-blind-safe palette — the house style.
_PAL = ["#E69F00", "#56B4E9", "#0072B2", "#009E73", "#D55E00", "#CC79A7"]
_SHOW_TITLES = False
@dataclass(frozen=True)
class PlotStyle:
    grey_diag:    str = "#888888"
    grey_annot:   str = "#555555"
    grey_thresh:  str = "#BBBBBB"
    grey_overlap: str = "#A0A0A0"
    grey_border:  str = "#D8D8D8"
    grey_coast:   str = "#B5B5B5"
    # Minimalist basemap: neutral grey land on a white ocean keeps the coloured
    # data layer (residuals / SHAP hotspots) as the visual focus.
    land:         str = "#ECECEC"
    ocean:        str = "#FFFFFF"
    edge:         str = "white"
    c_sat:   str = _PAL[0]
    c_nosat: str = _PAL[2]
    cmap_seq: "matplotlib.colors.Colormap" = dataclasses.field(
        default_factory=lambda: plt.cm.viridis)
    cmap_div: "matplotlib.colors.Colormap" = dataclasses.field(
        default_factory=lambda: plt.cm.RdBu_r)
    # Hotspot maps: PuOr_r keeps warm = increase, is colorblind-safe, and both
    # ends stay distinct from the neutral grey/white basemap.
    cmap_hotspot: "matplotlib.colors.Colormap" = dataclasses.field(
        default_factory=lambda: plt.cm.PuOr_r)
PLOT = PlotStyle()
_FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "flood_area_km2":        "Flood extent (km²)",
    "pop_exposed":           "Exposed pop.",
    "ntl_outage_pop_acute":      "NTL outage pop., days 0–10",
    "ntl_outage_pop_persistent": "NTL outage pop., days 30–60",
    "disaster_type":         "Hazard type",
    "infant_mortality_rate": "IMR",
    "hdi":                   "HDI",
    "peak_precip_event":        "Peak rainfall (mm h$^{-1}$)",
    "accum_precip_event":       "Event rainfall (mm)",
    "antecedent_precip_30d": "30-d rainfall (mm)",
    "peak_wind_event":       "Peak wind (m s$^{-1}$)",
    "mean_slope_deg":        "Slope (°)",
    "mean_twi":              "TWI",
    "built_surface_m2":      "Built-up (m²)",
    "rwi_mean":              "RWI (mean)",
    "rwi_std":               "RWI (s.d.)",
    "target":                "transformed(displaced persons)",
}
def _feat_label(n: str) -> str:
    if n == "target":
        return target_transform().axis_label
    return _FEATURE_DISPLAY_NAMES.get(n, n)
def _dn(names) -> list[str]:
    return [_feat_label(n) for n in names]
def _sat_feature_names(fn: list, cfg: Config) -> list:
    no_sat = set(cfg.features_no_satellite)
    return [f for f in fn if f not in no_sat]
# Content-derived figure sizing. Figures size themselves from how many panels,
# heatmap cells, or map tiles they hold — never a hardcoded page width — and
# matplotlib's constrained_layout engine (enabled in rcParams below) then
# guarantees nothing overlaps and every decoration gets room within that size.
# Scale a figure by changing these panel budgets, not by editing each figure.
_PANEL_W:      float = 3.4   # width  (in) of one standard plot panel
_PANEL_H:      float = 2.7   # height (in) of one standard plot panel
_HEATMAP_CELL: float = 0.42  # side   (in) budget per heatmap row / column
_MAP_W:        float = 3.7   # width  (in) of one Robinson world-map panel (~2:1)
_MAP_H:        float = 2.0   # height (in) of one Robinson world-map panel
def _apply_plot_style() -> None:
    sns.set_theme(context="notebook", style="ticks", palette=_PAL)
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial", "DejaVu Sans"],
        # Point sizes below are the final printed sizes; figures scale by adding
        # panels and whitespace, never by shrinking text. Keep >= 7 pt lettering
        # (6 pt for sub/superscripts) for legibility at print size.
        "font.size":          8,
        "axes.titlesize":     9,
        "axes.labelsize":     9,
        "xtick.labelsize":    7.5,
        "ytick.labelsize":    7.5,
        "legend.fontsize":    7.5,
        "figure.titlesize":   10,
        "axes.titleweight":   "bold",
        "axes.labelcolor":    "black",
        "axes.edgecolor":     "black",
        "figure.titleweight": "bold",
        "xtick.color":        "black",
        "ytick.color":        "black",
        "text.color":         "black",
        "legend.frameon":     False,
        "axes.grid":          False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.6,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "savefig.facecolor":  "white",
        "savefig.edgecolor":  "white",
        # No fixed page width: each figure computes its own size from its
        # content (this default only applies to any stray, uncustomised figure).
        "figure.figsize":     (2 * _PANEL_W, _PANEL_H),
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        # Automatic layout: constrained_layout reserves room for titles, tick
        # labels, colorbars and "outside" legends so nothing overlaps — it
        # replaces every manual tight_layout()/subplots_adjust() call. Note it
        # arranges within the figure size we set; it does not resize the figure,
        # which is why content-derived figsize (above constants) still matters.
        "figure.constrained_layout.use":    True,
        "figure.constrained_layout.h_pad":  4 / 72,   # inches above/below a panel
        "figure.constrained_layout.w_pad":  4 / 72,   # inches left/right of a panel
        "figure.constrained_layout.hspace": 0.03,     # fraction of panel height
        "figure.constrained_layout.wspace": 0.03,     # fraction of panel width
    })
_apply_plot_style()
def _savefig(fig, path: str, dpi: int = 300) -> None:
    # constrained_layout owns the internal spacing; bbox_inches="tight" only
    # crops the saved file to the actually-drawn content (incl. panel-letter
    # annotations that sit just outside the axes), with a hair of padding.
    # dpi only affects rasterized layers embedded in the vector PDF; figures
    # that rasterize scatter / pcolormesh pass dpi=500 for combination artwork.
    _kw = dict(dpi=dpi, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    try:
        fig.savefig(path, **_kw)
    except ValueError:
        with matplotlib.rc_context({"font.sans-serif":
                ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"]}):
            fig.savefig(path, **_kw)
    plt.close(fig)
    print(f"  Saved → {os.path.basename(path)}")
def _panel_letter(ax, letter: str, x: float = -0.05, y: float = 1.02) -> None:
    ax.text(x, y, letter, transform=ax.transAxes,
            fontweight="bold", va="bottom", ha="left")
def _set_title(ax, text: str, **kw):
    if _SHOW_TITLES:
        ax.set_title(text, **kw)
def _suptitle(fig, text: str, **kw):
    if _SHOW_TITLES:
        fig.suptitle(text, **kw)
def plot_target_distribution(split: SplitDataset, out: str) -> None:
    y_all = np.concatenate([split.y_train, split.y_test])
    raw   = target_transform().inverse(y_all, clip=False)
    fig, axes = plt.subplots(1, 2, figsize=(2 * _PANEL_W, _PANEL_H + 0.4),
                             layout="constrained")
    sns.histplot(raw, bins=60, ax=axes[0], color=_PAL[0], edgecolor=PLOT.edge,
                 kde=True)
    axes[0].xaxis.set_major_locator(mticker.MaxNLocator(5))
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))
    axes[0].set_xlabel("Displaced persons")
    axes[0].set_ylabel("Count")
    _set_title(axes[0], "Raw scale")
    sns.histplot(y_all, bins=60, ax=axes[1], color=_PAL[1], edgecolor=PLOT.edge,
                 kde=True)
    axes[1].set_xlabel(target_transform().axis_label)
    axes[1].set_ylabel("Count")
    _set_title(axes[1], f"{target_transform().short_label} scale")
    _panel_letter(axes[0], "a", x=-0.12)
    _panel_letter(axes[1], "b", x=-0.12)
    _savefig(fig, os.path.join(out, "01_target_distribution.pdf"))
def plot_missingness(split: SplitDataset, cfg: Config, out: str) -> None:
    df   = split.df_train[list(cfg.features)]
    miss = df.isna().mean().sort_values(ascending=False) * 100
    colors = [_PAL[0] if v > 20 else _PAL[2] for v in miss.values]
    # Height grows with the number of features so every bar/label has room.
    fig, ax = plt.subplots(figsize=(2 * _PANEL_W, 0.32 * len(miss) + 1.2),
                           layout="constrained")
    ax.barh(_dn(miss.index), miss.values, color=colors, edgecolor=PLOT.edge)
    ax.axvline(20, color=PLOT.grey_thresh, linestyle="--", alpha=0.8)
    ax.text(20, ax.get_ylim()[1], " 20% threshold", color=PLOT.grey_thresh,
            va="top", ha="left", fontsize=7)
    ax.set_xlabel("Missing values (%)")
    _set_title(ax, "Feature missingness — training set")
    _savefig(fig, os.path.join(out, "02_feature_missingness.pdf"))
def plot_correlation_heatmap(
    split: SplitDataset, cfg: Config, out: str,
    feature_names: Optional[List[str]] = None,
) -> None:
    feats = feature_names if feature_names is not None else list(cfg.features)
    df    = split.df_train[feats + ["target"]].copy()
    corr  = df.corr(method="spearman")
    corr.columns = corr.index = _dn(corr.columns.tolist())
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    # Size from the number of cells so each stays square and legible; extra
    # width/height leaves room for the long y-labels and the colorbar.
    side = corr.shape[0] * _HEATMAP_CELL
    fig, ax = plt.subplots(figsize=(side + 3.0, side + 1.5), layout="compressed")
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        annot_kws={"size": 7},  # fits the per-cell budget set by _HEATMAP_CELL
        cmap=PLOT.cmap_div,
        center=0, square=True, ax=ax,
        cbar_kws={"shrink": 0.6, "aspect": 30, "pad": 0.02, "label": "Spearman ρ"},
        vmin=-1, vmax=1,
    )
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _savefig(fig, os.path.join(out, "03_correlation_heatmap.pdf"))
def plot_test_scatter(*results: ModelResults, split: SplitDataset, out: str) -> None:
    # Same layout as Fig 05: model families as rows, no-satellite vs
    # satellite-augmented as the two columns. Column headers on the top row,
    # model family as the left-column label. The suite is ordered in
    # (no-satellite, satellite) pairs, so a row-major 2-column fill puts every
    # baseline in column 0 and its satellite counterpart in column 1.
    n     = len(results)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    short = target_transform().short_label
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(ncols * _PANEL_W, nrows * _PANEL_H),
                             layout="constrained")
    for i, r in enumerate(results):
        ri, ci = divmod(i, ncols)
        ax = axes[ri, ci]
        pred = ensemble_test_pred(r, split)
        is_nosat = "no satellite" in r.name.lower()
        color = PLOT.c_nosat if is_nosat else PLOT.c_sat
        cmap_m = matplotlib.colors.LinearSegmentedColormap.from_list(
            "pal_m", ["#ffffff", color])
        sns.kdeplot(x=pred, y=split.y_test, ax=ax,
                    cmap=cmap_m, fill=True, levels=8, thresh=0.05)
        lo = min(split.y_test.min(), pred.min())
        hi = max(split.y_test.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], color=PLOT.grey_diag, linestyle="--")
        ticks = [t for t in [0, 2, 4, 6, 8, 10] if lo - 0.5 <= t <= hi + 0.5]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        r_val, p_val = stats.pearsonr(split.y_test, pred)
        p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        rmse = _rmse(split.y_test, pred)
        ax.text(
            lo + 0.05 * (hi - lo), hi - 0.02 * (hi - lo),
            f"r = {r_val:.2f}  ({p_str})\nR² = {r2_score(split.y_test, pred):.3f}\n"
            f"RMSE = {rmse:.3f}",
            va="top", ha="left",
        )
        if ci == 0:
            family = r.name.replace(" (no satellite)", "").strip()
            ax.set_ylabel(family, fontweight="semibold", labelpad=10)
        else:
            ax.set_ylabel("")
        if ri == nrows - 1:
            # Keep the column identity on the x-axis so it survives with titles off.
            variant = "no-satellite baseline" if is_nosat else "satellite-augmented"
            ax.set_xlabel(f"Predicted ({short})\n{variant}")
        else:
            ax.set_xlabel("")
        _panel_letter(ax, chr(97 + i))
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.supylabel(f"Observed ({short})")
    _savefig(fig, os.path.join(out, "04_test_scatter.pdf"))
def plot_satellite_contribution(
    ridge_ns: ModelResults, ridge: ModelResults,
    rf_ns: ModelResults,    rf: ModelResults,
    xgb_ns: ModelResults,   xgb: ModelResults,
    split: SplitDataset, cfg: Config, out: str,
    boot: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> None:
    if boot is None:
        boot = compute_bootstrap_distributions(
            ridge_ns, ridge, rf_ns, rf, xgb_ns, xgb, split=split, cfg=cfg)
    C_NS, C_SAT, C_OVL = PLOT.c_nosat, PLOT.c_sat, PLOT.grey_overlap
    models  = [("Ridge",         ridge_ns.name, ridge.name),
               ("Random Forest", rf_ns.name,    rf.name),
               ("XGBoost",       xgb_ns.name,   xgb.name)]
    metrics = [("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R²")]
    col_xlim = []
    for col, _ in metrics:
        allv = np.concatenate([boot[k][col] for _, k_ns, k_sat in models
                               for k in (k_ns, k_sat)])
        lo, hi = float(allv.min()), float(allv.max())
        pad    = (hi - lo) * 0.18 + 1e-9
        col_xlim.append((lo - pad, hi + pad))
    fig, axes = plt.subplots(3, 3, figsize=(3 * _PANEL_W, 3 * _PANEL_H),
                             layout="constrained")
    letters = iter("abcdefghi")
    for ri, (model_name, k_ns, k_sat) in enumerate(models):
        for ci, (col, mlabel) in enumerate(metrics):
            ax = axes[ri, ci]
            d_ns, d_sat = boot[k_ns][col], boot[k_sat][col]
            x_lo, x_hi  = col_xlim[ci]
            grid = np.linspace(x_lo, x_hi, 512)
            k1  = gaussian_kde(d_ns)(grid)
            k2  = gaussian_kde(d_sat)(grid)
            ovl = np.minimum(k1, k2)
            ax.fill_between(grid, k1, color=C_NS,  alpha=0.30)
            ax.fill_between(grid, k2, color=C_SAT, alpha=0.30)
            ax.plot(grid, k1, color=C_NS)
            ax.plot(grid, k2, color=C_SAT)
            ax.fill_between(grid, ovl, color=C_OVL, alpha=0.45,
                            zorder=2)
            mu_ns, mu_sat = d_ns.mean(), d_sat.mean()
            ax.vlines(mu_ns,  0, np.interp(mu_ns,  grid, k1),
                      color=C_NS,  linestyle=(0, (3, 2)), zorder=3)
            ax.vlines(mu_sat, 0, np.interp(mu_sat, grid, k2),
                      color=C_SAT, linestyle=(0, (3, 2)), zorder=3)
            delta      = d_sat - d_ns
            mu_d       = float(delta.mean())
            lo_d, hi_d = np.percentile(delta, [2.5, 97.5])
            ovl_coef   = min(max(float(np.trapz(ovl, grid)), 0.0), 1.0)
            sgn = "+" if mu_d >= 0 else "−"
            ax.text(0.97, 0.96,
                    f"Δ = {sgn}{abs(mu_d):.3f}\n[{lo_d:+.3f}, {hi_d:+.3f}]\n"
                    f"overlap {ovl_coef * 100:.0f}%",
                    transform=ax.transAxes, ha="right", va="top",
                    color=PLOT.grey_annot, linespacing=1.4)
            ax.set_yticks([])
            _panel_letter(ax, next(letters), x=-0.03, y=1.02)
            if ci == 0:
                ax.set_ylabel(model_name,
                              fontweight="semibold", labelpad=12)
            if ri == len(models) - 1:
                ax.set_xlabel(f"{mlabel} ({target_transform().short_label} scale)")
            sns.despine(ax=ax, left=True)
    fig.legend(
        handles=[
            Patch(facecolor=C_NS,  alpha=0.5, label="No-satellite baseline"),
            Patch(facecolor=C_SAT, alpha=0.5, label="Satellite-augmented"),
            Patch(facecolor=C_OVL, alpha=0.55, label="Distribution overlap"),
        ],
        loc="outside lower center",  # constrained_layout reserves the margin
        ncol=3,
    )
    _savefig(fig, os.path.join(out, "05_satellite_contribution.pdf"))
def plot_shap_combined(res: ModelResults, split: SplitDataset,
                       cfg: Config, out: str,
                       block_perm: Optional[Dict] = None) -> None:
    if res.shap_values is None or res.X_train_imp is None:
        print("  SHAP artefacts missing — skipping combined SHAP figure")
        return
    nf      = len(res.feature_names)
    fn      = _dn(res.feature_names)
    sv      = res.shap_values
    X_imp   = res.X_train_imp
    # Beeswarm spans the full width on top; global importance and block
    # permutation share the bottom row below it.
    fig, axd = plt.subplot_mosaic(
        [["bee", "bee"],
         ["bar", "perm"]],
        figsize=(2.5 * _PANEL_W + 1.0, 2.9 * _PANEL_H),
        gridspec_kw={"height_ratios": [1.7, 1.1]},
        layout="constrained",
    )
    ax_bee, ax_bar, ax_perm = axd["bee"], axd["bar"], axd["perm"]
    mean_abs = np.abs(sv).mean(axis=0)
    order    = np.argsort(mean_abs)
    for yi, fi in enumerate(order):
        s   = sv[:, fi]
        fv  = X_imp[:, fi].astype(float)
        fvn = (fv - np.nanmin(fv)) / (np.nanmax(fv) - np.nanmin(fv) + 1e-9)
        sc = ax_bee.scatter(s, np.full(len(s), yi), c=fvn, cmap=PLOT.cmap_seq,
                            vmin=0, vmax=1, alpha=0.5, rasterized=True)
    ax_bee.set_yticks(range(nf))
    ax_bee.set_yticklabels([fn[i] for i in order])
    ax_bee.axvline(0, color=PLOT.grey_diag, alpha=0.5)
    ax_bee.set_xlabel("SHAP value")
    _set_title(ax_bee, "Beeswarm", pad=14)
    _panel_letter(ax_bee, "a", x=-0.07, y=1.01)
    cb1 = fig.colorbar(sc, ax=ax_bee, aspect=35, pad=0.02, shrink=0.55)
    cb1.set_label("Feature value")
    cb1.set_ticks([0, 1])
    cb1.set_ticklabels(["Low", "High"])
    vals = mean_abs[order]
    bars = ax_bar.barh(range(nf), vals,
                       color=_PAL[0], edgecolor=PLOT.edge)
    max_v = vals.max()
    for bar, v in zip(bars, vals):
        bar.set_alpha(0.4 + 0.6 * v / max_v)
    ax_bar.set_xlabel("Mean |SHAP value|")
    _set_title(ax_bar, "Global importance", pad=14)
    ax_bar.set_yticks(range(nf))
    ax_bar.set_yticklabels([fn[i] for i in order])
    _panel_letter(ax_bar, "b", x=-0.14, y=1.01)
    _draw_block_permutation(ax_perm, res, split, cfg, perm=block_perm)
    _set_title(ax_perm, "Block permutation", pad=14)
    _panel_letter(ax_perm, "c", x=-0.14, y=1.01)
    _savefig(fig, os.path.join(out, "06_shap_combined.pdf"), dpi=500)
def _running_median(x: np.ndarray, y: np.ndarray, nbins: int = 12):
    x_lo, x_hi = (float(v) for v in np.nanpercentile(x, [1, 99]))
    if x_hi <= x_lo:
        return None, None, (x_lo, x_hi)
    edges   = np.linspace(x_lo, x_hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.array([float(np.median(y[(x >= a) & (x < b)]))
                    if np.any((x >= a) & (x < b)) else np.nan
                    for a, b in zip(edges[:-1], edges[1:])])
    return centers, med, (x_lo, x_hi)
def _dominant_interaction(X_phys: np.ndarray, y_shap: np.ndarray,
                          fi: int) -> Tuple[Optional[int], float]:
    others = [j for j in range(X_phys.shape[1]) if j != fi]
    if not others:
        return None, 0.0
    def _absr(j: int) -> float:
        a = X_phys[:, j]
        if np.std(a) < 1e-12 or np.std(y_shap) < 1e-12:
            return 0.0
        return abs(float(np.corrcoef(a, y_shap)[0, 1]))
    ci = max(others, key=_absr)
    return ci, _absr(ci)
def _sci_tick(x: float, _pos=None) -> str:
    # Render axis ticks as "1e2", "2e3" (mantissa + bare exponent, no padding).
    if x == 0:
        return "0"
    mant, exp = f"{x:.0e}".split("e")
    return f"{mant}e{int(exp)}"
def plot_shap_dependence(res: ModelResults, split: SplitDataset,
                         cfg: Config, out: str) -> None:
    if res.shap_values is None or res.X_train_imp is None:
        print("  SHAP artefacts missing — skipping SHAP dependence figure")
        return
    fn = list(res.feature_names)
    sv = res.shap_values
    X_phys = SimpleImputer(strategy="median").fit_transform(
        split.df_train[fn].values).astype(float)
    sat_feats = _sat_feature_names(fn, cfg)
    if not sat_feats:
        print("  No satellite features survived VIF — skipping SHAP dependence figure")
        return
    sat_idx = sorted([fn.index(f) for f in sat_feats],
                     key=lambda i: np.abs(sv[:, i]).mean(), reverse=True)
    ncols = min(3, len(sat_idx))  # up to 3 scatter panels per row
    nrows = int(np.ceil(len(sat_idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(ncols * _PANEL_W, nrows * _PANEL_H),
                             layout="constrained")
    axes_flat = axes.flatten()
    for panel, fi in enumerate(sat_idx):
        ax = axes_flat[panel]
        x  = X_phys[:, fi]
        y  = sv[:, fi]
        ci, _ = _dominant_interaction(X_phys, y, fi)
        if ci is not None:
            cvals  = X_phys[:, ci]
            c_lo, c_hi = np.nanpercentile(cvals, [2, 98])
            cnorm  = np.clip((cvals - c_lo) / (c_hi - c_lo + 1e-9), 0, 1)
            sc = ax.scatter(x, y, c=cnorm, cmap=PLOT.cmap_seq, vmin=0, vmax=1,
                            s=10, alpha=0.55, rasterized=True)
            cb = fig.colorbar(sc, ax=ax, aspect=30, pad=0.02, shrink=0.85)
            cb.set_label(_feat_label(fn[ci]))
            cb.set_ticks([0, 1])
            cb.set_ticklabels(["Low", "High"])
        else:
            ax.scatter(x, y, color=_PAL[0], s=10, alpha=0.55,
                       rasterized=True)
        centers, med, (x_lo, x_hi) = _running_median(x, y)
        if centers is not None:
            ax.plot(centers, med, color=PLOT.grey_annot, zorder=5)
        ax.axhline(0, color=PLOT.grey_diag, linestyle="--", alpha=0.5)
        ax.set_xlabel(_feat_label(fn[fi]))
        if fn[fi] == "pop_exposed":
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_sci_tick))
        if panel % ncols == 0:
            ax.set_ylabel("SHAP value")
        _panel_letter(ax, chr(97 + panel), x=-0.02, y=1.02)
    for ax in axes_flat[len(sat_idx):]:
        ax.set_visible(False)
    _suptitle(fig, f"SHAP dependence — satellite-derived features ({res.name})")
    _savefig(fig, os.path.join(out, "07_shap_dependence.pdf"), dpi=500)
def plot_geographic_residuals(
    res: ModelResults, split: SplitDataset, out: str
) -> None:
    pred  = ensemble_test_pred(res, split)
    resid = split.y_test - pred
    df = split.df_test.copy()
    df["residual"]     = resid
    df["abs_residual"] = np.abs(resid)
    centroids = df.geometry.centroid
    lons = centroids.x.values
    lats = centroids.y.values
    vmax = float(np.percentile(np.abs(resid), 95))
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = PLOT.cmap_div
    # Stack vertically (map above, country-bias bar below) so the world map gets
    # the full figure width; constrained_layout sizes the gap between the two.
    fig = plt.figure(figsize=(2 * _MAP_W, 2 * _MAP_H + _PANEL_H),
                     layout="constrained")
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.7, 1])
    ax_map = fig.add_subplot(gs[0], projection=ccrs.Robinson())
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND,      facecolor=PLOT.land, zorder=1)
    ax_map.add_feature(cfeature.OCEAN,     facecolor=PLOT.ocean, zorder=1)
    ax_map.add_feature(cfeature.BORDERS,   edgecolor=PLOT.grey_border, linewidth=0.4, zorder=2)
    ax_map.add_feature(cfeature.COASTLINE, edgecolor=PLOT.grey_coast, linewidth=0.5, zorder=2)
    sc = ax_map.scatter(
        lons, lats, c=resid, cmap=cmap, norm=norm,
        s=42, alpha=0.8, edgecolors=PLOT.edge, linewidths=0.4,
        transform=ccrs.PlateCarree(), zorder=5,
    )
    cb = plt.colorbar(sc, ax=ax_map, orientation="horizontal",
                      pad=0.04, shrink=0.5, aspect=32)
    cb.set_label(f"Residual ({target_transform().short_label})   ← over · under →")
    mae_val = float(np.mean(np.abs(resid)))
    bias    = float(np.mean(resid))
    n_under = int((resid > 0).sum())
    n_over  = int((resid < 0).sum())
    _set_title(
        ax_map,
        f"Geographic distribution of prediction errors — {res.name} (test set)\n"
        f"MAE = {mae_val:.3f}  ·  mean bias = {bias:+.3f}  "
        f"·  under: {n_under}  over: {n_over}  (n = {len(resid)})",
        pad=5,
    )
    ax_bar = fig.add_subplot(gs[1])
    country_bias = (
        df.groupby("ISO3")["residual"]
        .agg(mean_resid="mean", n="count")
        .query("n >= 2")
    )
    n_show  = min(16, len(country_bias))
    top_iso = country_bias["mean_resid"].abs().nlargest(n_show).index
    cb_data = country_bias.loc[top_iso].sort_values("mean_resid")
    bar_colors = [cmap(norm(v)) for v in cb_data["mean_resid"]]
    bars = ax_bar.barh(
        cb_data.index, cb_data["mean_resid"],
        color=bar_colors, edgecolor=PLOT.edge, )
    for bar, (_, row) in zip(bars, cb_data.iterrows()):
        w = bar.get_width()
        ax_bar.text(
            w + (0.03 if w >= 0 else -0.03), bar.get_y() + bar.get_height() / 2,
            f"n={row['n']:.0f}",
            va="center", ha="left" if w >= 0 else "right",
            color=PLOT.grey_annot,
        )
    ax_bar.axvline(0, color=PLOT.grey_diag, linestyle="--", alpha=0.6)
    ax_bar.set_xlabel(f"Mean residual ({target_transform().short_label})")
    _set_title(ax_bar, "Country bias\n(top ± by |mean|)")
    _panel_letter(ax_map, "a", x=-0.02)
    _panel_letter(ax_bar, "b", x=-0.08)
    _savefig(fig, os.path.join(out, "08_geographic_residuals.pdf"), dpi=500)
def plot_residuals_by_disaster_type(
    *results: ModelResults, split: SplitDataset, out: str
) -> None:
    type_map = {0: "Flood", 1: "Storm"}
    df_test  = split.df_test.copy()
    df_test["Hazard type"] = df_test["disaster_type"].map(type_map)
    type_pal = {"Flood": _PAL[0], "Storm": _PAL[1]}
    metric_rows = []
    for r in results:
        pred = ensemble_test_pred(r, split)
        for dt_code, dt_label in type_map.items():
            mask = (df_test["disaster_type"].values == dt_code)
            if mask.sum() < 2:
                continue
            y_t, p_t = split.y_test[mask], pred[mask]
            metric_rows.append({
                "Model":       r.name,
                "Hazard type": dt_label,
                "RMSE":  _rmse(y_t, p_t),
                "MAE":   mean_absolute_error(y_t, p_t),
                "R²":    r2_score(y_t, p_t),
            })
    df_metrics = pd.DataFrame(metric_rows)
    fig, axes = plt.subplots(1, 3, figsize=(3 * _PANEL_W, _PANEL_H + 0.3),
                             layout="constrained")
    handles, labels = None, None
    for col_idx, (ax, metric) in enumerate(zip(axes, ["RMSE", "MAE", "R²"])):
        sns.barplot(
            data=df_metrics, x="Model", y=metric, hue="Hazard type",
            palette=type_pal, ax=ax, edgecolor=PLOT.edge, )
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right",
                           rotation_mode="anchor")
        leg = ax.get_legend()
        if leg:
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
            leg.remove()
        _panel_letter(ax, chr(97 + col_idx))
    if handles:
        fig.legend(handles, labels, loc="outside upper center",
                   ncol=len(labels), frameon=False)
    _savefig(fig, os.path.join(out, "09_residuals_by_disaster_type.pdf"))
def plot_shap_geographic_hotspots(
    res: ModelResults, split: SplitDataset, cfg: Config, out: str,
    grid_res: float = 1.0, kde_sigma: float = 1.5, density_floor: float = 1e-3,
    saturation_pct: float = 96.0,
) -> None:
    if res.shap_values is None or res.X_train_imp is None:
        print("  SHAP artefacts missing — skipping geographic hotspot maps")
        return
    feats = _sat_feature_names(res.feature_names, cfg)
    if not feats:
        print("  No satellite features retained — skipping hotspot maps")
        return
    cent   = split.df_train.geometry.centroid
    lons   = cent.x.values
    lats   = cent.y.values
    finite = np.isfinite(lons) & np.isfinite(lats)
    lons, lats = lons[finite], lats[finite]
    lon_edges = np.arange(-180.0, 180.0 + grid_res, grid_res)
    lat_edges = np.arange(-90.0,   90.0 + grid_res, grid_res)
    count_hist, _, _ = np.histogram2d(lons, lats, bins=[lon_edges, lat_edges])
    dens   = gaussian_filter(count_hist.T, sigma=kde_sigma, mode=("nearest", "wrap"))
    d_mask = dens < density_floor * dens.max()
    fields = []
    for f in feats:
        ci = res.feature_names.index(f)
        sv = res.shap_values[finite, ci]
        wh, _, _ = np.histogram2d(lons, lats, bins=[lon_edges, lat_edges], weights=sv)
        fld = gaussian_filter(wh.T, sigma=kde_sigma, mode=("nearest", "wrap"))
        fields.append(np.where(d_mask, np.nan, fld))
    vmax = float(np.nanpercentile(np.abs(np.stack(fields)), saturation_pct)) or 1e-9
    norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
    n     = len(feats)
    ncols = min(3, n)  # up to 3 world maps per row
    nrows = int(np.ceil(n / ncols))
    # "compressed" packs the fixed-aspect Robinson panels together and pushes
    # the slack to the outer margin; +0.6 in height leaves room for the colorbar.
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * _MAP_W, nrows * _MAP_H + 0.6),
        subplot_kw={"projection": ccrs.Robinson()},
        layout="compressed",
    )
    axes = np.atleast_1d(axes).ravel()
    mesh = None
    for i, (ax, f, fld) in enumerate(zip(axes, feats, fields)):
        ax.set_global()
        ax.add_feature(cfeature.LAND,      facecolor=PLOT.land, zorder=1)
        ax.add_feature(cfeature.OCEAN,     facecolor=PLOT.ocean, zorder=1)
        mesh = ax.pcolormesh(
            lon_edges, lat_edges, fld, cmap=PLOT.cmap_hotspot, norm=norm,
            transform=ccrs.PlateCarree(), shading="flat", zorder=2, rasterized=True,
        )
        ax.add_feature(cfeature.BORDERS,   edgecolor=PLOT.grey_border, linewidth=0.4, zorder=3)
        ax.add_feature(cfeature.COASTLINE, edgecolor=PLOT.grey_coast, linewidth=0.5, zorder=3)
        _set_title(ax, _feat_label(f), pad=3, fontweight="normal")
        _panel_letter(ax, chr(97 + i), x=0.0, y=1.02)
    for ax in axes[n:]:
        ax.set_visible(False)
    cb = fig.colorbar(
        mesh, ax=list(axes[:n]), orientation="horizontal",
        fraction=0.045, pad=0.03, shrink=0.5, aspect=38,
    )
    cb.set_label(
        "SHAP-weighted influence on predicted displacement  "
        f"(← decreases · increases →,  {target_transform().short_label} scale)"
    )
    _savefig(fig, os.path.join(out, "10_shap_geographic_hotspots.pdf"), dpi=500)
def _draw_block_permutation(
    ax, res: ModelResults, split: SplitDataset, cfg: Config,
    perm: Optional[Dict] = None,
) -> float:
    if perm is None or perm.get("model") != res.name:
        perm = compute_block_permutation(res, split, cfg)
    baseline_rmse = perm["baseline_rmse"]
    perm_results  = perm["deltas"]
    sorted_items = sorted(perm_results.items(), key=lambda kv: kv[1].mean())
    labels = [k for k, _ in sorted_items]
    means  = np.array([v.mean() for _, v in sorted_items])
    lo95   = np.array([np.percentile(v, 2.5)  for _, v in sorted_items])
    hi95   = np.array([np.percentile(v, 97.5) for _, v in sorted_items])
    colors = [_PAL[0] if lbl in _SATELLITE_BLOCKS else _PAL[2] for lbl in labels]
    x_span = max(hi95.max() - min(lo95.min(), 0), 0.01)
    ax.barh(range(len(labels)), means, color=colors,
            edgecolor=PLOT.edge)
    for i, (mu, lo, hi) in enumerate(zip(means, lo95, hi95)):
        ax.plot([lo, hi], [i, i], color=PLOT.grey_diag, zorder=5)
        for cap in (lo, hi):
            ax.plot([cap, cap], [i - 0.13, i + 0.13], color=PLOT.grey_diag,
                    zorder=5)
        sign = "+" if mu >= 0 else ""
        ax.text(hi + 0.025 * x_span, i, f"{sign}{mu:.3f}",
                va="center", ha="left", color=PLOT.grey_annot)
    ax.axvline(0, color=PLOT.grey_diag, linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("ΔRMSE when block permuted")
    # Reserve an empty band above the top (longest) bar and place the legend
    # there so it never overlaps the bars, error bars or value annotations.
    n = len(labels)
    ax.set_ylim(-0.6, n - 0.5 + max(2.0, 0.28 * n))
    ax.legend(
        handles=[
            Patch(facecolor=_PAL[0], label="Satellite-derived blocks"),
            Patch(facecolor=_PAL[2], label="No-satellite (baseline) blocks"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        framealpha=0.9,
        fontsize="small",
    )
    return baseline_rmse
def plot_magnitude_stratified_error(
    *results: ModelResults, split: SplitDataset, out: str,
) -> None:
    n_bins    = 5
    quantiles = np.quantile(split.y_test, np.linspace(0, 1, n_bins + 1))
    bin_idx   = np.digitize(split.y_test, quantiles[1:-1])
    def _fmt(v: float) -> str:
        if v >= 1e6: return f"{v / 1e6:.1f}M"
        if v >= 1e3: return f"{v / 1e3:.0f}k"
        return f"{v:.0f}"
    bin_labels = [
        f"Q{i + 1}\n{_fmt(float(target_transform().inverse(quantiles[i], clip=False)))}–"
        f"{_fmt(float(target_transform().inverse(quantiles[i + 1], clip=False)))}"
        f"\n(n={(bin_idx == i).sum()})"
        for i in range(n_bins)
    ]
    metric_store: dict = {r.name: {"rmse": []} for r in results}
    for i in range(n_bins):
        mask = bin_idx == i
        for r in results:
            pred = ensemble_test_pred(r, split)
            if mask.sum() < 2:
                metric_store[r.name]["rmse"].append(np.nan)
            else:
                metric_store[r.name]["rmse"].append(
                    _rmse(split.y_test[mask], pred[mask]))
    x      = np.arange(n_bins)
    n_mdls = len(results)
    width  = 0.80 / n_mdls
    fig, ax = plt.subplots(figsize=(2 * _PANEL_W, _PANEL_H + 0.6),
                           layout="constrained")
    ax.yaxis.grid(True, color=PLOT.grey_diag, alpha=0.30, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for j, (r, color) in enumerate(zip(results, _PAL)):
        offset  = (j - (n_mdls - 1) / 2) * width
        heights = metric_store[r.name]["rmse"]
        bars = ax.bar(x + offset, heights, width=width * 0.9,
                      color=color, edgecolor=PLOT.edge, label=r.name,
                      alpha=0.9, zorder=3)
        for b, h in zip(bars, heights):
            if np.isfinite(h):
                ax.annotate(
                    f"{h:.2f}", (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                    color=PLOT.grey_annot)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Displacement magnitude quintile (test set)")
    ax.set_ylabel(f"RMSE ({target_transform().short_label} scale)")
    ax.margins(y=0.16)
    ax.tick_params(axis="x", length=0)
    sns.despine(ax=ax, bottom=True)
    h_, l_ = ax.get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside upper center",
               frameon=False, ncol=min(3, len(results)))
    _savefig(fig, os.path.join(out, "11_magnitude_stratified_error.pdf"))
def run_plot_pipeline(outputs: Dict, cfg: Config,
                      plot_dir: Optional[str] = None) -> None:
    if plot_dir is None:
        plot_dir = os.path.join(outputs.get("run_dir", cfg.output_dir), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    global _SHOW_TITLES
    _SHOW_TITLES = cfg.show_figure_titles
    split        = outputs["split"]
    models       = outputs["models"]
    sat_features = outputs.get("sat_features")
    all_mdls     = _suite(models)
    best_res     = outputs.get("best", models["xgb"])
    with _banner(f"Plot pipeline  →  {plot_dir}\n"
                 f"Single-model figures use: {best_res.name}"):
        plot_target_distribution(split, plot_dir)
        plot_missingness(split, cfg, plot_dir)
        plot_correlation_heatmap(split, cfg, plot_dir, feature_names=sat_features)
        plot_test_scatter(*all_mdls, split=split, out=plot_dir)
        plot_satellite_contribution(*all_mdls, split=split, cfg=cfg, out=plot_dir,
                                    boot=outputs.get("boot"))
        plot_shap_combined(best_res, split, cfg, plot_dir,
                           block_perm=outputs.get("block_perm"))
        plot_shap_dependence(best_res, split, cfg, plot_dir)
        plot_geographic_residuals(best_res, split, plot_dir)
        plot_residuals_by_disaster_type(*all_mdls, split=split, out=plot_dir)
        plot_shap_geographic_hotspots(best_res, split, cfg, plot_dir)
        plot_magnitude_stratified_error(*all_mdls, split=split, out=plot_dir)
        n_saved = len([f for f in os.listdir(plot_dir) if f.endswith(".pdf")])
        print(f"{'─' * 55}")
        print(f"  {n_saved} figures saved  →  {plot_dir}")

# %% [markdown]
#  %% [markdown]

# %%
def _md_table(headers: List[str], rows: List[list]) -> str:
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep  = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join("" if c is None else str(c) for c in r) + " |"
            for r in rows]
    return "\n".join([head, sep, *body])
def _f(v, nd: int = 4) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(x):
        return "inf" if x > 0 else "—"
    return f"{x:.{nd}f}"
def _fi(v) -> str:
    try:
        return f"{int(round(float(v))):,}"
    except (TypeError, ValueError):
        return str(v)
def _fp(v) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(x):
        return "—"
    a = abs(x)
    if a >= 1e6: return f"{x / 1e6:.2f}M"
    if a >= 1e3: return f"{x / 1e3:.1f}k"
    return f"{x:.0f}"
def _log_raw_metrics(y_log: np.ndarray, pred_log: np.ndarray) -> dict:
    ye = target_transform().inverse(y_log, clip=False)
    pe = target_transform().inverse(pred_log, clip=True)
    abs_err = np.abs(ye - pe)
    return {
        "rmse_log":  _rmse(y_log, pred_log),
        "mae_log":   mean_absolute_error(y_log, pred_log),
        "r2_log":    r2_score(y_log, pred_log),
        "rmse_raw":  _rmse(ye, pe),
        "mae_raw":   float(abs_err.mean()),
        "medae_raw": float(np.median(abs_err)),
        "r2_raw":    r2_score(ye, pe),
    }
def summarize_shap_dependence(sv: np.ndarray, X_phys: np.ndarray,
                              fn: List[str],
                              sat_feats: List[str]) -> List[dict]:
    out: List[dict] = []
    order = sorted((fn.index(f) for f in sat_feats),
                   key=lambda i: -np.abs(sv[:, i]).mean())
    for fi in order:
        x, y = X_phys[:, fi], sv[:, fi]
        d: dict = {
            "feature":  fn[fi],
            "mean_abs": float(np.abs(y).mean()),
            "rho": (float(stats.spearmanr(x, y).correlation)
                    if np.std(x) > 1e-12 and np.std(y) > 1e-12 else float("nan")),
        }
        centers, med, (x_lo, x_hi) = _running_median(x, y)
        d["x_lo"], d["x_hi"] = x_lo, x_hi
        if centers is None:
            cs = ms = np.array([])
        else:
            keep   = np.isfinite(med)
            cs, ms = centers[keep], med[keep]
        d.update(x_zero=None, n_cross=0, pct_pos_below=None,
                 pct_pos_above=None, median_sign=None)
        if len(ms) >= 2:
            sgn   = np.sign(ms)
            cross = [k for k in range(len(ms) - 1) if sgn[k] * sgn[k + 1] < 0]
            d["n_cross"] = len(cross)
            if cross:
                k  = cross[0]
                x0 = cs[k] + (cs[k + 1] - cs[k]) * (0 - ms[k]) / (ms[k + 1] - ms[k])
                d["x_zero"] = float(x0)
                if np.any(x < x0):
                    d["pct_pos_below"] = float(100 * np.mean(y[x < x0] > 0))
                if np.any(x >= x0):
                    d["pct_pos_above"] = float(100 * np.mean(y[x >= x0] > 0))
            elif np.all(ms > 0):
                d["median_sign"] = "positive"
            elif np.all(ms < 0):
                d["median_sign"] = "negative"
        d["shape"], d["net_shift"] = "—", None
        if len(ms) >= 3:
            amp = float(ms.max() - ms.min())
            d["net_shift"] = float(ms[-1] - ms[0])
            dif  = np.diff(ms)
            mono = abs(float(dif.sum())) / (float(np.abs(dif).sum()) + 1e-12)
            noise_amp = 6.5 * float(np.std(y)) / max(np.sqrt(len(y) / 12.0), 1.0)
            if amp < max(noise_amp, 0.15 * (d["mean_abs"] + 1e-12)):
                d["shape"] = "≈ flat"
            elif mono >= 0.7:
                mid = len(ms) // 2
                r1, r2 = ms[mid] - ms[0], ms[-1] - ms[mid]
                sat = abs(r1) > 1e-12 and abs(r2) < 0.25 * abs(r1)
                d["shape"] = (("monotonic ↑" if dif.sum() > 0 else "monotonic ↓")
                              + (", saturating" if sat else ""))
            else:
                d["shape"] = "non-monotonic"
        d.update(int_feature=None, int_absr=None, int_gap=None)
        ci, absr = _dominant_interaction(X_phys, y, fi)
        if ci is not None:
            d["int_feature"], d["int_absr"] = fn[ci], absr
            hi = X_phys[:, ci] >= np.median(X_phys[:, ci])
            if 0 < hi.sum() < len(hi):
                d["int_gap"] = float(y[hi].mean() - y[~hi].mean())
        out.append(d)
    return out
def _dependence_comment(d: dict, sl: str, log_target: bool) -> str:
    rho   = d["rho"]
    parts: List[str] = []
    if np.isfinite(rho) and abs(rho) > 0.05:
        parts.append(f"higher values **{'raise' if rho > 0 else 'lower'}** "
                     f"predicted {sl} IDP (ρ = {rho:+.2f})")
    else:
        parts.append(f"no clear monotone effect (ρ = {_f(rho, 2)})")
    flat = d["shape"] == "≈ flat"
    if d["x_zero"] is not None and not flat:
        pb, pa = d["pct_pos_below"], d["pct_pos_above"]
        flank = ""
        if pb is not None and pa is not None:
            if np.isfinite(rho) and rho < 0:
                flank = (f" — {pb:.0f}% of events below it contribute "
                         f"positively, {100 - pa:.0f}% above it negatively")
            else:
                flank = (f" — {100 - pb:.0f}% of events below it contribute "
                         f"negatively, {pa:.0f}% above it positively")
        more = (f" (median re-crosses zero {d['n_cross'] - 1}× further on)"
                if d["n_cross"] > 1 else "")
        parts.append("the running-median contribution crosses zero at "
                     f"≈ {d['x_zero']:.3g}{flank}{more}")
    elif d["median_sign"] is not None and not flat:
        parts.append(f"the median contribution stays {d['median_sign']} "
                     "across the dense range")
    if d["shape"] != "—":
        parts.append(f"effect shape: {d['shape']}")
    if d["net_shift"] is not None and not flat:
        size = (f"moving across the dense range [{d['x_lo']:.3g}, "
                f"{d['x_hi']:.3g}] shifts the median contribution by "
                f"Δ = {d['net_shift']:+.2f} {sl}")
        if log_target:
            size += (f" (≈ ×{float(np.exp(d['net_shift'])):.2f} in predicted "
                     "displaced persons)")
        parts.append(size)
    if d["int_feature"] is not None:
        gap = ""
        if d["int_gap"] is not None:
            gap = (f" — events with above-median {_feat_label(d['int_feature'])} "
                   f"average {d['int_gap']:+.2f} {sl} in this feature's SHAP")
        parts.append("vertical spread is best tracked by "
                     f"**{_feat_label(d['int_feature'])}** "
                     f"(|r| = {_f(d['int_absr'], 2)}){gap}")
    return "; ".join(parts) + "."
_IDP_BANDS: List[tuple] = [
    (None,    100,    "< 100"),
    (100,    1_000,   "100 – 1,000"),
    (1_000,  10_000,  "1,000 – 10,000"),
    (10_000, None,    "> 10,000"),
]
def _idp_band_mask(arr: np.ndarray, lo, hi) -> np.ndarray:
    if lo is None:
        return arr < hi
    if hi is None:
        return arr >= lo
    return (arr >= lo) & (arr < hi)
def generate_results_report(outputs: Dict, cfg: Config,
                            path: Optional[str] = None) -> str:
    for _req in ("split", "models"):
        if _req not in outputs:
            raise KeyError(
                f"generate_results_report: required key '{_req}' missing from "
                "outputs dict. Re-run run_pipeline() to populate it."
            )
    split    = outputs["split"]
    ds       = outputs.get("ds")
    models   = outputs["models"]
    ridge,    rf_res, xgb_res = (models[f]         for f in _FAMILIES)
    ridge_ns, rf_ns,  xgb_ns  = (models[f"{f}_ns"] for f in _FAMILIES)
    run_dir  = outputs.get("run_dir", cfg.output_dir)
    sat_features = list(outputs.get("sat_features", xgb_res.feature_names))
    ns_features  = list(xgb_ns.feature_names)
    all_models   = list(_suite(models))
    best_res = outputs.get("best", xgb_res)
    best_ns  = models[f"{outputs.get('best_tag', 'xgb')}_ns"]
    if path is None:
        path = os.path.join(run_dir, "results.md")
    preds = {r.name: ensemble_test_pred(r, split) for r in all_models}
    tm    = {r.name: _log_raw_metrics(split.y_test, preds[r.name]) for r in all_models}
    tt = target_transform()
    sl = tt.short_label
    bdist = outputs.get("boot")
    if bdist is None:
        bdist = compute_bootstrap_distributions(*all_models, split=split, cfg=cfg)
    md: List[str] = []
    w = md.append
    w(f"# Results dossier — `{os.path.basename(run_dir)}`")
    w("")
    w(f"_Auto-generated by `generate_results_report()` on "
      f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}._")
    w("")
    w("> **Purpose.** Verbose, single-source-of-truth record of every numerical "
      "result the pipeline produced, written to support drafting the *Results "
      "and discussion* section and to fill the bracketed placeholders in the "
      "abstract / highlights. Each section is annotated with what the numbers "
      "mean and which figure (`NN_*.pdf`) they back, where applicable.")
    w("")
    _conv_target = (f"`target = box-cox(1 + distributed_figure, λ={tt.lam:.3f})`"
                    if tt.lam is not None else "`target = log1p(distributed_figure)`")
    w(f"**Conventions.** Target is {_conv_target} "
      f"(displaced persons). Metrics are on the {sl} scale unless suffixed "
      "*(persons)*. **Model A** = non-satellite contextual baseline; "
      "**Model B** = full model (+ satellite features). Δ = Model B − Model A. "
      "For RMSE/MAE a *negative* Δ means satellite features improve the model; "
      "for R² a *positive* Δ is the improvement.")
    w("")
    w("## Contents")
    w("")
    _toc = [
        "Executive summary", "Run configuration", "Dataset & data inventory",
        "Target distribution", "Feature inventory & missingness",
        "Collinearity screen & feature selection", "Feature–target association",
        "Train / test split composition", "Model hyper-parameters",
        "Cross-validation results", "Hold-out test results",
        "Bootstrap uncertainty", "Satellite contribution (paired bootstrap)",
        "Block permutation importance", "SHAP global importance & geographic hotspots",
        "Stratified error analysis", "Geographic residual summary",
        "Artefact manifest",
    ]
    for i, t in enumerate(_toc, 1):
        w(f"{i}. {t}")
    w("")
    w("---")
    w("")
    w("## Key numbers at a glance")
    w("")
    w(f"> Quick-reference table for copy-pasting into the abstract, highlights, "
      f"and results section. Bootstrap CIs are 95th-percentile over {cfg.n_boot} "
      "resamples (same draws as §12–§13). Full tables in the sections indicated.")
    w("")
    _bm        = tm[best_res.name]
    _br2_ci    = np.percentile(bdist[best_res.name]["r2"],  [2.5, 97.5])
    _bmae_ci   = np.percentile(bdist[best_res.name]["mae"], [2.5, 97.5])
    _b_cv_r2   = best_res.cv_scores["r2"]
    _deltas_r2 = {fam: (tm[models[fam].name]["r2_log"]
                        - tm[models[f"{fam}_ns"].name]["r2_log"])
                  for fam in _FAMILIES}
    _best_fam   = max(_deltas_r2, key=lambda f: _deltas_r2[f])
    _best_label = _FAMILIES[_best_fam][0]
    _best_delta = _deltas_r2[_best_fam]
    _d_r2_boot  = (bdist[models[_best_fam].name]["r2"]
                   - bdist[models[f"{_best_fam}_ns"].name]["r2"])
    _d_ci       = np.percentile(_d_r2_boot, [2.5, 97.5])
    _sig_str    = ("✓ significant (CI excludes 0)"
                   if (_d_ci[0] > 0 or _d_ci[1] < 0) else "n.s.")
    w(_md_table(["Metric", "Value", "Model / context", "See"],
        [
            [f"Hold-out R² ({sl})",
             _f(_bm["r2_log"]),
             f"{best_res.name} (Model B)", "§11"],
            [f"Hold-out R² 95% CI ({sl})",
             f"[{_br2_ci[0]:.4f}, {_br2_ci[1]:.4f}]",
             f"{best_res.name} (Model B)", "§12"],
            [f"Hold-out MAE ({sl})",
             _f(_bm["mae_log"]),
             f"{best_res.name} (Model B)", "§11"],
            [f"Hold-out MAE 95% CI ({sl})",
             f"[{_bmae_ci[0]:.4f}, {_bmae_ci[1]:.4f}]",
             f"{best_res.name} (Model B)", "§12"],
            ["Hold-out MAE (persons)",
             _fp(_bm["mae_raw"]),
             f"{best_res.name} (Model B)", "§11"],
            ["Hold-out RMSE (persons)",
             _fp(_bm["rmse_raw"]),
             f"{best_res.name} (Model B)", "§11"],
            [f"CV R² mean ± std ({sl}, month-level)",
             f"{_b_cv_r2.mean():.4f} ± {_b_cv_r2.std():.4f}",
             f"{best_res.name} (Model B)", "§10"],
            [f"Max ΔR² satellite gain B−A ({sl})",
             f"{_best_delta:+.4f}",
             f"{_best_label}", "§1, §13"],
            ["ΔR² 95% CI (bootstrap, best family)",
             f"[{_d_ci[0]:+.4f}, {_d_ci[1]:+.4f}]",
             f"{_best_label}", "§13"],
            ["ΔR² bootstrap significance", _sig_str,
             f"{_best_label}", "§13"],
        ]))
    w("")
    w(f"> **Draft abstract sentence:** "
      f"\"The {best_res.name.split()[0]} model (Model B, including Sentinel-1 SAR "
      f"and VIIRS nightlight features) achieved R² = {_f(_bm['r2_log'])} "
      f"and MAE = {_fp(_bm['mae_raw'])} displaced persons on the held-out "
      f"test set ({sl} scale). Adding satellite-derived features improved R² by "
      f"{_best_delta:+.4f} over the contextual baseline ({_best_label} family; "
      f"bootstrap 95 % CI: [{_d_ci[0]:+.4f}, {_d_ci[1]:+.4f}]; "
      f"{_sig_str}).\"")
    w("")
    w("---")
    w("")
    w("## 1. Executive summary")
    w("")
    w("Headline hold-out test performance for all six models. *#feat* is the "
      "number of VIF-retained inputs. Raw-scale columns are back-transformed to "
      "displaced persons. This is the table to mine for the abstract's "
      "*“[insert key performance metrics and comparative findings]”* placeholder.")
    w("")
    rows = []
    for r in all_models:
        m = tm[r.name]
        kind = "B (+sat)" if r in (ridge, rf_res, xgb_res) else "A (ctx)"
        rows.append([r.name, kind, len(r.feature_names),
                     _f(m["r2_log"]), _f(m["rmse_log"]), _f(m["mae_log"]),
                     _fp(m["rmse_raw"]), _fp(m["mae_raw"]), _fp(m["medae_raw"])])
    w(_md_table(["Model", "Set", "#feat", f"R² ({sl})", f"RMSE ({sl})", f"MAE ({sl})",
                 "RMSE (persons)", "MAE (persons)", "Median AE (persons)"], rows))
    w("")
    w(f"- **Best model (Model B, satellite — drives §14–17):** {best_res.name} — "
      f"R² = {_f(tm[best_res.name]['r2_log'])}, "
      f"RMSE = {_f(tm[best_res.name]['rmse_log'])}, MAE = {_f(tm[best_res.name]['mae_log'])} "
      f"({target_transform().short_label}); MAE = {_fp(tm[best_res.name]['mae_raw'])} persons.")
    overall = max(all_models, key=lambda r: tm[r.name]["r2_log"])
    if overall.name != best_res.name:
        w(f"- **Note — higher hold-out R² overall:** {overall.name} "
          f"(R² = {_f(tm[overall.name]['r2_log'])}) edges out the plot-selected "
          f"Model B above; §14–17 still analyse {best_res.name} as the satellite "
          "representative.")
    expl = " (Δ = B − A; ΔR² > 0 and ΔMAE < 0 ⇒ imagery improves the model)"
    for fam, (label, _) in _FAMILIES.items():
        mB, mA = tm[models[fam].name], tm[models[f"{fam}_ns"].name]
        w(f"- **Satellite contribution ({label}):** "
          f"ΔR² = {mB['r2_log'] - mA['r2_log']:+.4f}, "
          f"ΔMAE = {mB['mae_log'] - mA['mae_log']:+.4f}{expl}.")
        expl = ""
    w("")
    w("See §13 for whether these gaps are statistically significant "
      "(paired bootstrap).")
    w("")
    w("## 2. Run configuration")
    w("")
    w("Full `Config` snapshot — every knob that defines this run, for exact "
      "reproducibility (seeds, split ratios, CV folds, bootstrap/permutation "
      "counts, GEE windows and scales).")
    w("")
    cfg_rows = [(fld.name, f"`{getattr(cfg, fld.name)}`")
                for fld in dataclasses.fields(cfg)
                if not isinstance(getattr(cfg, fld.name), tuple)]
    w(_md_table(["Config field", "Value"], cfg_rows))
    w("")
    w(f"- **Reproducibility:** `random_state = {cfg.random_state}`, "
      f"`n_folds = {cfg.n_folds}`, `test_size = {cfg.test_size:.0%}`, "
      f"`n_boot = {cfg.n_boot}`, `n_perm = {cfg.n_perm}`.")
    w(f"- **All candidate features ({len(cfg.features)}):** "
      + ", ".join(f"`{x}`" for x in cfg.features) + ".")
    w(f"- **Non-satellite / contextual features ({len(cfg.features_no_satellite)}):** "
      + ", ".join(f"`{x}`" for x in cfg.features_no_satellite) + ".")
    w(f"- **VIF-retained Model B inputs ({len(sat_features)}):** "
      + ", ".join(f"`{x}`" for x in sat_features) + ".")
    w(f"- **VIF-retained Model A inputs ({len(ns_features)}):** "
      + ", ".join(f"`{x}`" for x in ns_features) + ".")
    w("")
    w("## 3. Dataset & data inventory")
    w("")
    if ds is not None and getattr(ds, "gdf", None) is not None:
        g = ds.gdf
        n_rows = len(g)
        ev_col = "event_id" if "event_id" in g.columns else "Event ID"
        w("Counts describe the full modelled population (after dropping rows with "
          "missing geometry / dates / target), i.e. the union of the train and "
          "test splits. Backs the IDMC dataset description.")
        w("")
        inv = [
            ("Event records (rows)", _fi(n_rows)),
            ("Unique events", _fi(g[ev_col].nunique()) if ev_col in g.columns else _fi(n_rows)),
            ("Countries (ISO3)", _fi(g["ISO3"].nunique())),
            ("Start-date range", f"{g['Start date'].min().date()} → {g['Start date'].max().date()}"),
        ]
        if "admin3_gid" in g.columns:
            n_poly = int(g["admin3_gid"].notna().sum())
            inv.append(("Matched to GADM Admin3 polygon",
                        f"{_fi(n_poly)} ({100 * n_poly / n_rows:.1f}%)"))
            inv.append(("Region-sized point-buffer fallback",
                        f"{_fi(n_rows - n_poly)} ({100 * (n_rows - n_poly) / n_rows:.1f}%)"))
        w(_md_table(["Quantity", "Value"], inv))
        w("")
        w("**Disaster-type composition** (binarised target used by the models). "
          "Reports event counts and the per-type target distribution (mean / "
          f"median, on both the {target_transform().short_label} and person "
          "scales):")
        w("")
        lab = {0: "Flood (0)", 1: "Storm (1)"}
        _sl = target_transform().short_label
        dt_rows = []
        for k in sorted(g["disaster_type"].dropna().unique()):
            m    = g["disaster_type"] == k
            tv   = g.loc[m, "target"].astype(float)
            rawv = g.loc[m, "distributed_figure"].astype(float)
            dt_rows.append([
                lab.get(int(k), str(k)), _fi(int(m.sum())),
                f"{100 * int(m.sum()) / n_rows:.1f}%",
                _f(tv.mean(), 3), _f(tv.median(), 3), _f(tv.std(), 3),
                _fp(rawv.mean()), _fp(rawv.median()),
            ])
        w(_md_table(["disaster_type", "Events", "Share",
                     f"mean ({_sl})", f"median ({_sl})", f"std ({_sl})",
                     "mean (persons)", "median (persons)"], dt_rows))
        w("")
        if "Hazard type" in g.columns:
            w("**Original hazard-type labels** (pre-binarisation, for the methods text):")
            w("")
            hz = g["Hazard type"].value_counts()
            w(_md_table(["Hazard type", "Events", "Share"],
                        [(str(k), _fi(v), f"{100 * v / n_rows:.1f}%") for k, v in hz.items()]))
            w("")
        w("**Geographic composition by region:**")
        w("")
        reg = g["ISO3"].map(_ISO3_TO_REGION).fillna("Other")
        w(_md_table(["Region", "Events", "Share"],
                    [(k, _fi(v), f"{100 * v / n_rows:.1f}%")
                     for k, v in reg.value_counts().items()]))
        w("")
        w("**Top 15 countries by event count:**")
        w("")
        w(_md_table(["ISO3", "Events", "Share"],
                    [(k, _fi(v), f"{100 * v / n_rows:.1f}%")
                     for k, v in g["ISO3"].value_counts().head(15).items()]))
        w("")
        w("**Events per year:**")
        w("")
        yr = g["event_year"].value_counts().sort_index()
        w(_md_table(["Year", "Events", "Share"],
                    [[str(int(k)), _fi(v), f"{100 * v / n_rows:.1f}%"]
                     for k, v in yr.items()]))
    else:
        w("_`outputs['ds']` not available — dataset-level inventory skipped._")
    w("")
    w("## 4. Target distribution")
    w("")
    w(f"Distribution of the displacement target, raw (persons) and "
      f"{target_transform().short_label}. The heavy right skew on the raw scale "
      f"motivates the {target_transform().short_label} transform used for "
      "training. Backs Fig. `01_target_distribution.pdf`.")
    w("")
    if ds is not None and getattr(ds, "gdf", None) is not None:
        raw = ds.gdf["distributed_figure"].astype(float)
        logt = ds.gdf["target"].astype(float)
        w(_md_table(["Statistic", "Raw (persons)", target_transform().short_label],
            [("count",  _fi(raw.count()),                  _fi(logt.count())),
             ("mean",   _fp(raw.mean()),                   _f(logt.mean(), 3)),
             ("std",    _fp(raw.std()),                    _f(logt.std(), 3)),
             ("min",    _fp(raw.min()),                    _f(logt.min(), 3)),
             ("25%",    _fp(raw.quantile(0.25)),           _f(logt.quantile(0.25), 3)),
             ("median", _fp(raw.median()),                 _f(logt.median(), 3)),
             ("75%",    _fp(raw.quantile(0.75)),           _f(logt.quantile(0.75), 3)),
             ("max",    _fp(raw.max()),                    _f(logt.max(), 3)),
             ("skewness", _f(raw.skew(), 3),               _f(logt.skew(), 3))]))
        w("")
        w("**Magnitude bins** (the operational displacement bands used in the "
          "stratified analysis, §16):")
        w("")
        rv     = raw.values
        masks  = [_idp_band_mask(rv, lo, hi) for lo, hi, _ in _IDP_BANDS]
        labels = [label for _, _, label in _IDP_BANDS]
        w(_md_table(["IDP band", "Events", "Share", "Mean (persons)"],
            [(lab, _fi(m.sum()), f"{100 * m.sum() / len(rv):.1f}%",
              _fp(rv[m].mean()) if m.sum() else "—")
             for lab, m in zip(labels, masks)]))
    else:
        w(f"_`outputs['ds']` not available — using split arrays for the "
          f"{target_transform().short_label} distribution only._")
        y_all = np.concatenate([split.y_train, split.y_test])
        w(_md_table(["Statistic", target_transform().short_label],
            [("count", _fi(len(y_all))), ("mean", _f(y_all.mean(), 3)),
             ("std", _f(y_all.std(), 3)), ("min", _f(y_all.min(), 3)),
             ("max", _f(y_all.max(), 3))]))
    w("")
    w("## 5. Feature inventory & missingness")
    w("")
    w("Per-feature summary statistics and missing-value fraction on the "
      "**training set** (median imputation is applied downstream). Features with "
      ">20 % missingness are flagged ⚠. Backs Fig. `02_feature_missingness.pdf`.")
    w("")
    dft = split.df_train
    rows = []
    for feat in cfg.features:
        if feat not in dft.columns:
            continue
        col = dft[feat].astype(float)
        miss = col.isna().mean() * 100
        vif_tag = ("✓ B+A" if feat in set(ns_features) else
                   "✓ B only" if feat in set(sat_features) else "dropped")
        rows.append([
            _feat_label(feat) + (" ⚠" if miss > 20 else ""),
            vif_tag,
            _fi(col.count()), f"{miss:.1f}%",
            _f(col.mean(), 3), _f(col.std(), 3), _f(col.min(), 3),
            _f(col.quantile(0.25), 3), _f(col.median(), 3),
            _f(col.quantile(0.75), 3), _f(col.max(), 3),
        ])
    w(_md_table(["Feature", "VIF-retained", "n", "miss%", "mean", "std", "min",
                 "25%", "median", "75%", "max"], rows))
    w("")
    w("_(VIF-retained: ✓ B+A = retained contextual feature, used by both Model A and B; "
      "✓ B only = retained satellite feature, used only by Model B; "
      "dropped = removed by the collinearity screen — details in §6.)_")
    w("")
    w("## 6. Collinearity screen & feature selection")
    w("")
    w(f"The VIF-based pruning applied before SHAP (Spearman screen "
      f"at |ρ| > {cfg.corr_threshold}, then iterative VIF removal at cutoff "
      f"{cfg.vif_cutoff}) on the training "
      "set. Correlated features split SHAP attribution, so this step is required "
      "for clean attributions. Backs the *Methods* collinearity description and "
      "the feature set behind Fig. `03_correlation_heatmap.pdf`.")
    w("")
    names        = list(cfg.features)
    retained     = [f for f in names if f in set(sat_features)]
    dropped_auth = [f for f in names if f not in set(sat_features)]
    collin       = outputs.get("collinearity")
    drift_vn     = None
    if collin is not None:
        flagged     = list(collin["flagged_pairs"])
        removal_vif = dict(collin["dropped"])
        final_vifs  = dict(collin["final_vifs"])
    else:
        Xc = SimpleImputer(strategy="median").fit_transform(
            dft[names].values).astype(float)
        vn, fb      = _collinearity_screen(Xc, names,
                                           cfg.corr_threshold, cfg.vif_cutoff)
        flagged     = list(fb["flagged_pairs"])
        removal_vif = dict(fb["dropped"])
        drift_vn    = vn
        if set(vn) == set(retained):
            final_vifs = fb["final_vifs"]
        else:
            ret_idx    = [names.index(f) for f in retained]
            X_ret      = Xc[:, ret_idx]
            final_vifs = {retained[i]: variance_inflation_factor(X_ret, i)
                          for i in range(len(retained))}
    w(f"**Spearman-flagged pairs (|ρ| > {cfg.corr_threshold}):**")
    w("")
    if flagged:
        w(_md_table(["Feature A", "Feature B", "|ρ|"],
                    [(_feat_label(a), _feat_label(b), _f(r, 3)) for a, b, r in flagged]))
    else:
        w(f"_No pairs exceed the {cfg.corr_threshold} threshold._")
    w("")
    w("**Features dropped by iterative VIF removal:**")
    w("")
    if dropped_auth:
        w(_md_table(["Dropped feature", "VIF at removal"],
                    [(_feat_label(d), _f(removal_vif[d], 2) if d in removal_vif else "—")
                     for d in dropped_auth]))
    else:
        w("_No features dropped — all VIF below cutoff._")
    w("")
    w("**Final VIF of retained features:**")
    w("")
    w(_md_table(["Feature", "VIF"],
                [(_feat_label(k), _f(v, 2)) for k, v in
                 sorted(final_vifs.items(), key=lambda kv: -kv[1])]))
    w("")
    w(f"- **Retained ({len(retained)}):** "
      + ", ".join(f"`{x}`" for x in retained) + ".")
    if drift_vn is not None and set(drift_vn) != set(retained):
        w(f"- ⚠ **Drift check:** the in-report VIF reproduction retained "
          f"`{sorted(drift_vn)}`, which differs from the pipeline's `sat_features` "
          f"(`{sorted(retained)}`). The pipeline set is authoritative above — "
          "investigate `drop_correlated_features()` for a divergent change.")
    w("")
    w("## 7. Feature–target association")
    w("")
    w(f"Spearman rank correlation between each candidate feature and the "
      f"{target_transform().short_label} target on the training set, ranked by "
      "absolute strength. A descriptive "
      "(univariate, monotonic) view that complements the multivariate SHAP "
      "importances in §15.")
    w("")
    cm = dft[list(cfg.features) + ["target"]].corr(method="spearman")["target"].drop("target")
    cm = cm.reindex(cm.abs().sort_values(ascending=False).index)
    w(_md_table(["Feature", "Spearman ρ vs target"],
                [(_feat_label(k), _f(v, 3)) for k, v in cm.items()]))
    w("")
    w("## 8. Train / test split composition")
    w("")
    w(f"Target-stratified random hold-out (ungrouped): the test set is a fresh "
      f"quantile-bin stratified sample of rows on the "
      f"{target_transform().short_label} target ({cfg.n_target_bins} bins), so it "
      "reflects the full deployment distribution and both partitions span the full "
      "range of displacement magnitudes. Month-level leakage control "
      "(month-year grouping) is applied to the cross-validation folds "
      "only (§10), not to this hold-out — by design the test set shares "
      "calendar months with the training set. Below: split sizes plus a balance "
      "check on disaster type and target distribution.")
    w("")
    w(_md_table(["Quantity", "Value"],
        [("Training rows", _fi(len(split.df_train))),
         ("Test rows", _fi(len(split.df_test))),
         ("Test fraction (config)", f"{cfg.test_size:.0%}"),
         ("Stratification bins", str(cfg.n_target_bins))]))
    w("")
    rows = []
    for nm, dframe, yv in [("Train", split.df_train, split.y_train),
                           ("Test",  split.df_test,  split.y_test)]:
        flood_share = (dframe["disaster_type"] == 0).mean() * 100
        rows.append([nm, _fi(len(dframe)), f"{flood_share:.1f}%",
                     _f(yv.mean(), 3), _f(np.median(yv), 3),
                     _f(yv.min(), 3), _f(yv.max(), 3)])
    w(_md_table(["Split", "n", "Flood share", "target mean", "target median",
                 "target min", "target max"], rows))
    w("")
    if "Start date" in split.df_train.columns:
        _train_my = split.df_train["Start date"].dt.to_period("M").astype(str)
        _test_my  = split.df_test["Start date"].dt.to_period("M").astype(str)
        _shared_my = set(_train_my) & set(_test_my)
        w(f"- **Shared calendar months (train ∩ test):** {len(_shared_my)} of "
          f"{_train_my.nunique()} training months also appear in the test set — "
          "by design (the hold-out is ungrouped; month-level leakage control "
          "applies to the CV folds only, see §10).")
    w("")
    w("## 9. Model hyper-parameters")
    w("")
    _ridge_alpha = ridge.fitted_models[0].named_steps["ridge"].get_params()["alpha"]
    _xgb_esr     = xgb_res.fitted_models[0].get_params().get("early_stopping_rounds")
    w(f"Ridge uses a fixed `alpha = {_ridge_alpha}` with median imputation and standardised "
      "features (identical pipeline for Models A and B). XGBoost and Random Forest "
      f"hyper-parameters were each tuned with Optuna (multivariate TPE, "
      f"{cfg.n_optuna_trials} trials, "
      "pruning); the selected values are read back from the fitted estimators below. "
      f"XGBoost is tuned by `xgb.cv` over the same {cfg.n_folds}-fold month-level CV "
      "splits with median pruning, while Random Forest is tuned on a single grouped "
      "80/20 inner split with Hyperband pruning on tree count. For XGBoost, "
      f"`n_estimators` is capped by early stopping ({_xgb_esr} rounds), so the *trees actually "
      "used* (mean best_iteration + 1 over folds) is the meaningful complexity figure.")
    w("")
    def _xgb_params(res: ModelResults) -> dict:
        p = res.fitted_models[0].get_params()
        keys = ["max_depth", "learning_rate", "subsample", "colsample_bytree",
                "min_child_weight", "gamma", "reg_alpha", "reg_lambda",
                "n_estimators", "objective"]
        return {k: p.get(k) for k in keys}
    def _best_iters(res: ModelResults) -> List[int]:
        out = []
        for m in res.fitted_models:
            bi = getattr(m, "best_iteration", None)
            if bi is not None:
                out.append(int(bi) + 1)
        return out
    def _rf_params(res: ModelResults) -> dict:
        p = res.fitted_models[0].named_steps["rf"].get_params()
        keys = ["n_estimators", "max_depth", "max_features",
                "min_samples_split", "min_samples_leaf"]
        return {k: p.get(k) for k in keys}
    pB, pA = _xgb_params(xgb_res), _xgb_params(xgb_ns)
    w(_md_table(["XGBoost param", "Model B (sat)", "Model A (ns)"],
                [(k, f"`{pB[k]}`", f"`{pA[k]}`") for k in pB]))
    w("")
    biB, biA = _best_iters(xgb_res), _best_iters(xgb_ns)
    if biB:
        w(f"- **XGBoost trees used (best_iteration + 1):** Model B mean = "
          f"{np.mean(biB):.0f} (folds: {biB}); Model A mean = "
          f"{np.mean(biA):.0f} (folds: {biA}).")
    w("")
    rfB, rfA = _rf_params(rf_res), _rf_params(rf_ns)
    w(_md_table(["Random Forest param", "Model B (sat)", "Model A (ns)"],
                [(k, f"`{rfB[k]}`", f"`{rfA[k]}`") for k in rfB]))
    w("")
    w(f"- **Ridge:** `alpha = {_ridge_alpha}`, median imputation, StandardScaler "
      "(Models A and B identical).")
    w("")
    w("## 10. Cross-validation results")
    w("")
    w(f"{cfg.n_folds}-fold StratifiedGroupKFold (shuffle, seed = {cfg.random_state}), "
      "grouped by event month-year (month-level) and stratified on "
      "target quantiles, on the training set — a whole calendar month is held out per "
      "fold, so no month appears in both train and validation; this keeps each event's "
      "disaggregated rows together and stops same-month events leaking across the "
      "boundary. Per-fold metrics, fold mean ± std, and the "
      "out-of-fold (OOF) aggregate (all training rows scored on the fold where "
      f"they were held out) are reported for each model — all on the {target_transform().short_label} scale.")
    for r in all_models:
        cv = r.cv_scores
        rows = [[f"Fold {i + 1}", _f(row["rmse"]), _f(row["mae"]), _f(row["r2"])]
                for i, (_, row) in enumerate(cv.iterrows())]
        rows.append(["**Mean**", _f(cv["rmse"].mean()), _f(cv["mae"].mean()),
                     _f(cv["r2"].mean())])
        rows.append(["**Std**", _f(cv["rmse"].std()), _f(cv["mae"].std()),
                     _f(cv["r2"].std())])
        mask = np.isfinite(r.oof_preds)
        oofm = _fold_metrics(split.y_train[mask], r.oof_preds[mask])
        rows.append(["**OOF aggregate**", _f(oofm["rmse"]), _f(oofm["mae"]),
                     _f(oofm["r2"])])
        w("")
        w(f"### {r.name}")
        w("")
        w(_md_table(["Fold", "RMSE", "MAE", "R²"], rows))
    iso3_cv = outputs.get("iso3_cv")
    if iso3_cv is None:
        folds   = _get_iso3_cv_splits(split, cfg)
        iso3_cv = {
            "folds_n": len(folds),
            "n_iso3":  int(split.df_train["ISO3"].nunique()),
            "per_model": {
                r.name: dict(zip(("cv", "oof"),
                                 cross_validate_iso3(r, split, cfg, folds)))
                for r in all_models
            },
        }
    w("")
    w("### ISO3-grouped cross-validation (leave-countries-out)")
    w("")
    w(f"Geographic-generalisation robustness check. The same six models are "
      f"re-cross-validated on {iso3_cv['folds_n']}-fold StratifiedGroupKFold grouped "
      f"by **ISO3 country** ({iso3_cv['n_iso3']} countries) instead of the main "
      "month-year (month-level) grouping, so "
      "no country appears in both a fold's train and validation portions — each fold "
      "is scored only on countries it never trained on. **Hyper-parameters are "
      "reused from the month-level tuning above (estimators are cloned, nothing is "
      "re-tuned)**; only the fold grouping changes. Metrics are mean ± std over "
      "folds plus the out-of-fold (OOF) aggregate, on the "
      f"{target_transform().short_label} scale. Read against the month-level fold "
      "means above: a marked drop here flags reliance on country-specific signal "
      "that does not transfer to unseen countries.")
    w("")
    rows = []
    for r in all_models:
        rec = iso3_cv["per_model"][r.name]
        cv_iso, oof_iso = rec["cv"], rec["oof"]
        month_r2  = float(r.cv_scores["r2"].mean())
        iso3_r2   = float(cv_iso["r2"].mean())
        delta_r2  = iso3_r2 - month_r2
        delta_str = f"{delta_r2:+.4f}" + (" ↓" if delta_r2 < -0.05 else "")
        rows.append([
            r.name,
            f"{cv_iso['rmse'].mean():.4f} ± {cv_iso['rmse'].std():.4f}",
            f"{cv_iso['mae'].mean():.4f} ± {cv_iso['mae'].std():.4f}",
            f"{cv_iso['r2'].mean():.4f} ± {cv_iso['r2'].std():.4f}",
            _f(oof_iso["rmse"]), _f(oof_iso["mae"]), _f(oof_iso["r2"]),
            delta_str,
        ])
    w(_md_table(["Model", "RMSE (mean±std)", "MAE (mean±std)", "R² (mean±std)",
                 "OOF RMSE", "OOF MAE", "OOF R²", "ΔR² vs month-level"], rows))
    w("")
    _n_drop = sum(
        1 for r in all_models
        if (float(iso3_cv["per_model"][r.name]["cv"]["r2"].mean())
            - float(r.cv_scores["r2"].mean())) < -0.05
    )
    if _n_drop:
        w(f"- **{_n_drop} of {len(all_models)} models** drop > 0.05 R² when "
          "moving from month-level to country-level CV (↓ flagged above) — "
          "these models rely partly on country-specific signal that does not "
          "transfer to unseen countries.")
    w("")
    w("## 11. Hold-out test results")
    w("")
    w(f"Final fold-ensemble performance on the untouched test set, on both the "
      f"{target_transform().short_label} modelling scale and the back-transformed "
      "person scale (RMSE, MAE, "
      "median absolute error, R²). Backs Fig. `04_test_scatter.pdf`.")
    w("")
    rows = []
    for r in all_models:
        m = tm[r.name]
        rows.append([r.name, len(r.feature_names),
                     _f(m["r2_log"]), _f(m["rmse_log"]), _f(m["mae_log"]),
                     _fp(m["rmse_raw"]), _fp(m["mae_raw"]), _fp(m["medae_raw"]),
                     _f(m["r2_raw"])])
    w(_md_table(["Model", "#feat", f"R² ({sl})", f"RMSE ({sl})", f"MAE ({sl})",
                 "RMSE (persons)", "MAE (persons)", "Median AE (persons)",
                 "R² (persons)"], rows))
    w("")
    _best_all = max(all_models, key=lambda r: tm[r.name]["r2_log"])
    w(f"> **Draft sentence (Results §1):** \"The {best_res.name.split()[0]} model "
      f"(Model B) achieved the best hold-out R² = {_f(tm[best_res.name]['r2_log'])} "
      f"and MAE = {_fp(tm[best_res.name]['mae_raw'])} displaced persons "
      f"(Median AE = {_fp(tm[best_res.name]['medae_raw'])} persons). "
      + (f"Note: {_best_all.name} posts a higher R² = {_f(tm[_best_all.name]['r2_log'])} "
         f"overall but is a contextual baseline (Model A); {best_res.name} is the "
         "satellite-enhanced representative used in §14–§17.\"" if
         _best_all.name != best_res.name else
         "Bootstrap uncertainty on these point estimates is quantified in §12.\""))
    w("")
    w("## 12. Bootstrap uncertainty")
    w("")
    w(f"Non-parametric bootstrap over the test set (n = {cfg.n_boot} resamples, "
      "shared resample indices across models). Mean, std and 95 % percentile CI "
      f"per metric, on the {target_transform().short_label} scale.")
    w("")
    rows = []
    for r in all_models:
        d = bdist[r.name]
        cells = [r.name]
        for k in ("rmse", "mae", "r2"):
            a = d[k]
            lo, hi = np.percentile(a, [2.5, 97.5])
            cells += [_f(a.mean()), _f(a.std()), f"[{lo:.4f}, {hi:.4f}]"]
        rows.append(cells)
    w(_md_table(["Model",
                 "RMSE mean", "RMSE std", "RMSE 95% CI",
                 "MAE mean", "MAE std", "MAE 95% CI",
                 "R² mean", "R² std", "R² 95% CI"], rows))
    w("")
    _tightest = min(all_models,
                    key=lambda r: (np.percentile(bdist[r.name]["r2"], 97.5)
                                   - np.percentile(bdist[r.name]["r2"], 2.5)))
    _t_lo, _t_hi = np.percentile(bdist[_tightest.name]["r2"], [2.5, 97.5])
    w(f"- **Tightest R² CI:** {_tightest.name} — width = {_t_hi - _t_lo:.4f} "
      f"([{_t_lo:.4f}, {_t_hi:.4f}]); narrower CI indicates more stable "
      "test-set predictions across bootstrap resamples.")
    w("")
    w("## 13. Satellite contribution (paired bootstrap)")
    w("")
    w("Paired difference Δ = metric(Model B) − metric(Model A) on matched "
      "bootstrap resamples, with the overlapping coefficient (OVL) of the two "
      "bootstrap densities. *Significance* uses whether the 95 % CI excludes 0 "
      "**and** the direction is an improvement (RMSE/MAE down, R² up). This is "
      "the statistical backbone of Fig. `05_satellite_contribution.pdf`.")
    w("")
    metric_defs = [("rmse", "RMSE", False), ("mae", "MAE", False), ("r2", "R²", True)]
    grids = {}
    for key, _, _ in metric_defs:
        allv = np.concatenate([bdist[r.name][key] for r in all_models])
        lo, hi = float(allv.min()), float(allv.max())
        pad = (hi - lo) * 0.18 + 1e-9
        grids[key] = np.linspace(lo - pad, hi + pad, 512)
    rows = []
    for fam, (model_label, _) in _FAMILIES.items():
        r_ns, r_sat = models[f"{fam}_ns"], models[fam]
        for key, mlabel, higher in metric_defs:
            d_ns, d_sat = bdist[r_ns.name][key], bdist[r_sat.name][key]
            delta = d_sat - d_ns
            mu = float(delta.mean())
            lo, hi = np.percentile(delta, [2.5, 97.5])
            try:
                grid = grids[key]
                ovl = float(np.trapz(np.minimum(gaussian_kde(d_ns)(grid),
                                                 gaussian_kde(d_sat)(grid)), grid))
                ovl = min(max(ovl, 0.0), 1.0)
                ovl_s = f"{ovl * 100:.0f}%"
            except Exception:
                ovl_s = "—"
            excl = (lo > 0) or (hi < 0)
            improves = (d_sat.mean() > d_ns.mean()) if higher else (d_sat.mean() < d_ns.mean())
            flag = ("significant ✓" if (excl and improves)
                    else "significant ✗ (worse)" if excl else "n.s.")
            rows.append([model_label, mlabel, f"{mu:+.4f}",
                         f"[{lo:+.4f}, {hi:+.4f}]", ovl_s, flag])
    w(_md_table(["Model", "Metric", "Δ (B−A) mean", "Δ 95% CI",
                 "Overlap", "Significance"], rows))
    w("")
    w("> **OVL (overlap coefficient):** area of intersection of the two bootstrap "
      "densities — lower is better separation. OVL < 50 % ⇒ distributions are "
      "well-separated; OVL > 80 % ⇒ near-complete overlap, the gap is practically "
      "negligible regardless of the CI test.")
    w("")
    _sig_fams = []
    for fam, (fam_label, _) in _FAMILIES.items():
        _d_ns_r2  = bdist[models[f"{fam}_ns"].name]["r2"]
        _d_sat_r2 = bdist[models[fam].name]["r2"]
        _delta_r2 = _d_sat_r2 - _d_ns_r2
        _lo_r2, _hi_r2 = np.percentile(_delta_r2, [2.5, 97.5])
        _excl = (_lo_r2 > 0) or (_hi_r2 < 0)
        _up   = _d_sat_r2.mean() > _d_ns_r2.mean()
        if _excl and _up:
            _sig_fams.append(fam_label)
    if _sig_fams:
        w("- **Significant R² improvement (95% CI excludes 0, direction = ↑):** "
          + ", ".join(_sig_fams) + ".")
    else:
        w("- **No model family shows a statistically significant R² improvement** "
          "at the 95 % level — satellite features improve the point-estimate R² "
          "but the bootstrap CIs include zero. See §1 for point-estimate deltas.")
    w("")
    w("## 14. Block permutation importance")
    w("")
    w("Each modality block is jointly permuted on the test set and the resulting "
      "increase in RMSE is recorded (mean ± 95 % CI over permutations). Larger "
      f"ΔRMSE ⇒ the block matters more. {best_res.name}, Model B. Shown as panel c of "
      "Fig. `06_shap_combined.pdf`.")
    w("")
    bp = outputs.get("block_perm")
    if bp is None or bp.get("model") != best_res.name:
        bp = compute_block_permutation(best_res, split, cfg)
    base_rmse = bp["baseline_rmse"]
    w(f"- **Baseline test RMSE (unpermuted):** {base_rmse:.4f}  "
      f"·  {cfg.n_perm} permutations per block.")
    w("")
    rows = []
    for k, v in sorted(bp["deltas"].items(), key=lambda kv: -kv[1].mean()):
        lo, hi = np.percentile(v, [2.5, 97.5])
        rows.append([k, "satellite" if k in _SATELLITE_BLOCKS else "contextual",
                     f"{v.mean():+.4f}", f"[{lo:+.4f}, {hi:+.4f}]"])
    w(_md_table(["Modality block", "Group", "ΔRMSE mean", "ΔRMSE 95% CI"], rows))
    w("")
    _sat_total = sum(v.mean() for k, v in bp["deltas"].items()
                     if k in _SATELLITE_BLOCKS)
    _ctx_total = sum(v.mean() for k, v in bp["deltas"].items()
                     if k not in _SATELLITE_BLOCKS)
    _perm_total = _sat_total + _ctx_total + 1e-12
    w(f"- **Satellite blocks combined:** ΔRMSE = {_sat_total:+.4f} "
      f"({100 * _sat_total / _perm_total:.0f}% of total modality-permutation "
      "importance).")
    w(f"- **Contextual blocks combined:** ΔRMSE = {_ctx_total:+.4f} "
      f"({100 * _ctx_total / _perm_total:.0f}%).")
    w("")
    w("## 15. SHAP global importance")
    w("")
    w(f"SHAP computed on the plot-selected model: **{best_res.name}**, Model B.")
    w("")
    if best_res.shap_values is not None:
        sv = best_res.shap_values
        fn = list(best_res.feature_names)
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        total = mean_abs.sum() + 1e-12
        w("### 15a. Global importance")
        w("")
        w(f"Global feature importance as mean |SHAP value| (impact on "
          f"{target_transform().short_label} IDP) "
          "over the training set, with each feature's share of total importance. "
          "Backs Fig. `06_shap_combined.pdf` (panels a–b).")
        w("")
        w(_md_table(["Rank", "Feature", "Mean |SHAP|", "Share"],
            [[i + 1, _feat_label(fn[j]), _f(mean_abs[j], 4), f"{100 * mean_abs[j] / total:.1f}%"]
             for i, j in enumerate(order)]))
        w("")
        sat_feats = _sat_feature_names(fn, cfg)
        _sl       = target_transform().short_label
        w("### 15b. Geographic SHAP hotspots — satellite-derived features")
        w("")
        w("**Fig. `10_shap_geographic_hotspots.pdf`** — Geographic SHAP "
          "hotspots, satellite-derived features (weighted KDE, training "
          "events). The figure no longer carries an in-panel title; this "
          "caption is its label.")
        w("")
        if not sat_feats:
            w("_No satellite-derived feature survived VIF selection — no geographic "
              "hotspot map produced._")
        elif "ISO3" not in split.df_train.columns:
            w("_`ISO3` unavailable on the training split — geographic SHAP breakdown "
              "skipped._")
        else:
            iso_all = split.df_train["ISO3"].astype(str).values
            try:
                _cent  = split.df_train.geometry.centroid
                finite = np.isfinite(_cent.x.values) & np.isfinite(_cent.y.values)
            except Exception:
                finite = np.ones(len(iso_all), dtype=bool)
            iso    = iso_all[finite]
            sv_geo = sv[finite]
            n_drop = int((~finite).sum())
            w("For each satellite-derived predictor, *where* its SHAP influence "
              "concentrates: per-event signed SHAP **summed by country** "
              f"(Σ SHAP = magnitude × spatial concentration, on the {_sl} scale — the "
              "quantity Fig. `10_shap_geographic_hotspots.pdf` renders as a smoothed "
              "field). Positive Σ ⇒ the feature **raises** predicted displacement "
              "there (warm on the map); negative ⇒ **lowers** it (cool). `mean` is "
              "the typical per-event effect and `n` the country's training-event "
              "count, so a hotspot driven by many small events is distinguishable "
              "from one driven by a few large ones. The final column is the share of "
              "each feature's total |country-level Σ SHAP| held by its three most-"
              "influential countries — higher ⇒ the feature's influence is more "
              "geographically localised.")
            w("")
            if n_drop:
                w(f"_{n_drop} training event(s) without a locatable centroid are "
                  "excluded here to match the hotspot map's event set._")
                w("")
            def _cell(c, ssum, smean, scnt):
                region = _ISO3_TO_REGION.get(c, "Other")
                return (f"{c} ({region}) — Σ={ssum.loc[c]:+.3f}, "
                        f"mean={smean.loc[c]:+.3f}, n={int(scnt.loc[c])}")
            extremes: List[tuple] = []
            rows = []
            for f in sat_feats:
                ci  = fn.index(f)
                grp = pd.DataFrame({"ISO3": iso, "sv": sv_geo[:, ci]}).groupby("ISO3")["sv"]
                ssum, smean, scnt = grp.sum(), grp.mean(), grp.count()
                warm, cool = ssum.idxmax(), ssum.idxmin()
                abs_net    = ssum.abs().sort_values(ascending=False)
                top3_share = 100 * abs_net.head(3).sum() / (abs_net.sum() + 1e-12)
                extremes += [(f, warm, float(ssum.loc[warm])),
                             (f, cool, float(ssum.loc[cool]))]
                rows.append([_feat_label(f), _f(mean_abs[ci], 4),
                             _cell(warm, ssum, smean, scnt),
                             _cell(cool, ssum, smean, scnt),
                             f"{top3_share:.0f}%"])
            w(_md_table(
                ["Feature", "Mean |SHAP|", "Raises most (warm)",
                 "Lowers most (cool)", "Top-3 conc."], rows))
            w("")
            reg     = pd.Series(iso).map(_ISO3_TO_REGION).fillna("Other").values
            sat_idx = [fn.index(f) for f in sat_feats]
            net_evt = sv_geo[:, sat_idx].sum(axis=1)
            rg = (pd.DataFrame({"region": reg, "net": net_evt})
                  .groupby("region")["net"].agg(net="sum", mean="mean", n="count")
                  .sort_values("net", ascending=False))
            w("**Net satellite-feature push by region** (Σ of every satellite SHAP "
              f"value, {_sl} scale; positive ⇒ the imagery features collectively "
              "raise predicted displacement across that region):")
            w("")
            w(_md_table(["Region", "Σ net SHAP", "Mean / event", "n events"],
                [[idx, f"{r.net:+.3f}", f"{r['mean']:+.4f}", _fi(r.n)]
                 for idx, r in rg.iterrows()]))
            w("")
            warm_hot = max(extremes, key=lambda t: t[2])
            cool_hot = min(extremes, key=lambda t: t[2])
            w(f"- **Strongest warm hotspot:** {_feat_label(warm_hot[0])} in "
              f"{warm_hot[1]} ({_ISO3_TO_REGION.get(warm_hot[1], 'Other')}), "
              f"Σ SHAP = {warm_hot[2]:+.3f} ({_sl}) — the single feature×country "
              "combination that most raises predicted displacement.")
            w(f"- **Strongest cool hotspot:** {_feat_label(cool_hot[0])} in "
              f"{cool_hot[1]} ({_ISO3_TO_REGION.get(cool_hot[1], 'Other')}), "
              f"Σ SHAP = {cool_hot[2]:+.3f} ({_sl}) — most lowers predicted "
              "displacement.")
        sl = target_transform().short_label
        w("")
        w("### 15c. SHAP dependence — satellite-feature effect shape")
        w("")
        sat_dep = _sat_feature_names(fn, cfg)
        if not sat_dep:
            w("_No satellite-derived feature survived VIF selection — no dependence "
              "figure produced._")
        else:
            X_phys = SimpleImputer(strategy="median").fit_transform(
                split.df_train[fn].values).astype(float)
            dep = summarize_shap_dependence(sv, X_phys, list(fn), sat_dep)
            w("For each satellite-derived predictor: the monotone direction of its "
              "effect (Spearman ρ between the feature value and its own SHAP value — "
              f"positive ⇒ higher values push predicted {sl} IDP up) and the feature "
              "it interacts with most strongly (largest |Pearson r| between that "
              "feature's value and this feature's SHAP contribution). These are the "
              "per-panel summaries rendered in Fig. `07_shap_dependence.pdf`.")
            w("")
            w(_md_table(
                ["Feature", "Mean |SHAP|", "ρ(value, SHAP)", "Direction",
                 "Top interaction", "|r|"],
                [[_feat_label(d["feature"]), _f(d["mean_abs"], 4), _f(d["rho"], 3),
                  ("—" if not np.isfinite(d["rho"]) else
                   "↑ raises" if d["rho"] > 0.05 else
                   "↓ lowers" if d["rho"] < -0.05 else "≈ flat"),
                  _feat_label(d["int_feature"]) if d["int_feature"] else "—",
                  _f(d["int_absr"], 3)] for d in dep]))
            w("")
            w("**Per-panel comment scaffolding** — one bullet per Fig. 07 panel "
              "(letters follow the figure's importance ordering), each giving "
              "the standard dependence-plot reading: direction → zero-crossing "
              "threshold (physical units) → effect shape → effect size over the "
              "dense range → dominant interaction:")
            w("")
            log_t = target_transform().lam is None
            for k, d in enumerate(dep):
                w(f"- **{_feat_label(d['feature'])}** (panel {chr(97 + k)}): "
                  + _dependence_comment(d, sl, log_t))
            w("")
            w("_Thresholds and shapes are read off the running median inside the "
              "[1, 99]-percentile dense range (the figure's x-limits); the tails "
              "outside it are sparse, so avoid quantitative claims there. SHAP "
              "values describe the model's learned association, not a causal "
              "effect of the feature._")
    else:
        w(f"_{best_res.name} SHAP values unavailable — section skipped._")
    w("")
    w("## 16. Stratified error analysis")
    w("")
    w("Error decomposed along three operationally-relevant axes: displacement "
      "magnitude, disaster type, and country (geographic generalisation).")
    w("")
    w("### 16a. By displacement-magnitude quintile (test RMSE)")
    w("")
    w(f"Test set partitioned into five equal-count bins on the "
      f"{target_transform().short_label} target; "
      "per-model RMSE within each bin. Reveals where models work (mid-range) vs "
      "struggle (extreme tails). Backs Fig. `11_magnitude_stratified_error.pdf`.")
    w("")
    nb = 5
    q = np.quantile(split.y_test, np.linspace(0, 1, nb + 1))
    bidx = np.digitize(split.y_test, q[1:-1])
    rows = []
    for i in range(nb):
        mask = bidx == i
        cells = [f"Q{i + 1}: {_fp(float(target_transform().inverse(q[i], clip=False)))}–"
                 f"{_fp(float(target_transform().inverse(q[i + 1], clip=False)))}",
                 _fi(int(mask.sum()))]
        for r in all_models:
            cells.append(
                _f(_rmse(split.y_test[mask], preds[r.name][mask]))
                if mask.sum() >= 2 else "—")
        rows.append(cells)
    w(_md_table(["Quintile (persons)", "n"] + [r.name for r in all_models], rows))
    w("")
    w(f"### 16b. By operational IDP band — Model A vs Model B ({best_res.name} MAE)")
    w("")
    w(f"MAE ({target_transform().short_label}) with 95 % bootstrap CI in each "
      "operational band, and the "
      "per-band ΔMAE = MAE(A) − MAE(B). Positive ΔMAE ⇒ satellite imagery lowers "
      "error in that band — directly addresses whether the imagery benefit is "
      "concentrated at particular displacement scales.")
    w("")
    raw_test = split.df_test["distributed_figure"].astype(float).values
    aerrA = np.abs(split.y_test - preds[best_ns.name])
    aerrB = np.abs(split.y_test - preds[best_res.name])
    rng_b = np.random.default_rng(cfg.random_state)
    def _mae_ci(a: np.ndarray):
        if len(a) == 0:
            return "—", "—"
        idx = rng_b.integers(0, len(a), size=(cfg.n_boot, len(a)))
        ms = a[idx].mean(axis=1)
        return f"{a.mean():.4f}", f"[{np.percentile(ms, 2.5):.4f}, {np.percentile(ms, 97.5):.4f}]"
    band_masks  = [_idp_band_mask(raw_test, lo, hi) for lo, hi, _ in _IDP_BANDS]
    band_labels = [label for _, _, label in _IDP_BANDS]
    rows = []
    for lab, m in zip(band_labels, band_masks):
        maeA, ciA = _mae_ci(aerrA[m])
        maeB, ciB = _mae_ci(aerrB[m])
        dmae = (aerrA[m].mean() - aerrB[m].mean()) if m.sum() else np.nan
        rows.append([lab, _fi(int(m.sum())), maeA, ciA, maeB, ciB, _f(dmae)])
    w(_md_table(["IDP band", "n", "MAE A", "A 95% CI", "MAE B", "B 95% CI",
                 "ΔMAE (A−B)"], rows))
    w("")
    w("### 16c. By disaster type (all models)")
    w("")
    w("Per-model RMSE/MAE/R² split by disaster type on the test set. Identifies "
      "whether floods vs cyclones benefit systematically. Backs Fig. "
      "`09_residuals_by_disaster_type.pdf`.")
    w("")
    type_map = {0: "Flood", 1: "Storm"}
    dt_test  = split.df_test["disaster_type"].values
    rows = []
    for code, lab in type_map.items():
        m = dt_test == code
        if m.sum() < 2:
            continue
        for r in all_models:
            pred = preds[r.name]
            rows.append([lab, r.name, _fi(int(m.sum())),
                         _f(_rmse(split.y_test[m], pred[m])),
                         _f(mean_absolute_error(split.y_test[m], pred[m])),
                         _f(r2_score(split.y_test[m], pred[m]))])
    w(_md_table(["Hazard type", "Model", "n", "RMSE", "MAE", "R²"], rows))
    w("")
    w(f"### 16d. Country-level generalisation ({best_res.name}, A vs B)")
    w("")
    w("For every country with ≥ 3 test events, per-country MAE for Model A and "
      "Model B and ΔMAE = MAE(A) − MAE(B). Positive ⇒ satellite features reduce "
      "error there. Mirrors the leave-country-out generalisation analysis.")
    w("")
    dfc = pd.DataFrame({"ISO3": split.df_test["ISO3"].values,
                        "aA": aerrA, "aB": aerrB})
    grp = dfc.groupby("ISO3").agg(n=("aA", "size"), maeA=("aA", "mean"),
                                  maeB=("aB", "mean"))
    grp = grp[grp["n"] >= 3].copy()
    grp["dMAE"] = grp["maeA"] - grp["maeB"]
    grp = grp.sort_values("dMAE", ascending=False)
    if len(grp):
        n_imp = int((grp["dMAE"] > 0).sum())
        n_wor = int((grp["dMAE"] < 0).sum())
        w(f"- **{n_imp}** of {len(grp)} countries improve with satellite features; "
          f"**{n_wor}** worsen. Mean ΔMAE = {grp['dMAE'].mean():+.4f}.")
        w("")
        w(_md_table(["ISO3", "n", "MAE A", "MAE B", "ΔMAE (A−B)"],
            [[idx, _fi(row.n), _f(row.maeA), _f(row.maeB), _f(row.dMAE)]
             for idx, row in grp.iterrows()]))
    else:
        w("_No country has ≥ 3 test events — per-country breakdown omitted._")
    w("")
    w("## 17. Geographic residual summary")
    w("")
    w(f"Signed test residuals (observed − predicted) for the best satellite model "
      f"(**{best_res.name}**); positive ⇒ under-prediction. Aggregate error/bias plus "
      "the countries with the largest mean bias (≥ 2 test events). Backs Fig. "
      "`08_geographic_residuals.pdf`.")
    w("")
    resid = split.y_test - preds[best_res.name]
    _sl = target_transform().short_label
    w(_md_table(["Quantity", "Value"],
        [("Test events", _fi(len(resid))),
         (f"MAE ({_sl})", _f(np.abs(resid).mean())),
         (f"Mean bias ({_sl})", f"{resid.mean():+.4f}"),
         ("Under-predicted (resid > 0)", _fi(int((resid > 0).sum()))),
         ("Over-predicted (resid < 0)", _fi(int((resid < 0).sum())))]))
    w("")
    dfb = pd.DataFrame({"ISO3": split.df_test["ISO3"].values, "resid": resid})
    cb = dfb.groupby("ISO3")["resid"].agg(mean_resid="mean", n="count")
    cb = cb[cb["n"] >= 2].copy()
    cb["abs"] = cb["mean_resid"].abs()
    cb = cb.sort_values("abs", ascending=False).head(16)
    if len(cb):
        w("**Top countries by |mean residual| (≥ 2 test events):**")
        w("")
        w(_md_table(["ISO3", "n", f"Mean residual ({target_transform().short_label})"],
            [[idx, _fi(row.n), f"{row.mean_resid:+.4f}"] for idx, row in cb.iterrows()]))
    w("")
    w("## 18. Artefact manifest")
    w("")
    w("Files persisted for this run (for the *Code availability* section and to "
      "regenerate any figure or number without re-running the GEE export / "
      "training). Sizes are approximate.")
    w("")
    for label, d in [("Models & data artefacts", os.path.join(run_dir, "models")),
                     ("Figures", os.path.join(run_dir, "plots"))]:
        w(f"**{label}** — `{d}`")
        w("")
        if os.path.isdir(d):
            files = sorted(os.listdir(d))
            rows = []
            for fname in files:
                fp = os.path.join(d, fname)
                try:
                    kb = os.path.getsize(fp) / 1024.0
                    size = f"{kb / 1024:.1f} MB" if kb >= 1024 else f"{kb:.1f} KB"
                except OSError:
                    size = "—"
                rows.append([f"`{fname}`", size])
            w(_md_table(["File", "Size"], rows) if rows else "_(empty)_")
        else:
            w("_(directory not present)_")
        w("")
    w("---")
    w(f"_End of dossier — {len(_toc)} sections._")
    text = "\n".join(md) + "\n"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    with _banner("Verbose results dossier written"):
        print(f"  Path     : {path}")
        print(f"  Sections : {len(_toc)}")
        print(f"  Size     : {len(text):,} chars")
    return path

# %% [markdown]
#  %% [markdown]

# %%
if __name__ == "__main__":
    cfg = Config()
    RECOMPUTE = False   # set False to load the latest cached run instead of recomputing
    if RECOMPUTE:
        outputs = run_pipeline(cfg)
    else:
        _init_environment(cfg, init_gee=False)
        latest  = sorted(glob.glob(os.path.join(cfg.output_dir, "*")))[-1]
        outputs = joblib.load(os.path.join(latest, "outputs.pkl"))
        print(f"Loaded cached outputs from {latest}")
    run_plot_pipeline(outputs, cfg)
    generate_results_report(outputs, cfg)
    print("\nPipeline complete.")



