# %% [markdown]
# # Read GADM
#
# This notebook inspects the GADM GeoPackage and loads the first available layer into a GeoDataFrame.

# %%
from pathlib import Path
import geopandas as gpd

base_path = Path("/Users/su28355962/Documents/codes/migration-estimation")
gpkg_path = base_path / "src" / "data" / "gadm_410.gpkg"

layers = gpd.list_layers(gpkg_path)
layers

# %%
admin3_columns = [
    "GID_3",
    "NAME_3",
    "VARNAME_3",
    "NL_NAME_3",
    "HASC_3",
    "CC_3",
    "TYPE_3",
    "ENGTYPE_3",
    "VALIDFR_3",
]

present_admin3_columns = [column for column in admin3_columns if column in gadm.columns]
admin3_mask = gadm["GID_3"].fillna("").str.strip().ne("")
admin3_rows = gadm.loc[admin3_mask]

type_3_counts = (
    admin3_rows.loc[admin3_rows["TYPE_3"].fillna("").str.strip().ne(""), "TYPE_3"]
    .value_counts()
    .head(10)
)
engtype_3_counts = (
    admin3_rows.loc[admin3_rows["ENGTYPE_3"].fillna("").str.strip().ne(""), "ENGTYPE_3"]
    .value_counts()
    .head(10)
)

print(f"Admin3 columns present: {present_admin3_columns}")
print(f"Rows with admin3 values: {len(admin3_rows):,} / {len(gadm):,}")
print(f"Share with admin3 values: {len(admin3_rows) / len(gadm):.2%}")
print("Top TYPE_3 values:")
print(type_3_counts)
print("Top ENGTYPE_3 values:")
print(engtype_3_counts)

display(
    admin3_rows[
        [
            "GID_0",
            "NAME_0",
            "GID_1",
            "NAME_1",
            "GID_2",
            "NAME_2",
            "GID_3",
            "NAME_3",
            "TYPE_3",
            "ENGTYPE_3",
        ]
    ].head(10)
)

# %%
import matplotlib.pyplot as plt

admin1 = gadm.drop_duplicates(subset="GID_1").copy()

fig, ax = plt.subplots(figsize=(18, 10))
admin1.plot(
    ax=ax, column="NAME_0", linewidth=0.2, edgecolor="white", cmap="tab20", legend=False
)
ax.set_title("GADM Admin1 Boundaries - Global Map")
ax.set_axis_off()
plt.show()

# %%
import matplotlib.pyplot as plt

admin2 = gadm.drop_duplicates(subset="GID_2").copy()

fig, ax = plt.subplots(figsize=(18, 10))
admin2.plot(
    ax=ax,
    column="NAME_0",
    linewidth=0.15,
    edgecolor="white",
    cmap="tab20",
    legend=False,
)
ax.set_title("GADM Admin2 Boundaries - Global Map")
ax.set_axis_off()
plt.show()

# %%
import matplotlib.pyplot as plt

admin3 = gadm.drop_duplicates(subset="GID_3").copy()

fig, ax = plt.subplots(figsize=(18, 10))
admin3.plot(
    ax=ax, column="NAME_0", linewidth=0.1, edgecolor="white", cmap="tab20", legend=False
)
ax.set_title("GADM Admin3 Boundaries - Global Map")
ax.set_axis_off()
plt.show()

# %%
from IPython.display import display
import matplotlib.pyplot as plt

admin3_area = admin3.to_crs(6933).copy()
admin3_area["area_km2"] = admin3_area.geometry.area / 1_000_000

continent_stats = (
    admin3_area.dropna(subset=["CONTINENT"])
    .groupby("CONTINENT")["area_km2"]
    .agg(
        count="count",
        mean_km2="mean",
        median_km2="median",
        std_km2="std",
        min_km2="min",
        max_km2="max",
    )
    .sort_values("mean_km2", ascending=False)
)

overall_stats = admin3_area["area_km2"].agg(
    count="count",
    mean_km2="mean",
    median_km2="median",
    std_km2="std",
    min_km2="min",
    p25_km2=lambda s: s.quantile(0.25),
    p75_km2=lambda s: s.quantile(0.75),
    max_km2="max",
)

print("Admin3 area statistics in square kilometers")
display(overall_stats.to_frame(name="value"))
print("Admin3 area statistics by continent")
display(continent_stats)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
admin3_area["area_km2"].plot(
    kind="hist", bins=60, ax=axes[0], color="#4C78A8", edgecolor="white"
)
axes[0].set_title("Admin3 area distribution")
axes[0].set_xlabel("Area (km^2)")
axes[0].set_ylabel("Count")

admin3_area.boxplot(column="area_km2", by="CONTINENT", ax=axes[1], rot=45)
axes[1].set_title("Admin3 area by continent")
axes[1].set_xlabel("Continent")
axes[1].set_ylabel("Area (km^2)")

fig.suptitle("")
plt.tight_layout()
plt.show()

# %%
# Lookup function: find admin3 polygon(s) for a point and optionally plot using cartopy
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
except Exception:
    ccrs = None


def lookup_admin3(
    point,
    gadm_df=None,
    buffer_m=0,
    return_polygon=True,
    plot=False,
    ax=None,
    highlight_kwargs=None,
):
    """
    Find the `admin3` polygon(s) that contain a point.

    Parameters:
    - point: shapely.geometry.Point or (lon, lat) tuple (EPSG:4326)
    - gadm_df: GeoDataFrame with GADM polygons (defaults to `gadm` variable in notebook)
    - buffer_m: buffer around the point in meters (0 = no buffer)
    - return_polygon: return matching GeoDataFrame if True
    - plot: show a map using cartopy if True
    - ax: an existing matplotlib Axes (Cartopy) to draw on
    - highlight_kwargs: dict of kwargs passed to GeoDataFrame.plot for highlight

    Returns:
    - If plot=True: (matches_gdf, fig, ax)
    - If plot=False and return_polygon=True: matches_gdf
    - If plot=False and return_polygon=False: None
    """
    if gadm_df is None:
        try:
            gadm_df = gadm
        except NameError:
            raise ValueError("No `gadm` GeoDataFrame found; pass `gadm_df=`.")

    # Normalize input point to shapely Point
    if isinstance(point, (list, tuple)) and len(point) == 2:
        geom = Point(point[0], point[1])
    elif hasattr(point, "geom_type") and point.geom_type == "Point":
        geom = point
    else:
        raise ValueError("`point` must be a (lon, lat) tuple or a shapely Point")

    # Prepare point GeoDataFrame (assume input is lon/lat EPSG:4326)
    point_gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

    # Ensure gadm has a CRS; assume EPSG:4326 if missing
    if gadm_df.crs is None:
        gadm_df = gadm_df.set_crs("EPSG:4326", allow_override=True)

    # Reproject point to gadm CRS for spatial operations
    if point_gdf.crs != gadm_df.crs:
        point_proj = point_gdf.to_crs(gadm_df.crs)
    else:
        point_proj = point_gdf

    search_geom = point_proj.geometry.iloc[0]

    # If buffering in meters, project to a metric CRS (EPSG:3857) to buffer, then back
    if buffer_m and buffer_m > 0:
        metric_crs = "EPSG:3857"
        point_metric = point_proj.to_crs(metric_crs)
        buf = point_metric.geometry.iloc[0].buffer(buffer_m)
        buffer_geom = gpd.GeoSeries([buf], crs=metric_crs).to_crs(gadm_df.crs).iloc[0]
        mask = gadm_df.geometry.intersects(buffer_geom)
    else:
        # prefer contains, but fall back to intersects for boundary cases
        mask = gadm_df.geometry.contains(search_geom) | gadm_df.geometry.intersects(
            search_geom
        )

    matches = gadm_df.loc[mask].copy()

    if matches.empty:
        print("No matching admin3 polygon found at this location.")
    else:
        display(matches.head())

    fig_out = None
    ax_out = None

    if plot:
        if ccrs is None:
            raise ImportError("cartopy is required for plotting (install cartopy).")

        proj = ccrs.PlateCarree()
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=proj)
        else:
            fig = ax.get_figure()

        # Plot a light basemap of country outlines (use gadm boundaries)
        try:
            gadm_df.boundary.plot(
                ax=ax, linewidth=0.5, edgecolor="gray", transform=proj, zorder=0
            )
        except Exception:
            # fallback if gadm boundaries fail
            pass

        if not matches.empty:
            hkwargs = dict(edgecolor="red", facecolor="none", linewidth=2)
            if highlight_kwargs:
                hkwargs.update(highlight_kwargs)
            matches.plot(ax=ax, transform=proj, zorder=3, **hkwargs)

        # plot the query point in blue
        point_gdf.plot(ax=ax, color="blue", markersize=60, transform=proj, zorder=4)

        # set extent to match matches if present, otherwise center on point
        if not matches.empty:
            minx, miny, maxx, maxy = matches.total_bounds
            padx = (maxx - minx) * 0.2 if (maxx - minx) > 0 else 0.1
            pady = (maxy - miny) * 0.2 if (maxy - miny) > 0 else 0.1
            ax.set_extent(
                [minx - padx, maxx + padx, miny - pady, maxy + pady], crs=proj
            )
        else:
            lon, lat = geom.x, geom.y
            ax.set_extent([lon - 0.5, lon + 0.5, lat - 0.5, lat + 0.5], crs=proj)

        ax.coastlines(resolution="110m")
        plt.show()
        fig_out = fig
        ax_out = ax

    if plot and return_polygon:
        return matches, fig_out, ax_out
    elif return_polygon:
        return matches


# Example usage:
# With plot: returns (polygon_gdf, fig, ax)
matches, fig, ax = lookup_admin3((12.5, 41.9), buffer_m=0, plot=True)  # Rome approx.
# Without plot: returns just polygon_gdf
# matches = lookup_admin3((12.5, 41.9), buffer_m=0, plot=False)
