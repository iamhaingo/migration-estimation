import requests
import geopandas as gpd
import os
from shapely.geometry import MultiPoint
from pathlib import Path

from config import API_TOKEN


def fetch_geojson(api_token: str, limit: int = 20) -> gpd.GeoDataFrame:
    url = "https://helix-tools-api.idmcdb.org/external-api/gidd/disaggregations/disaggregation-geojson/"
    r = requests.get(url, params={"client_id": api_token, "limit": limit})
    r.raise_for_status()
    return gpd.GeoDataFrame.from_features(r.json()["features"])


def update_geometry_to_centroid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf["geometry"] = gdf["geometry"].apply(
        lambda x: x.centroid if hasattr(x, "centroid") else x
    )
    gdf = gdf.set_geometry("geometry", crs="EPSG:4326")
    return gdf


def compute_centroid(points):
    multipoint = MultiPoint(points)
    return multipoint.centroid


def group_flood_events_by_centroid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grouped = gdf.groupby(
        ["ISO3", "Event start date", "Event end date", "Total figures"], as_index=False
    ).agg({"geometry": lambda geoms: compute_centroid(list(geoms))})

    grouped = grouped.set_geometry("geometry")
    grouped.set_crs("EPSG:4326", inplace=True)
    return grouped


def filter_and_save_disasters(gdf: gpd.GeoDataFrame, output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)

    gdf_disasters = gdf[
        (gdf["Figure cause"] == "Disaster") & (gdf["Total figures"] != 0)
    ]

    gdf_flood = gdf_disasters[gdf_disasters["Hazard type"] == "Flood"]

    gdf_disasters = group_flood_events_by_centroid(gdf_disasters)
    gdf_flood = group_flood_events_by_centroid(gdf_flood)

    gdf_flood.to_parquet(os.path.join(output_dir, "gd_flood.parquet"))
    gdf_disasters.to_parquet(os.path.join(output_dir, "gd_disasters.parquet"))


def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    output_dir = SCRIPT_DIR / ".." / "data" / "raw"

    gdf = fetch_geojson(API_TOKEN, limit=20)
    gdf = update_geometry_to_centroid(gdf)
    filter_and_save_disasters(gdf, output_dir=output_dir)

    print(f"Data collection completed and saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
