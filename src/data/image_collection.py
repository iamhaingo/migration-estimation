import ee
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import geemap
from pathlib import Path
from shapely.geometry import MultiPoint
from skimage.transform import resize
from config import PROJECT_ID

ee.Authenticate()
ee.Initialize(project=PROJECT_ID)

folder_path: Path = Path("../data/raw/")
path_parquet_flood = folder_path / "gd_flood.parquet"
path_parquet_disasters = folder_path / "gd_disasters.parquet"

# Read IDMC files
gdf_flood = gpd.read_parquet(path_parquet_flood)
gdf_disasters = gpd.read_parquet(path_parquet_disasters)

# parameters
buffer_distance = 10000
target_size = (330, 330)
viirs_scale = 500
landsat_scale = 30


def get_nightlight(
    point_index, dataframe, buffer_distance=buffer_distance, start_or_end="Start date"
):
    point = dataframe.iloc[point_index]
    ee_point = ee.Geometry.Point([point.geometry.x, point.geometry.y])
    buffer_geom = ee_point.buffer(buffer_distance)

    bounds_rect = buffer_geom.bounds()

    date_type = point[start_or_end]
    date_start = date_type - pd.Timedelta(days=4)
    date_end = date_type + pd.Timedelta(days=3)

    nightlight_collection = (
        ee.ImageCollection("NOAA/VIIRS/001/VNP46A1")
        .filterDate(date_start, date_end)
        .select("DNB_At_Sensor_Radiance_500m")
        .map(lambda image: image.clip(bounds_rect))
    )

    return nightlight_collection, buffer_geom, bounds_rect


def resize_image(img_arr, target_size):
    return resize(img_arr, target_size, anti_aliasing=False)


def process_image(image_obj, region_info, scale, target_size):
    """Helper function to process a single image."""
    image = ee.Image(image_obj)
    date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()

    img_arr = geemap.ee_to_numpy(image, region=region_info, scale=scale)
    resized_img = resize_image(img_arr, target_size=target_size)

    return resized_img, date, img_arr.flatten()


def scale_landsat(img):
    optical = img.select("SR_B.").multiply(0.0000275).add(-0.2)
    return img.addBands(optical, None, True)


def cloud_mask_landsat9(image):
    qa = image.select("QA_PIXEL")
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(cloud_mask)


def mask_viirs_clouds(image):
    cloud_mask = image.select("QF_Cloud_Mask")
    cloud_confidence = cloud_mask.rightShift(6).bitwiseAnd(3)
    clear_mask = cloud_confidence.lte(1)
    return image.updateMask(clear_mask)


def ensure_crs_alignment(gdf, target_crs="EPSG:4326"):
    """Ensure GeoDataFrame is in the target CRS (WGS84 by default for Earth Engine)"""
    if gdf.crs != target_crs:
        print(f"Converting CRS from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)
    return gdf


def create_matched_geometry(point, buffer_distance, target_crs="EPSG:4326"):
    if hasattr(point, "to_crs"):
        point_wgs84 = point.to_crs(target_crs)
        ee_point = ee.Geometry.Point([point_wgs84.geometry.x, point_wgs84.geometry.y])
    else:
        ee_point = ee.Geometry.Point([point.geometry.x, point.geometry.y])

    buffer_geom = ee_point.buffer(buffer_distance)
    bounds_rect = buffer_geom.bounds()

    return ee_point, buffer_geom, bounds_rect


def get_landsat9_rgb_median_numpy(
    point_index,
    gdf,
    buffer_distance=buffer_distance,
    start_or_end="Start date",
    scale=landsat_scale,
    target_size=target_size,
    target_crs="EPSG:4326",
):
    gdf_aligned = ensure_crs_alignment(gdf, target_crs)

    point = gdf_aligned.iloc[point_index]
    _, _, bounds_rect = create_matched_geometry(point, buffer_distance)

    # start and end dates , 6 months window
    start_date = point["Start date"]
    end_date = point["End date"]
    if isinstance(start_date, str) or isinstance(end_date, str):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    date = start_date + (end_date - start_date) / 2  # Use midpoint for the date
    date_start = (date - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    date_end = (date + pd.Timedelta(days=90)).strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterDate(date_start, date_end)
        .filterBounds(bounds_rect)
        .filter(ee.Filter.lt("CLOUD_COVER", 60))
        .map(cloud_mask_landsat9)
        .map(scale_landsat)
    )

    collection_size = collection.size().getInfo()
    if collection_size == 0:
        print(
            f"No Landsat images found for point {point_index} between {date_start} and {date_end}"
        )
        return None, None

    composite = collection.median().clip(bounds_rect)
    rgb = composite.select(["SR_B4", "SR_B3", "SR_B2"])
    date_str = f"{date_start}_to_{date_end}"

    try:
        np_image = geemap.ee_to_numpy(rgb, region=bounds_rect, scale=scale)

        if np_image.ndim != 3:
            print(
                f"Unexpected image dimensions for point {point_index}: {np_image.shape}"
            )
            return None, None

        resized = resize_image(np_image, target_size=target_size)

        return resized, date_str

    except Exception as e:
        print(f"Error processing Landsat data for point {point_index}: {str(e)}")
        return None, None


def get_viirs_nightlight_median_numpy(
    point_index,
    gdf,
    buffer_distance=buffer_distance,
    start_or_end="Start date",
    scale=viirs_scale,
    target_size=target_size,
    target_crs="EPSG:4326",
):
    gdf_aligned = ensure_crs_alignment(gdf, target_crs)

    point = gdf_aligned.iloc[point_index]
    _, _, bounds_rect = create_matched_geometry(point, buffer_distance)

    date = point[start_or_end]
    if isinstance(date, str):
        date = pd.to_datetime(date)

    date_start = (date - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    date_end = (date + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection("NOAA/VIIRS/001/VNP46A1")
        .filterDate(date_start, date_end)
        .filterBounds(bounds_rect)
        .map(mask_viirs_clouds)
    )

    collection_size = collection.size().getInfo()
    if collection_size == 0:
        print(
            f"No VIIRS images found for point {point_index} between {date_start} and {date_end}"
        )
        return None, None

    composite = collection.median().clip(bounds_rect)
    date_str = f"{date_start}_to_{date_end}"

    try:
        np_image = geemap.ee_to_numpy(
            composite.select("DNB_At_Sensor_Radiance_500m"),
            region=bounds_rect,
            scale=scale,
        )

        if np_image.ndim == 3 and np_image.shape[-1] == 1:
            np_image = np_image[:, :, 0]

        resized = resize_image(np_image, target_size=target_size)

        return resized, date_str

    except Exception as e:
        print(f"Error processing VIIRS data for point {point_index}: {str(e)}")
        return None, None
