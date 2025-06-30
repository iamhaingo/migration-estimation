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

gdf_flood = gpd.read_parquet(path_parquet_flood)
gdf_disasters = gpd.read_parquet(path_parquet_disasters)
