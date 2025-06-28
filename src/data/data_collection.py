import requests
import geopandas as gpd
from config import API_TOKEN


def fetch_geojson(api_token: str, limit: int = 20) -> gpd.GeoDataFrame:
    url = "https://helix-tools-api.idmcdb.org/external-api/gidd/disaggregations/disaggregation-geojson/"
    r = requests.get(url, params={"client_id": api_token, "limit": limit})
    r.raise_for_status()
    return gpd.GeoDataFrame.from_features(r.json()["features"])


# df = fetch_geojson(API_TOKEN, limit=20)
