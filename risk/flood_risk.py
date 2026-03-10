import geopandas as gpd
import xarray as xr
from shapely.geometry import Point
from pathlib import Path
import numpy as np

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

WEATHER_DIR = BASE_DIR / "data" / "weather" / "raw"
GIS_DIR = BASE_DIR / "data" / "gis"


# -----------------------------
# Weather Loading
# -----------------------------

def load_rainfall_year(year=2020):
    """
    Find rainfall NetCDF file for a given year automatically.
    """
    matches = list(WEATHER_DIR.glob(f"*{year}*.nc"))

    if not matches:
        raise FileNotFoundError(
            f"No rainfall file for {year} found in {WEATHER_DIR}"
        )

    path = matches[0]
    print(f"Using rainfall file: {path.name}")

    ds = xr.open_dataset(path)
    return ds


def clip_to_pune(ds):
    """Clip IMD rainfall dataset to Pune bounding box."""
    return ds.sel(
        LATITUDE=slice(18.3, 18.8),
        LONGITUDE=slice(73.6, 74.1)
    )


# -----------------------------
# GIS Loading
# -----------------------------

def load_roads():
    roads = gpd.read_file(GIS_DIR / "roads_final.geojson")

    # Convert to lat/lon CRS if needed
    if roads.crs is None or roads.crs.to_epsg() != 4326:
        roads = roads.to_crs(epsg=4326)

    return roads


# -----------------------------
# Rain Sampling
# -----------------------------

def sample_rainfall_timeseries(ds, lat, lon):
    """Return full rainfall time series at nearest grid cell."""
    vals = ds["RAINFALL"].sel(
        LATITUDE=lat,
        LONGITUDE=lon,
        method="nearest"
    )
    return vals.values


# -----------------------------
# Flood Risk Scoring
# -----------------------------

def compute_flood_risk(series):
    """
    Convert rainfall time series into flood risk score [0,1].
    Uses percentile-based heuristic.
    """
    p90 = np.nanpercentile(series, 90)
    p99 = np.nanpercentile(series, 99)

    if p99 == 0:
        return 0.0

    risk = p90 / p99
    return float(np.clip(risk, 0, 1))


# -----------------------------
# Main Attachment Function
# -----------------------------

def attach_rainfall_to_roads(year=2020):

    print(f"Loading rainfall for {year}...")

    ds = load_rainfall_year(year)
    pune_ds = clip_to_pune(ds)

    print("Loading roads GIS...")
    roads = load_roads()

    rainfall_means = []
    flood_risks = []

    for geom in roads.geometry:

        if geom is None or geom.is_empty:
            rainfall_means.append(np.nan)
            flood_risks.append(np.nan)
            continue

        # MultiLineString → take longest piece
        if geom.geom_type == "MultiLineString":
            geom = max(geom.geoms, key=lambda g: g.length)

        if geom.geom_type == "LineString":
            mid = geom.interpolate(0.5, normalized=True)

        elif geom.geom_type == "Point":
            mid = geom

        else:
            rainfall_means.append(np.nan)
            flood_risks.append(np.nan)
            continue

        series = sample_rainfall_timeseries(
            pune_ds, mid.y, mid.x
        )

        mean_rain = float(np.nanmean(series))
        risk = compute_flood_risk(series)

        rainfall_means.append(mean_rain)
        flood_risks.append(risk)

    roads["rain_mm_mean"] = rainfall_means
    roads["flood_risk"] = flood_risks

    print("Rainfall + flood risk attached to roads.")

    return roads


# -----------------------------
# Run Standalone
# -----------------------------

if __name__ == "__main__":

    roads = attach_rainfall_to_roads(year=2020)

    print("\nPreview:")
    print(roads[["rain_mm_mean", "flood_risk"]].head())
