import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

RAW_DIR = Path("data/weather/raw")


def load_year(filepath):
    """Load one IMD NetCDF rainfall file."""
    ds = xr.open_dataset(filepath)
    print(ds)
    return ds


def preview_year(filepath):
    ds = load_year(filepath)

    # Print variable names
    for var in ds.data_vars:
        print("Variable:", var)

    var_name = list(ds.data_vars)[0]

    # Use correct dimension name
    rain = ds[var_name].isel(TIME=0)

    rain.plot()
    plt.title(f"Rainfall snapshot: {filepath.name}")
    plt.show()

def clip_to_pune(ds):
    """
    Clip IMD rainfall dataset to Pune bounding box.
    """
    pune_ds = ds.sel(
        LATITUDE=slice(18.3, 18.8),
        LONGITUDE=slice(73.6, 74.1)
    )
    return pune_ds


if __name__ == "__main__":
    files = sorted(RAW_DIR.glob("*.nc"))

    print("Found files:")
    for f in files[:5]:
        print(" -", f.name)

    ds = load_year(files[0])

    pune_ds = clip_to_pune(ds)

    print("\nClipped to Pune:")
    print(pune_ds)

    pune_ds["RAINFALL"].isel(TIME=0).plot()
    plt.title("Pune rainfall snapshot")
    plt.show()
