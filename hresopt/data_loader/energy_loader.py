import xarray as xr
import pandas as pd
import numpy as np

# =========================================================
# DEFINE ENERGY SYSTEM COMPONENTS
# =========================================================
def define_components(
    wind=False,
    wave=False,
    solar=False,
    geothermal=False,
    battery=False,
    hydrogen=False
):
    """
    Define desired energy sources and storage in the system.
    """

    components = {
        "wind": wind,
        "wave": wave,
        "solar": solar,
        "geothermal": geothermal,
        "battery": battery,
        "hydrogen": hydrogen
    }

    # --- validation: at least one energy source ---
    if not (wind or wave or solar or geothermal):
        raise ValueError("At least one energy source must be selected.")

    # --- validation: at least one storage ---
    if not (battery or hydrogen):
        raise ValueError("At least one energy storage must be selected.")

    return components

# =========================================================
# LOAD WIND DATA
# =========================================================
def load_wind(path, height="100m", lat=None, lon=None):
    """
    Load wind data from NetCDF (ERA5) or CSV.

    Returns:
    - df: pandas DataFrame with columns [time, wind_speed]
    - meta: dictionary with metadata (lat, lon, source)
    """

    # --- CASE 1: NetCDF ---
    if path.endswith(".nc"):

        ds = xr.open_dataset(path)

        if lat is None or lon is None:
            raise ValueError(
                "For NetCDF input, 'lat' and 'lon' must be provided."
            )

        ds = ds.sel(latitude=lat, longitude=lon, method="nearest")

        # select height
        if height == "10m":
            u = ds["u10"]
            v = ds["v10"]
        elif height == "100m":
            u = ds["u100"]
            v = ds["v100"]
        else:
            raise ValueError("height must be '10m' or '100m'")

        # compute wind speed resultant
        wind_speed = np.sqrt(u**2 + v**2)

        # convert to DataFrame
        df = wind_speed.to_dataframe(name="wind_speed").reset_index()
        df = df.rename(columns={"valid_time": "time"})
        df = df[["time", "wind_speed"]]

        # extract actual coordinates
        try:
            lat_val = float(ds.latitude.values)
            lon_val = float(ds.longitude.values)
        except:
            lat_val, lon_val = None, None

        meta = {
            "lat": lat_val,
            "lon": lon_val,
            "source": "ERA5"
        }

        return df, meta

    # --- CASE 2: CSV ---
    elif path.endswith(".csv"):

        df = pd.read_csv(path)

        required_columns = ["time", "wind_speed"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(
                    f"CSV must contain column '{col}'. Found: {list(df.columns)}"
                )

        meta = {
            "lat": None,
            "lon": None,
            "source": "CSV"
        }

        return df, meta

    else:
        raise ValueError("Unsupported file format")

# =========================================================
# LOAD WAVE DATA
# =========================================================
def load_wave(path, lat=None, lon=None):
    """
    Load wave data from NetCDF (ERA5) or CSV.

    Returns:
    - df: pandas DataFrame with columns [time, swh, pp1d]
    - meta: dictionary with metadata (lat, lon, source)
    """

    # --- CASE 1: NetCDF ---
    if path.endswith(".nc"):

        ds = xr.open_dataset(path)

        # select coordinate if provided
        if lat is not None and lon is not None:
            ds = ds.sel(latitude=lat, longitude=lon, method="nearest")

        # check required variables
        if "swh" not in ds or "pp1d" not in ds:
            raise ValueError("ERA5 file must contain 'swh' and 'pp1d'")

        df = ds[["swh", "pp1d"]].to_dataframe().reset_index()
        df = df.rename(columns={"valid_time": "time"})
        df = df[["time", "swh", "pp1d"]]

        # extract actual coordinates
        try:
            lat_val = float(ds.latitude.values)
            lon_val = float(ds.longitude.values)
        except:
            lat_val, lon_val = None, None

        meta = {
            "lat": lat_val,
            "lon": lon_val,
            "source": "ERA5"
        }

        return df, meta

    # --- CASE 2: CSV ---
    elif path.endswith(".csv"):

        df = pd.read_csv(path)

        required_columns = ["time", "swh", "mwp"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(
                    f"CSV must contain column '{col}'. Found: {list(df.columns)}"
                )

        meta = {
            "lat": None,
            "lon": None,
            "source": "CSV"
        }

        return df, meta

    else:
        raise ValueError("Unsupported file format")

# =========================================================
# LOAD RESOURCES SIMULTANEOUSLY
# =========================================================
def load_resources(
    components,
    wind_path=None,
    wave_path=None,
    height="100m",
    lat=None,
    lon=None
):
    """
    Load all required resources based on selected components.

    Returns:
    - resources: dict of data
    - metadata: dict of metadata for each resource
    """

    resources = {}
    metadata = {}

    # --- WIND ---
    if components.get("wind"):
        if wind_path is None:
            raise ValueError("wind_path must be provided when wind=True")

        wind, wind_meta = load_wind(
            wind_path,
            lat=lat,
            lon=lon
        )

        resources["wind"] = wind
        metadata["wind"] = wind_meta

    # --- WAVE ---
    if components.get("wave"):
        if wave_path is None:
            raise ValueError("wave_path must be provided when wave=True")

        wave, wave_meta = load_wave(
            wave_path,
            lat=lat,
            lon=lon
        )

        resources["wave"] = wave
        metadata["wave"] = wave_meta

    return resources, metadata