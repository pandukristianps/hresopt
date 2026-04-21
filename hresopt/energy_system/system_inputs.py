import numpy as np

from hresopt.data_loader.energy_loader import define_components, load_resources
from hresopt.data_loader.demand_loader import load_demand
from hresopt.energy_generation.wind_power import compute_wind_power
from hresopt.energy_generation.wave_power import compute_wave_power


def system_inputs(
    WIND_FILE=None,
    WAVE_FILE=None,
    DEMAND_FILE=None,
    CURVE_FILE=None,
    MATRIX_FILE=None,
    LAT=None,
    LON=None,
    Z_HUB=None,
):
    # =========================
    # FLAGS
    # =========================
    use_wind = WIND_FILE is not None
    use_wave = WAVE_FILE is not None

    if not use_wind and not use_wave:
        raise ValueError("At least one of WIND_FILE or WAVE_FILE must be provided")
    if DEMAND_FILE is None:
        raise ValueError("DEMAND_FILE cannot be None")
    if (use_wind or use_wave) and (LAT is None or LON is None):
        raise ValueError("LAT and LON must be provided for resource loading") 
    if use_wind and CURVE_FILE is None:
        warnings.warn("Please provide CURVE_FILE for wind power computation")
    if use_wave and MATRIX_FILE is None:
        warnings.warn("Please provide MATRIX_FILE for wind power computation")

    components = define_components(
        wind=use_wind,
        wave=use_wave,
        battery=True
    )

    resources, meta = load_resources(
        components,
        wind_path=WIND_FILE,
        wave_path=WAVE_FILE,
        lat=LAT,
        lon=LON
    )

    if use_wind:
        df_wind = resources["wind"]
        df_wind = compute_wind_power(df_wind, CURVE_FILE, Z_HUB)
        wind_power = df_wind["wind_power"].values
    else:
        wind_power = None

    if use_wave:
        df_wave = resources["wave"]
        df_wave = compute_wave_power(df_wave, MATRIX_FILE)
        wave_power = df_wave["wave_power"].values
    else:
        wave_power = None

    df_demand, _ = load_demand(DEMAND_FILE)
    energy_demand = df_demand["demand"].values

    return wind_power, wave_power, energy_demand