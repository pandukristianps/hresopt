import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from hresopt.constants import SystemParams


# =========================================================
# LOAD WIND POWER CURVE
# =========================================================
def load_power_curve(path):
    """
    Load wind turbine power curve from CSV.

    Expected columns:
    - wind_speed (m/s)
    - power (kW)
    """

    df = pd.read_csv(path)

    required_cols = ["wind_speed", "power"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Power curve CSV must contain {required_cols}. "
                f"Found: {list(df.columns)}"
            )

    # sort to ensure proper interpolation
    df = df.sort_values("wind_speed").reset_index(drop=True)

    return df


# =========================================================
# HEIGHT CORRECTION FOR WIND SPEED
# =========================================================
def extrapolate_wind_speed(wind_speed, z_hub, z_ref=100, alpha=None, params: SystemParams = None):
    """
    Extrapolate wind speed to hub height using power law.

    Parameters
    ----------
    wind_speed : pd.Series
        Wind speed at reference height (m/s)
    z_ref : float
        Reference height (m)
    z_hub : float
        Hub height (m)
    alpha : float
        Shear exponent

    Returns
    -------
    pd.Series
        Wind speed at hub height (m/s)
    """

    if alpha is None:
        alpha = params.physical.shear_exponent

    factor = (z_hub / z_ref) ** alpha
    return wind_speed * factor


# =========================================================
# WIND POWER CALCULATION USING INTERPOLATION
# =========================================================
def power_from_curve(wind_speed, curve):
    """
    Convert wind speed time series to power using interpolation.

    Parameters
    ----------
    wind_speed : pd.Series
        Wind speed (m/s)
    curve : pd.DataFrame
        Power curve with columns [wind_speed, power]

    Returns
    -------
    pd.Series
        Power output (kW)
    """

    # sanity check
    if wind_speed.max() > 100:
        raise ValueError("Wind speed seems too high. Expected m/s.")

    v_curve = curve["wind_speed"].values
    p_curve = curve["power"].values

    # Create interpolator
    interpolate = interp1d(
        v_curve,
        p_curve,
        bounds_error=False,
        fill_value=0.0 
    )

    # Apply interpolation
    power = interpolate(wind_speed.values)

    return pd.Series(power, index=wind_speed.index)


# =========================================================
# COMPLETE WIND POWER CALCULATION (for use in energy_systems)
# =========================================================
def compute_wind_power(
    df,
    curve_path,
    z_hub,
    params: SystemParams = None,
    z_ref=100,
):
    """
    Full wind power calculation pipeline:
    1. Load power curve
    2. Height correction
    3. Power interpolation

    Parameters
    ----------
    df : pd.DataFrame
        Must contain column 'wind_speed'
    curve_path : str
        Path to power curve CSV
    z_ref : float
        Reference height (m)
    z_hub : float
        Hub height (m)
    alpha : float
        Shear exponent

    Returns
    -------
    pd.DataFrame
        Original df with:
        - wind_speed_hub
        - power
    """

    if params is None:
        params = SystemParams()

    if "wind_speed" not in df.columns:
        raise ValueError("DataFrame must contain 'wind_speed' column")

    df = df.copy()

    curve = load_power_curve(curve_path)

    alpha = params.physical.shear_exponent

    df["wind_speed_hub"] = extrapolate_wind_speed(
        df["wind_speed"],
        z_hub=z_hub,
        z_ref=z_ref,
        alpha=alpha
    )

    df["wind_power"] = power_from_curve(
        df["wind_speed_hub"],
        curve
    )

    return df