import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# =========================================================
# LOAD WAVE POWER MATRIX
# =========================================================
def load_power_matrix(path):
    """
    Load wave energy converter power matrix from CSV.

    Expected format:
    - Rows: significant wave height Hs (m)
    - Columns: peak period Tp (s)
    - Values: power (kW)
    """

    df = pd.read_csv(path, index_col=0)

    try:
        df.index = df.index.astype(float)     # Hs
        df.columns = df.columns.astype(float) # Tp
    except:
        raise ValueError("Wave matrix index/columns must be numeric (Hs, Tp)")

    df = df.sort_index().sort_index(axis=1)

    return df


# =========================================================
# WAVE POWER CALCULATION USING INTERPOLATION
# =========================================================
def power_from_matrix(hs, tp, matrix):
    """
    Compute wave power using 2D interpolation.

    Parameters
    ----------
    hs : pd.Series
        Significant wave height (m)
    tp : pd.Series
        Peak period (s)
    matrix : pd.DataFrame
        Power matrix (Hs x Tp)

    Returns
    -------
    pd.Series
        Power output (kW)
    """

    if len(hs) != len(tp):
        raise ValueError("hs and tp must have the same length")

    hs_vals = matrix.index.values
    tp_vals = matrix.columns.values
    matrix_vals = matrix.values

    # Create interpolator
    interpolate = RegularGridInterpolator(
        (hs_vals, tp_vals),
        matrix_vals,
        bounds_error=False,
        fill_value=0
    )

    # Prepare input points
    points = np.column_stack((hs.values, tp.values))

    power = interpolate(points)

    return pd.Series(power, index=hs.index)


# =========================================================
# COMPLETE WAVE POWER CALCULATION
# =========================================================
def compute_wave_power(df, matrix_path):
    """
    Full wave power calculation pipeline:
    1. Load power matrix
    2. Compute power using Hs and Tp

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        - 'swh'  (m)
        - 'pp1d' (s)

    matrix_path : str
        Path to wave power matrix CSV

    Returns
    -------
    pd.DataFrame
        Original df with:
        - wave_power
    """

    required_cols = ["swh", "pp1d"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain {required_cols}")

    df = df.copy()

    matrix = load_power_matrix(matrix_path)

    df["wave_power"] = power_from_matrix(
        df["swh"],
        df["pp1d"],
        matrix
    )

    return df