from hresopt.data_loader.energy_loader import define_components, load_resources
from hresopt.energy_generation.wind_power import compute_wind_power


# =========================================================
# CONFIG
# =========================================================
WIND_FILE = "data/wind.nc"
CURVE_FILE = "data/wind_curve.csv"

LAT = 28
LON = -18

Z_HUB = 136


# =========================================================
# TEST FUNCTION
# =========================================================
def test_wind_power():

    print("Defining components...")
    components = define_components(
        wind=True,
        wave=False,
        battery=True
    )

    print("Loading wind resource...")
    resources, meta = load_resources(
        components,
        wind_path=WIND_FILE,
        lat=LAT,
        lon=LON
    )

    df_wind = resources["wind"]

    print("\n--- RAW WIND DATA ---")
    print(df_wind.head())
    print(meta["wind"])

    print("\nComputing wind power...")
    df_wind = compute_wind_power(
        df_wind,
        curve_path=CURVE_FILE,
        z_hub=Z_HUB,
        params=None
    )

    print("\n--- RESULT ---")
    print(df_wind.head())

    print("\n--- BASIC CHECKS ---")

    # Check columns
    assert "wind_speed_hub" in df_wind.columns, "Missing wind_speed_hub"
    assert "wind_power" in df_wind.columns, "Missing power column"

    # Check non-negative power
    assert (df_wind["wind_power"] >= 0).all(), "Negative power detected"

    # Check presence of zero power (cut-in / cut-out)
    assert (df_wind["wind_power"] == 0).any(), "No zero power values detected"

    print("All checks passed ✅")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    test_wind_power()