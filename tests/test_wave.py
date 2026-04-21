from hresopt.data_loader.energy_loader import define_components, load_resources
from hresopt.energy_generation.wave_power import compute_wave_power


# =========================================================
# CONFIG
# =========================================================
WAVE_FILE = "data/wave.nc"
MATRIX_FILE = "data/wave_matrix.csv"

LAT = 28
LON = -18


# =========================================================
# TEST FUNCTION
# =========================================================
def test_wave_power():

    print("Defining components...")
    components = define_components(
        wind=False,
        wave=True,
        battery=True
    )

    print("Loading wave resource...")
    resources, meta = load_resources(
        components,
        wave_path=WAVE_FILE,
        lat=LAT,
        lon=LON
    )

    df_wave = resources["wave"]

    print("\n--- RAW WAVE DATA ---")
    print(df_wave.head())
    print(meta["wave"])

    print("\nComputing wave power...")
    df_wave = compute_wave_power(
        df_wave,
        matrix_path=MATRIX_FILE
    )

    print("\n--- RESULT ---")
    print(df_wave.head())

    print("\n--- BASIC CHECKS ---")

    # Check columns
    assert "wave_power" in df_wave.columns, "Missing wave_power column"

    # Check non-negative power
    assert (df_wave["wave_power"] >= 0).all(), "Negative wave power detected"

    # Check variation (not constant)
    assert df_wave["wave_power"].nunique() > 1, "Wave power seems constant"

    print("All checks passed ✅")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    test_wave_power()