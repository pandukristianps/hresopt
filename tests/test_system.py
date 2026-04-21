from hresopt.data_loader.energy_loader import define_components, load_resources
from hresopt.data_loader.demand_loader import load_demand
from hresopt.energy_generation.wind_power import compute_wind_power
from hresopt.energy_generation.wave_power import compute_wave_power
from hresopt.energy_system.energy_system import simulate_energy_system


# =========================================================
# CONFIG
# =========================================================
WIND_FILE = "data/wind.nc"
CURVE_FILE = "data/wind_curve.csv"

WAVE_FILE = "data/wave.nc"
MATRIX_FILE = "data/wave_matrix.csv"

DEMAND_FILE = "data/demand_elhierro.csv"

LAT = 28
LON = -18

Z_HUB = 136

NUM_WIND = 17
NUM_WAVE = 308
BATT_CAP = 2126400

INIT_SOC = 0


# =========================================================
# TEST FUNCTION
# =========================================================
def test_energy_system():

    print("Defining components...")
    components = define_components(
        wind=True,
        wave=True,
        battery=True
    )

    print("\nLoading wind and wave resources...")
    resources, meta = load_resources(
        components,
        wind_path=WIND_FILE,
        wave_path=WAVE_FILE,
        lat=LAT,
        lon=LON
    )

    df_wind = resources["wind"]
    df_wave = resources["wave"]

    print("\nComputing wind power...")
    df_wind = compute_wind_power(
        df_wind,
        curve_path=CURVE_FILE,
        z_hub=Z_HUB,
        params=None
    )

    print("\nComputing wave power...")
    df_wave = compute_wave_power(
        df_wave,
        matrix_path=MATRIX_FILE
    )

    print("\nLoading demand data...")
    df_demand, meta = load_demand(DEMAND_FILE)

    # =========================================================
    # PREPARE INPUTS
    # =========================================================
    wind_power = df_wind["wind_power"].values

    # Temporary: no wave yet
    wave_power = df_wave["wave_power"].values

    # Simple demand (constant or scaled)
    energy_demand = df_demand["demand"]

    print("\n--- INPUT SUMMARY ---")
    print(f"Mean wind power: {wind_power.mean():.2f} kW")
    print(f"Mean wave power: {wave_power.mean():.2f} kW")
    print(f"Mean demand: {energy_demand.mean():.2f} kW")

    # =========================================================
    # RUN SYSTEM
    # =========================================================
    print("\nSimulating energy system...")

    results = simulate_energy_system(
        wind_power=wind_power,
        wave_power=wave_power,
        energy_demand=energy_demand,
        num_wind=NUM_WIND,
        num_wave=NUM_WAVE,
        batt_cap=BATT_CAP,
        params=None,
        init_soc=INIT_SOC
    )

    print("\n--- RESULTS ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("\n--- BASIC CHECKS ---")

    assert results["LCOE"] > 0, "LCOE should be positive"
    assert 0 <= results["LPSP"] <= 1, "LPSP out of bounds"
    assert 0 <= results["SOC_final"] <= 1, "SOC out of bounds"
    assert results["energy_met"] >= 0, "Negative energy met"

    print("All checks passed ✅")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    test_energy_system()