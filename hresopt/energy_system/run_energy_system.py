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
# MAIN RUN FUNCTION
# =========================================================
def run_energy_system():

    components = define_components(
        wind=True,
        wave=True,
        battery=True
    )

    # Load resources
    resources, _ = load_resources(
        components,
        wind_path=WIND_FILE,
        wave_path=WAVE_FILE,
        lat=LAT,
        lon=LON
    )

    df_wind = resources["wind"]
    df_wave = resources["wave"]

    # Compute power
    df_wind = compute_wind_power(df_wind, CURVE_FILE, Z_HUB)
    df_wave = compute_wave_power(df_wave, MATRIX_FILE)

    # Load demand
    df_demand, _ = load_demand(DEMAND_FILE)

    # Prepare inputs
    wind_power = df_wind["wind_power"].values
    wave_power = df_wave["wave_power"].values
    energy_demand = df_demand["demand"].values

    # Run simulation
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

    # Attach time series back
    df = df_wind.copy()
    df["wave_power"] = df_wave["wave_power"].values
    df["energy_met_ratio"] = results["energy_met_ratio_ts"]
    df["SOC"] = results["SOC_ts"]

    return df, results