import time
import numpy as np
import pandas as pd

from hresopt.data_loader.energy_loader import define_components, load_resources
from hresopt.data_loader.demand_loader import load_demand
from hresopt.energy_generation.wind_power import compute_wind_power
from hresopt.energy_generation.wave_power import compute_wave_power
from hresopt.metaheuristics.de import run_de


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

N_RUNS = 5
POP_SIZE = 10
NUM_ITER = 20
LPSP_TARGET = 0.05


# =========================================================
# TEST FUNCTION
# =========================================================
def test_de():

    print("Defining components...")
    components = define_components(wind=True, wave=True, battery=True)

    print("\nLoading wind and wave resources...")
    resources, _ = load_resources(
        components,
        wind_path=WIND_FILE,
        wave_path=WAVE_FILE,
        lat=LAT,
        lon=LON
    )

    df_wind = resources["wind"]
    df_wave = resources["wave"]

    print("\nComputing wind power...")
    df_wind = compute_wind_power(df_wind, CURVE_FILE, Z_HUB)

    print("\nComputing wave power...")
    df_wave = compute_wave_power(df_wave, MATRIX_FILE)

    print("\nLoading demand data...")
    df_demand, _ = load_demand(DEMAND_FILE)

    # =========================================================
    # INPUTS
    # =========================================================
    wind_power = df_wind["wind_power"].values
    wave_power = df_wave["wave_power"].values
    energy_demand = df_demand["demand"].values

    # =========================================================
    # MULTIPLE RUNS
    # =========================================================
    print("\nRunning DE multiple times...")

    start_time = time.time()

    all_history_best = []
    all_best_lcoe = []
    all_best_lpsp = []
    all_best_config = []

    for run in range(N_RUNS):

        print(f"\nRun {run+1}/{N_RUNS}")

        results = run_de(
            wind_power=wind_power,
            wave_power=wave_power,
            energy_demand=energy_demand,
            population_size=POP_SIZE,
            num_iterations=NUM_ITER,
            LPSP_target=LPSP_TARGET,
            random_seed=run
        )

        history_best = np.array(results["history_best"])  # (iter, 4)
        all_history_best.append(history_best[:, 3])  # score (≈ LCOE if feasible)
        all_best_lcoe.append(results["LCOE"])
        all_best_lpsp.append(results["LPSP"])
        all_best_config.append(results["best_config"])

    # =========================================================
    # POST-PROCESS
    # =========================================================
    all_history_best = np.array(all_history_best)  # (runs, iterations)
    
    best_idx = np.argmin(all_best_lcoe)
    best_lcoe = all_best_lcoe[best_idx]
    best_lpsp = all_best_lpsp[best_idx]
    best_config = all_best_config[best_idx]

    median_lcoe = np.median(all_history_best, axis=0)

    # =========================================================
    # TIME
    # =========================================================
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nTotal runtime: {elapsed_time:.2f} seconds")
    print("\n=== GLOBAL BEST SOLUTION ===")
    print(f"Wind: {best_config[0]}")
    print(f"Wave: {best_config[1]}")
    print(f"Battery: {best_config[2]}")
    print(f"LCOE: {best_lcoe}")
    print(f"LPSP: {best_lpsp}")

    # =========================================================
    # SAVE RESULTS
    # =========================================================
    df_conv = pd.DataFrame({
        "iteration": np.arange(len(median_lcoe)),
        "median_LCOE": median_lcoe
    })

    df_conv.to_csv("results_de_convergence.csv", index=False)

    print("\nResults saved to results_de_convergence.csv")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    test_de()