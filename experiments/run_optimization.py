import time
import numpy as np

from hresopt.data_loader.energy_loader import define_components, load_resources
from hresopt.data_loader.demand_loader import load_demand
from hresopt.energy_generation.wind_power import compute_wind_power
from hresopt.energy_generation.wave_power import compute_wave_power

from hresopt.metaheuristics.nr_aco import run_nr_aco
from hresopt.metaheuristics.s_aco import run_s_aco
from hresopt.metaheuristics.ga import run_ga


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_optimization(
    WIND_FILE,
    WAVE_FILE,
    DEMAND_FILE,
    CURVE_FILE,
    MATRIX_FILE,
    LAT,
    LON,
    Z_HUB,

    # Optimization settings
    N_RUNS,
    num_iterations,
    population_size,

    LPSP_target=0.05,
    random_seed_base=5,
):

    print("Defining components...")
    components = define_components(wind=True, wave=True, battery=True)

    print("\nLoading resources...")
    resources, meta = load_resources(
        components,
        wind_path=WIND_FILE,
        wave_path=WAVE_FILE,
        lat=LAT,
        lon=LON
    )

    df_wind = resources["wind"]
    print("\nWind resources coordinate: ", meta["wind"])

    df_wave = resources["wave"]
    print("Wave resources coordinate: ", meta["wave"])

    print("\nComputing wind power...")
    df_wind = compute_wind_power(df_wind, CURVE_FILE, Z_HUB)

    print("\nComputing wave power...")
    df_wave = compute_wave_power(df_wave, MATRIX_FILE)

    print("\nLoading demand...")
    df_demand, _ = load_demand(DEMAND_FILE)

    # =========================
    # INPUTS
    # =========================
    wind_power = df_wind["wind_power"].values
    wave_power = df_wave["wave_power"].values
    energy_demand = df_demand["demand"].values

    # =========================
    # ALGORITHMS
    # =========================
    algorithms = {
        "NR_ACO": run_nr_aco,
        "S_ACO": run_s_aco,
        "GA": run_ga,
    }

    results_all = {}

    # =========================================================
    # RUN ALL ALGORITHMS
    # =========================================================
    for name, algo in algorithms.items():

        print(f"\n=== Running {name} ===")

        start_time = time.time()

        all_history = []
        best_lcoe_runs = []
        best_config_runs = []

        for run in range(N_RUNS):

            print(f"{name} - Run {run+1}/{N_RUNS}")

            # ---- ACO ----
            if name in ["NR_ACO", "S_ACO"]:
                res = algo(
                    wind_power=wind_power,
                    wave_power=wave_power,
                    energy_demand=energy_demand,
                    num_iterations=num_iterations,
                    num_ants=population_size,
                    LPSP_target=LPSP_target,
                    random_seed=random_seed_base * run
                )

            # ---- GA ----
            else:
                res = algo(
                    wind_power=wind_power,
                    wave_power=wave_power,
                    energy_demand=energy_demand,
                    num_generations=num_iterations,
                    population_size=population_size,
                    LPSP_target=LPSP_target,
                    random_seed=random_seed_base * run
                )

            # Store per-run best
            best_lcoe_runs.append(res["LCOE"])
            best_config_runs.append(res["best_config"])

            # Store convergence
            hist = np.array(res["history_best"])
            all_history.append(hist[:, 3])  # LCOE

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTotal runtime: {elapsed_time:.2f} seconds")

        # =========================
        # POST PROCESS
        # =========================
        all_history = np.array(all_history)

        median_lcoe = np.median(all_history, axis=0)

        best_idx = np.argmin(best_lcoe_runs)

        results_all[name] = {
        "history_all": all_history,             
        "median_LCOE": median_lcoe,
        "best_LCOE_runs": np.array(best_lcoe_runs),
        "best_config": best_config_runs[best_idx],
        "best_LCOE": best_lcoe_runs[best_idx],
    }

    return results_all