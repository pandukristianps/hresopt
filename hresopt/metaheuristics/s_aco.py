import numpy as np
import pandas as pd
from hresopt.energy_system.energy_system import simulate_energy_system


def run_s_aco(
    wind_power,
    wave_power,
    energy_demand,

    alpha=0.5,
    evaporation_rate=0.4,
    Q=0.15,

    num_ants=20,
    num_iterations=50,

    LPSP_target=0.05,
    init_soc=0,

    wind_bounds=(0, 1000),
    wave_bounds=(0, 1000),
    battery_bounds=(0, 1e7),
    step_battery=100,

    random_seed=None,
):

    if random_seed is not None:
        np.random.seed(random_seed)

    # =========================
    # SEARCH SPACE
    # =========================
    wind_range = np.arange(wind_bounds[0], wind_bounds[1] + 1, 1)
    wave_range = np.arange(wave_bounds[0], wave_bounds[1] + 1, 1)
    battery_range = np.arange(battery_bounds[0], battery_bounds[1] + 1, step_battery)

    # =========================
    # PHEROMONES
    # =========================
    pheromone_wind = np.ones(len(wind_range))
    pheromone_wave = np.ones(len(wave_range))
    pheromone_batt = np.ones(len(battery_range))

    best_score = 1e10
    best_solution = None

    history = []
    history_best = []

    # =========================
    # MAIN LOOP
    # =========================
    for iteration in range(num_iterations):

        scores = []
        solutions = []

        # probabilities (safe)
        p_wind = np.maximum(pheromone_wind, 1e-10) ** alpha
        p_wind /= np.sum(p_wind) + 1e-10

        p_wave = np.maximum(pheromone_wave, 1e-10) ** alpha
        p_wave /= np.sum(p_wave) + 1e-10

        p_batt = np.maximum(pheromone_batt, 1e-10) ** alpha
        p_batt /= np.sum(p_batt) + 1e-10

        for ant in range(num_ants):

            wind_idx = np.random.choice(len(wind_range), p=p_wind)
            wave_idx = np.random.choice(len(wave_range), p=p_wave)
            batt_idx = np.random.choice(len(battery_range), p=p_batt)

            wind = wind_range[wind_idx]
            wave = wave_range[wave_idx]
            battery = battery_range[batt_idx]

            # =========================
            # SYSTEM EVALUATION
            # =========================
            results = simulate_energy_system(
                wind_power=wind_power,
                wave_power=wave_power,
                energy_demand=energy_demand,
                num_wind=wind,
                num_wave=wave,
                batt_cap=battery,
                init_soc=init_soc,
                params=None
            )

            LCOE = results["LCOE"]
            LPSP = results["LPSP"]
            SOC = results["SOC_final"]

            # penalty
            if LPSP > LPSP_target:
                score = LPSP * 1e10
            else:
                score = LCOE

            scores.append(score)
            solutions.append((wind_idx, wave_idx, batt_idx, LCOE, LPSP, SOC))

            history.append((wind, wave, battery, LCOE, LPSP, SOC))

        # =========================
        # EVAPORATION
        # =========================
        pheromone_wind *= (1 - evaporation_rate)
        pheromone_wave *= (1 - evaporation_rate)
        pheromone_batt *= (1 - evaporation_rate)

        # =========================
        # PHEROMONE UPDATE 
        # =========================
        for (wind_idx, wave_idx, batt_idx, LCOE, LPSP, SOC) in solutions:

            if LPSP <= LPSP_target:
                deposit = Q / (LCOE + 1e-10)

                pheromone_wind[wind_idx] += deposit
                pheromone_wave[wave_idx] += deposit
                pheromone_batt[batt_idx] += deposit

        # =========================
        # GLOBAL BEST
        # =========================
        best_idx = np.argmin(scores)
        best_iter = solutions[best_idx]

        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_solution = best_iter

        best_wind = wind_range[best_solution[0]]
        best_wave = wave_range[best_solution[1]]
        best_batt = battery_range[best_solution[2]]

        history_best.append((best_wind, best_wave, best_batt, best_solution[3]))

    # =========================
    # OUTPUT
    # =========================
    best_config = (
        wind_range[best_solution[0]],
        wave_range[best_solution[1]],
        battery_range[best_solution[2]],
    )

    df_history = pd.DataFrame(
        history,
        columns=["Wind", "Wave", "Battery", "LCOE", "LPSP", "SOC"]
    )

    return {
        "best_config": best_config,
        "LCOE": best_solution[3],
        "LPSP": best_solution[4],
        "SOC": best_solution[5],
        "history": df_history,
        "history_best": history_best
    }