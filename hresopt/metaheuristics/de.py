import numpy as np
import pandas as pd
from hresopt.energy_system.energy_system import simulate_energy_system


def run_de(
    wind_power,
    wave_power,
    energy_demand,

    F=0.8,
    CR=0.9,

    population_size=50,
    num_iterations=100,

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
    lower_bound = np.array([wind_bounds[0], wave_bounds[0], battery_bounds[0]])
    upper_bound = np.array([wind_bounds[1], wave_bounds[1], battery_bounds[1]])

    # =========================
    # INITIALIZATION
    # =========================
    population = np.random.uniform(lower_bound, upper_bound, (population_size, 3))
    scores = np.ones(population_size) * 1e10

    LCOEs = np.zeros(population_size)
    LPSPs = np.zeros(population_size)
    SOCs  = np.zeros(population_size)

    best_score = 1e10
    best_solution = None

    history = []
    history_best = []

    # =========================
    # INITIAL EVALUATION
    # =========================
    for i in range(population_size):

        x = population[i]

        wind = int(round(x[0]))
        wave = int(round(x[1]))
        battery = int(np.ceil(x[2] / step_battery)) * step_battery

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

        if LPSP > LPSP_target:
            score = LPSP * 1e10
        else:
            score = LCOE

        scores[i] = score
        LCOEs[i] = LCOE
        LPSPs[i] = LPSP
        SOCs[i]  = SOC
        history.append((wind, wave, battery, LCOE, LPSP, SOC))

        if score < best_score:
            best_score = score
            best_solution = [wind, wave, battery]

    history_best.append((*best_solution, best_score))

    # =========================
    # MAIN LOOP
    # =========================
    for iteration in range(num_iterations - 1):

        for i in range(population_size):

            idxs = [idx for idx in range(population_size) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)

            x_a, x_b, x_c = population[a], population[b], population[c]

            # =========================
            # MUTATION
            # =========================
            mutant = x_a + F * (x_b - x_c)
            mutant = np.clip(mutant, lower_bound, upper_bound)

            # =========================
            # CROSSOVER
            # =========================
            cross_points = np.random.rand(3) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, 3)] = True

            trial = np.where(cross_points, mutant, population[i])

            wind = int(round(trial[0]))
            wave = int(round(trial[1]))
            battery = int(np.ceil(trial[2] / step_battery)) * step_battery

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

            if LPSP > LPSP_target:
                score_trial = LPSP * 1e10
            else:
                score_trial = LCOE

            # =========================
            # SELECTION
            # =========================
            if score_trial < scores[i]:
                population[i] = trial
                scores[i] = score_trial

                LCOEs[i] = LCOE
                LPSPs[i] = LPSP
                SOCs[i]  = SOC

                history.append((wind, wave, battery, LCOE, LPSP, SOC))

            else:
                old = population[i]
                wind_old = int(round(old[0]))
                wave_old = int(round(old[1]))
                battery_old = int(np.ceil(old[2] / step_battery)) * step_battery

                history.append((
                    wind_old,
                    wave_old,
                    battery_old,
                    LCOEs[i],
                    LPSPs[i],
                    SOCs[i]
                ))

        # =========================
        # GLOBAL BEST UPDATE
        # =========================
        min_idx = np.argmin(scores)

        if scores[min_idx] < best_score:
            best_score = scores[min_idx]

            best_solution = [
                int(round(population[min_idx][0])),
                int(round(population[min_idx][1])),
                int(np.ceil(population[min_idx][2] / step_battery)) * step_battery
            ]

        history_best.append((*best_solution, best_score))

    # =========================
    # FINAL EVALUATION
    # =========================
    results_best = simulate_energy_system(
        wind_power=wind_power,
        wave_power=wave_power,
        energy_demand=energy_demand,
        num_wind=best_solution[0],
        num_wave=best_solution[1],
        batt_cap=best_solution[2],
        init_soc=init_soc,
        params=None
    )

    LCOE_best = results_best["LCOE"]
    LPSP_best = results_best["LPSP"]
    SOC_best = results_best["SOC_final"]

    df_history = pd.DataFrame(
        history,
        columns=["Wind", "Wave", "Battery", "LCOE", "LPSP", "SOC"]
    )

    return {
        "best_config": tuple(best_solution),
        "LCOE": LCOE_best,
        "LPSP": LPSP_best,
        "SOC": SOC_best,
        "history": df_history,
        "history_best": history_best
    }  