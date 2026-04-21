import numpy as np
import pandas as pd
from hresopt.energy_system.energy_system import simulate_energy_system


def run_pso(
    wind_power=None,
    wave_power=None,
    geo_power=None,
    energy_demand=None,

    w=0.7,
    c1=1.5,
    c2=1.5,

    num_particles=50,
    num_iterations=100,

    LPSP_target=0.05,
    init_soc=0,

    wind_bounds=(0, 1000),
    wave_bounds=(0, 1000),
    geo_bounds=(0,17.6e3),      # in kW
    battery_bounds=(0, 1e7),    # in Wh
    step_battery=100,

    random_seed=None,
):

    if random_seed is not None:
        np.random.seed(random_seed)

    # =========================
    # SEARCH SPACE
    # =========================
    lower_bound = np.array([wind_bounds[0], wave_bounds[0], geo_bounds[0], battery_bounds[0]])
    upper_bound = np.array([wind_bounds[1], wave_bounds[1], geo_bounds[1], battery_bounds[1]])

    # =========================
    # INITIALIZATION
    # =========================
    positions = np.random.uniform(lower_bound, upper_bound, (num_particles, 4))
    velocities = np.zeros((num_particles, 4))

    personal_best_positions = positions.copy()
    personal_best_scores = np.ones(num_particles) * 1e10

    global_best_position = None
    global_best_score = 1e10

    history = []
    history_best = []

    # =========================
    # MAIN LOOP
    # =========================
    for iteration in range(num_iterations):

        for i in range(num_particles):

            x = positions[i]

            wind = int(round(x[0]))
            wave = int(round(x[1]))
            geo = int(round(x[2]))
            battery = int(np.ceil(x[3] / step_battery)) * step_battery

            # =========================
            # SYSTEM EVALUATION
            # =========================
            results = simulate_energy_system(
                wind_power=wind_power,
                wave_power=wave_power,
                geo_power=geo_power,
                energy_demand=energy_demand,
                num_wind=wind,
                num_wave=wave,
                geo_cap=geo,
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

            history.append((wind, wave, geo, battery, LCOE, LPSP, SOC))

            # =========================
            # PERSONAL BEST
            # =========================
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = np.array([wind, wave, geo, battery])

            # =========================
            # GLOBAL BEST
            # =========================
            if score < global_best_score:
                global_best_score = score
                global_best_position = np.array([wind, wave, geo, battery])

        history_best.append((
            int(global_best_position[0]),
            int(global_best_position[1]),
            int(global_best_position[2]),
            int(global_best_position[3]),
            global_best_score
        ))

        # =========================
        # VELOCITY & POSITION UPDATE
        # =========================
        for i in range(num_particles):

            r1 = np.random.rand(4)
            r2 = np.random.rand(4)

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best_positions[i] - positions[i])
                + c2 * r2 * (global_best_position - positions[i])
            )

            positions[i] += velocities[i]

            positions[i] = np.clip(positions[i], lower_bound, upper_bound)

    # =========================
    # FINAL OUTPUT
    # =========================
    best_config = (
        int(global_best_position[0]),
        int(global_best_position[1]),
        int(global_best_position[2]),
        int(global_best_position[3]),
    )

    results_best = simulate_energy_system(
        wind_power=wind_power,
        wave_power=wave_power,
        geo_power=geo_power,
        energy_demand=energy_demand,
        num_wind=best_config[0],
        num_wave=best_config[1],
        geo_cap=best_config[2],
        batt_cap=best_config[3],
        init_soc=init_soc,
        params=None
    )

    LCOE_best = results_best["LCOE"]
    LPSP_best = results_best["LPSP"]
    SOC_best = results_best["SOC_final"]

    df_history = pd.DataFrame(
        history,
        columns=["Wind", "Wave", "Geo", "Battery", "LCOE", "LPSP", "SOC"]
    )

    return {
        "best_config": best_config,
        "LCOE": LCOE_best,
        "LPSP": LPSP_best,
        "SOC": SOC_best,
        "history": df_history,
        "history_best": history_best
    }