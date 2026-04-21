import numpy as np
import pandas as pd
from hresopt.energy_system.energy_system import simulate_energy_system


def run_ga(
    wind_power,
    wave_power,
    energy_demand,

    population_size=30,
    num_generations=50,

    crossover_rate=0.8,
    mutation_rate=0.3,
    tournament_size=2,

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

    wind_idx_range = np.arange(len(wind_range))
    wave_idx_range = np.arange(len(wave_range))
    batt_idx_range = np.arange(len(battery_range))

    # =========================
    # EVALUATION
    # =========================
    def evaluate(individual):

        w_idx, wa_idx, b_idx = individual

        wind = wind_range[w_idx]
        wave = wave_range[wa_idx]
        battery = battery_range[b_idx]

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

        return score, LCOE, LPSP, SOC

    # =========================
    # GA OPERATORS
    # =========================
    def tournament_selection(pop, scores):
        participants = np.random.choice(len(pop), tournament_size, replace=False)
        best = participants[np.argmin(scores[participants])]
        return pop[best]

    def crossover(parent1, parent2):
        if np.random.rand() < crossover_rate:
            point = np.random.randint(1, 3)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def mutate(ind):
        if np.random.rand() < mutation_rate:
            gene = np.random.randint(0, 3)

            if gene == 0:
                ind[0] = np.random.choice(wind_idx_range)
            elif gene == 1:
                ind[1] = np.random.choice(wave_idx_range)
            else:
                ind[2] = np.random.choice(batt_idx_range)

        return ind

    # =========================
    # INITIAL POPULATION
    # =========================
    population = np.column_stack([
        np.random.choice(wind_idx_range, population_size),
        np.random.choice(wave_idx_range, population_size),
        np.random.choice(batt_idx_range, population_size)
    ])

    best_score = 1e10
    best_solution = None

    history = []
    history_best = []

    # =========================
    # GA LOOP
    # =========================
    for generation in range(num_generations):

        scores = np.zeros(population_size)
        detailed = []

        # Evaluate
        for i, ind in enumerate(population):

            score, LCOE, LPSP, SOC = evaluate(ind)

            scores[i] = score
            detailed.append((ind[0], ind[1], ind[2], LCOE, LPSP, SOC))

            history.append((
                wind_range[ind[0]],
                wave_range[ind[1]],
                battery_range[ind[2]],
                LCOE, LPSP, SOC
            ))

        # Best of generation
        gen_best_idx = np.argmin(scores)
        gen_best = detailed[gen_best_idx]

        if scores[gen_best_idx] < best_score:
            best_score = scores[gen_best_idx]
            best_solution = gen_best

        best_wind = wind_range[best_solution[0]]
        best_wave = wave_range[best_solution[1]]
        best_batt = battery_range[best_solution[2]]

        history_best.append((best_wind, best_wave, best_batt, best_solution[3]))

        # =========================
        # NEW POPULATION
        # =========================
        new_population = []

        while len(new_population) < population_size:

            parent1 = tournament_selection(population, scores)
            parent2 = tournament_selection(population, scores)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.append(child1)

            if len(new_population) < population_size:
                new_population.append(child2)

        population = np.array(new_population)

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