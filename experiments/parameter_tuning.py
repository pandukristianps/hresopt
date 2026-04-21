import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# =========================================================
# GENERIC TUNER
# =========================================================
def tune_parameter(
    algo_func,
    param_name,
    base_params,
    param_start,
    step=0.1,
    n_datasets=4,
    n_runs=5,
):
    """
    algo_func   : run_de / run_pso / run_nr_aco
    param_name  : string (e.g. 'F', 'CR', 'w', etc.)
    base_params : dict of all required inputs for the algorithm
    """

    results_all = {}
    medians = []
    param_values = []

    direction = +1
    current_value = param_start

    print(f"\nTuning parameter: {param_name}")

    while len(results_all) < n_datasets:

        print(f"\nTesting {param_name} = {current_value:.3f}")

        run_results = []

        for run in range(n_runs):

            params = base_params.copy()
            params[param_name] = current_value
            params["random_seed"] = run

            res = algo_func(**params)

            run_results.append(res["LCOE"])

        median_val = np.median(run_results)

        key = f"{param_name}={current_value:.2f}"
        results_all[key] = run_results

        medians.append(median_val)
        param_values.append(current_value)

        print(f"Median LCOE: {median_val}")

        # =====================================================
        # DIRECTION CHECK (after 2 points)
        # =====================================================
        if len(medians) == 2:
            if medians[1] > medians[0]:
                direction = -1
                current_value = param_start + direction * step
                print("Reversing direction")
                continue

        # =====================================================
        # MOVE PARAMETER
        # =====================================================
        current_value += direction * step

    return results_all, param_values, medians


# =========================================================
# PLOT FUNCTION
# =========================================================
def plot_tuning(results_all, title, filename):

    data = []
    labels = []

    for name, values in results_all.items():
        data.append(values)
        labels.append(name)

    plt.figure(figsize=(6, 5))

    plt.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanline=False,
        meanprops={
            "marker": "D",
            "markerfacecolor": "blue",
            "markeredgecolor": "blue"
        },
        medianprops={
            "color": "red",
            "linewidth": 2
        }
    )

    plt.ylabel("LCOE")
    plt.title(title)

    os.makedirs("output", exist_ok=True)
    plt.savefig(f"output/{filename}", dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# EXAMPLE: DE TUNING
# =========================================================
def tune_de_F(run_de, base_params):

    results, values, medians = tune_parameter(
        algo_func=run_de,
        param_name="F",
        base_params=base_params,
        param_start=0.5,
        step=0.1,
        n_datasets=4,
        n_runs=5,
    )

    plot_tuning(results, "DE Parameter Tuning (F)", "de_F_tuning.png")

    return results, values, medians


def tune_de_CR(run_de, base_params):

    results, values, medians = tune_parameter(
        algo_func=run_de,
        param_name="CR",
        base_params=base_params,
        param_start=0.5,
        step=0.1,
        n_datasets=4,
        n_runs=5,
    )

    plot_tuning(results, "DE Parameter Tuning (CR)", "de_CR_tuning.png")

    return results, values, medians