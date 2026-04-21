import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os


# =========================================================
# STATISTICS
# =========================================================
def compute_statistics(results_all):

    stats = {}

    for name, res in results_all.items():

        lcoe_runs = res["best_LCOE_runs"]

        stats[name] = {
            "min": np.min(lcoe_runs),
            "mean": np.mean(lcoe_runs),
            "median": np.median(lcoe_runs),
            "std": np.std(lcoe_runs),
            "best_config": res["best_config"],
        }

    return stats


# =========================================================
# SAVE TEXT REPORT
# =========================================================
def save_results(stats, filename="1. optimization_results.txt"):

    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)

    with open(filepath, "w") as f:

        for name, s in stats.items():

            f.write(f"=== {name.upper()} ===\n")
            f.write(f"Min LCOE: {s['min']}\n")
            f.write(f"Mean LCOE: {s['mean']}\n")
            f.write(f"Median LCOE: {s['median']}\n")
            f.write(f"Std LCOE: {s['std']}\n")
            f.write(f"Best Config: {s['best_config']}\n\n")


# =========================================================
# BOXPLOT
# =========================================================
def plot_boxplot(results_all, show_legend=True):

    data = []
    labels = []

    for name, res in results_all.items():
        data.append(res["best_LCOE_runs"])
        labels.append(name.upper())

    plt.figure(figsize=(5, 6))
    
    plt.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanprops={
            "marker": "D",  
            "markerfacecolor": "blue",
            "markeredgecolor": "blue",
            "markersize": 4
        },
        medianprops={
            "color": "red",
            "linewidth": 2
        }
    )

    if show_legend:
    
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Median'),
            Line2D([0], [0], marker='D', color='w',
                   markerfacecolor='blue', markeredgecolor='blue',
                   markersize=4, label='Mean')
        ]

    plt.legend(handles=legend_elements)

    plt.ylabel("Best LCOE ($/kWh)")
    plt.title("Best LCOE Distribution")
    
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/2. boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# CONVERGENCE METRIC
# =========================================================
def compute_convergence(results_all, threshold=0.01):
    """
    Returns:
    - C_all: dict of convergence curves
    - t_conv_all: dict of convergence iterations
    """

    C_all = {}
    t_conv_all = {}

    for name, res in results_all.items():

        history = res["history_all"]

        # ---- Median LCOE ----
        median_curve = np.median(history, axis=0)

        # ---- Final best ----
        best_final = np.min(median_curve)

        # ---- Convergence curve ----
        C = (median_curve - best_final) / (best_final + 1e-10)

        C_all[name] = C

        # ---- Convergence iteration ----
        idx = np.where(C <= threshold)[0]

        t_conv_all[name] = int(idx[0]) if len(idx) > 0 else None

    return C_all, t_conv_all


def print_convergence_iteration(t_conv_all, threshold=0.01):

    print(f"\n=== CONVERGENCE ITERATION (C <= {threshold*100:.1f}%) ===")

    for name, t in t_conv_all.items():
        print(f"{name.upper()}: {t}")


# =========================================================
# MEDIAN LCOE PLOT
# =========================================================
def plot_convergence(results_all, num_iterations, threshold=0.01):

    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 6))

    # Compute convergence iteration
    _, t_conv_all = compute_convergence(results_all, threshold)

    for name, res in results_all.items():

        history = res["history_all"]
        median_curve = np.median(history, axis=0)

        x = np.arange(1, len(median_curve) + 1)
        line, = plt.plot(x, median_curve, label=name.upper())

    ymin, ymax = plt.gca().get_ylim()

    for name, res in results_all.items():

        history = res["history_all"]
        median_curve = np.median(history, axis=0)

        t_conv = t_conv_all[name]

        if t_conv is not None:

            x_conv = t_conv + 1
            lcoe_conv = median_curve[t_conv]

            line = plt.gca().lines[list(results_all.keys()).index(name)]
            color = line.get_color()

            plt.vlines(
                x=x_conv,
                ymin=ymin,
                ymax=lcoe_conv,
                linestyle="--",
                color=color,
                linewidth=1
            )

    plt.plot([], [], linestyle="--", color="black", label="1% Threshold")

    plt.xlabel("Iteration")
    plt.ylabel("Median LCOE ($/kWh)")
    plt.xlim(0.8, num_iterations)
    plt.ylim(ymin, ymax)
    plt.title("Convergence Plot")
    plt.legend(frameon=False)

    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", "3_convergence_plot.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# MASTER FUNCTION
# =========================================================
def analyze_results(results_all, num_iterations):

    stats = compute_statistics(results_all)

    save_results(stats)

    plot_boxplot(results_all)
    plot_convergence(results_all, num_iterations, threshold=0.01)

    _,t_conv = compute_convergence(results_all, threshold=0.01)
    print_convergence_iteration(t_conv, threshold=0.01)

    return stats