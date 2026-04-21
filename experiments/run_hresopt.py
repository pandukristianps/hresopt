from run_optimization import run_optimization
from result_optimization import analyze_results


# =========================================================
# CONFIG
# =========================================================
WIND_FILE = "data/wind.nc"
WAVE_FILE = "data/wave.nc"
DEMAND_FILE = "data/demand_elhierro.csv"

CURVE_FILE = "data/wind_curve.csv"
MATRIX_FILE = "data/wave_matrix.csv"

LAT = 28
LON = -18
Z_HUB = 136

NUM_RUNS = 10
NUM_ITERATIONS = 35
POPULATION_SIZE = 2

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":

    results = run_optimization(
        WIND_FILE=WIND_FILE,
        WAVE_FILE=WAVE_FILE,
        DEMAND_FILE=DEMAND_FILE,
        CURVE_FILE=CURVE_FILE,
        MATRIX_FILE=MATRIX_FILE,
        LAT=LAT,
        LON=LON,
        Z_HUB=Z_HUB,

        N_RUNS=NUM_RUNS,
        num_iterations=NUM_ITERATIONS,
        population_size=POPULATION_SIZE,
        LPSP_target = 0.05,
    )

    stats = analyze_results(results, num_iterations=NUM_ITERATIONS)