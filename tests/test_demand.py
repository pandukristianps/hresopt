from hresopt.data_loader.demand_loader import load_demand


# =========================================================
# CONFIG
# =========================================================
DEMAND_FILE = "data/demand_elhierro.csv"


# =========================================================
# TEST FUNCTION
# =========================================================
def test_demand():

    print("Loading demand data...")
    df, meta = load_demand(DEMAND_FILE)

    print("\n--- RAW DEMAND DATA ---")
    print(df.head())
    print(meta)

    print("\n--- BASIC CHECKS ---")

    # Check columns
    assert "time" in df.columns, "Missing time column"
    assert "demand" in df.columns, "Missing demand column"

    # Check non-negative demand
    assert (df["demand"] >= 0).all(), "Negative demand detected"

    # Check time is sorted
    assert df["time"].is_monotonic_increasing, "Time is not sorted"

    print("Basic checks passed ✅")

    print("\n--- STATISTICS ---")

    # Basic statistics
    mean_demand = df["demand"].mean()
    max_demand = df["demand"].max()
    min_demand = df["demand"].min()
    std_demand = df["demand"].std()

    # Energy (assuming hourly data → kWh)
    total_energy = df["demand"].sum()

    print(f"Mean demand: {mean_demand:.2f} kW")
    print(f"Max demand: {max_demand:.2f} kW")
    print(f"Min demand: {min_demand:.2f} kW")
    print(f"Std deviation: {std_demand:.2f} kW")
    print(f"Total energy: {total_energy:.2f} kWh")

    print("\nAll checks passed ✅")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    test_demand()