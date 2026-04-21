import pandas as pd


def load_demand(path):
    """
    Load demand data from CSV.

    Expected columns:
    - time
    - demand (kW)

    Returns
    -------
    df : pd.DataFrame
        Columns: [time, demand]
    meta : dict
    """

    df = pd.read_csv(path)

    required_cols = ["time", "demand"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"CSV must contain columns {required_cols}. "
                f"Found: {list(df.columns)}"
            )

    # Convert time column
    df["time"] = pd.to_datetime(df["time"], format="%d-%m-%y %H:%M")

    # Sort by time (important!)
    df = df.sort_values("time").reset_index(drop=True)

    meta = {
        "source": "CSV"
    }

    return df, meta