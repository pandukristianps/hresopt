import pandas as pd
import matplotlib.pyplot as plt
from hresopt.energy_system.run_energy_system import run_energy_system

df, results = run_energy_system()

df["time"] = pd.to_datetime(df["time"])
df_sep = df[df["time"].dt.month == 9]

fig, ax1 = plt.subplots()

# Left axis (energy)
#ax1.plot(df_sep["time"], df_sep["demand"], label="Demand")
ax1.plot(df_sep["time"], df_sep["energy_met_ratio"]*100, label="Energy Met Ratio")
ax1.plot(df_sep["time"], df_sep["SOC"]*100, linestyle="--", label="SOC")
ax1.set_ylabel("%")

'''
# Right axis (SOC)
ax2 = ax1.twinx()
ax2.plot(df_sep["time"], df_sep["SOC"]*100, linestyle="--", label="SOC")
ax2.set_ylabel("SOC (%)")
'''
# Legends
ax1.legend(loc="lower left")
#ax2.legend(loc="lower right")

plt.title("System Operation - September")
plt.xlabel("Time")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()