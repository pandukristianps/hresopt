import numpy as np
from hresopt.constants import SystemParams


# =========================
# COST MODEL
# =========================
def compute_costs(params: SystemParams = None):
    
    if params is None:
        params = SystemParams()

    econ = params.economic
    equip = params.equipment

    capex_wind = econ.capex_wind_per_kW * equip.P_rated_wind * econ.fcr_wind
    opex_wind = econ.opex_wind_per_kW * equip.P_rated_wind
    cost_wind = capex_wind + opex_wind

    capex_wave = econ.capex_wave_per_kW * equip.P_rated_wave * econ.fcr_wave
    opex_wave = econ.opex_wave_per_kW * equip.P_rated_wave
    cost_wave = capex_wave + opex_wave

    capex_geo = econ.capex_geo_per_kW * econ.fcr_geo
    opex_geo = econ.opex_geo_per_kW 
    cost_geo = capex_geo + opex_geo

    capex_batt = econ.capex_battery_per_kWh * econ.fcr_battery
    opex_batt = econ.opex_battery_per_kWh
    cost_batt = capex_batt + opex_batt

    return cost_wind, cost_wave, cost_geo, cost_batt


# =========================
# ENERGY SYSTEM SIMULATION
# =========================
def simulate_energy_system(
    wind_power=None,
    wave_power=None,
    geo_power=None,
    energy_demand=None,
    num_wind=0,
    num_wave=0,
    geo_cap=0,
    batt_cap=0,
    params: SystemParams = None,
    init_soc=0,
):
    
    if params is None:
        params = SystemParams()

    T = len(energy_demand)

    if wind_power is None:
        wind_power = np.zeros(T)
        num_wind = 0

    if wave_power is None:
        wave_power = np.zeros(T)
        num_wave = 0

    if geo_power is None:
        geo_power = np.zeros(T)
        geo_cap = 0

    if energy_demand is None:
        raise ValueError("energy_demand cannot be None")

    equip = params.equipment

    eta_charge = equip.eta_charge
    eta_discharge = equip.eta_discharge
    eta_generation = equip.eta_generation

    SOC_min = equip.SOC_min
    SOC_max = equip.SOC_max

    cost_wind, cost_wave, cost_geo, cost_battery = compute_costs(params)

    energy_generated = (wind_power * num_wind + wave_power * num_wave + geo_power) * eta_generation

    energy_stored = np.zeros(T + 1)
    energy_met = np.zeros(T)
    energy_unmet = np.zeros(T)
    SOC = np.zeros(T)

    stored = init_soc * batt_cap
    energy_stored[0] = stored
    
    E_min = SOC_min * batt_cap
    E_max = SOC_max * batt_cap


    for t in range(T):

        surplus = energy_generated[t] - energy_demand[t]

        if surplus >= 0:
            stored += surplus * eta_charge
            stored = min(stored, E_max)
            energy_met[t] = energy_demand[t]

        else:
            discharge = min(-surplus / eta_discharge, stored - E_min)
            stored -= discharge

            supplied = energy_generated[t] + discharge * eta_discharge
            energy_met[t] = supplied
            energy_unmet[t] = energy_demand[t] - supplied

        energy_stored[t + 1] = stored
        SOC[t] = 0 if batt_cap == 0 else stored / batt_cap

    # =========================
    # METRICS
    # =========================
    total_cost = num_wind * cost_wind + num_wave * cost_wave + geo_cap * cost_geo + batt_cap * cost_battery
    total_energy_met = np.sum(energy_met)
    total_energy_generated = np.sum(energy_generated)

    LCOE = total_cost / (total_energy_met + 1e-10)
    LPSP = np.sum(energy_unmet) / np.sum(energy_demand)
    SOC_final = SOC[-1]

    installed_capacity = float(num_wind * equip.P_rated_wind + num_wave * equip.P_rated_wave + geo_cap)
    capacity_ratio = total_energy_generated/(installed_capacity * T + 1e-10)
    effective_capacity_factor = total_energy_met / (installed_capacity * T + 1e-10)

    curtailment = total_energy_generated - total_energy_met
    curtailment_ratio = curtailment / (total_energy_generated + 1e-10)

    energy_met_ratio = energy_met / energy_demand

    return {
        "LCOE": LCOE,
        "LPSP": LPSP,
        "SOC_final": SOC_final,
        "energy_met": total_energy_met,
        "capacity_ratio" : capacity_ratio,
        "effective_capacity_factor" : effective_capacity_factor,
        "curtailment_ratio" : curtailment_ratio
    }