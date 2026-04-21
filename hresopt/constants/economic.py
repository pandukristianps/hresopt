from dataclasses import dataclass

@dataclass
class EconomicParams:
    # Capital Expenditures ($/kW or $/kWh)
    capex_wind_per_kW: float = 5460
    capex_wave_per_kW: float = 7000
    capex_battery_per_kWh: float = 200

    # Operational Expenditures ($/kW/year or $/kWh/year)
    opex_wind_per_kW: float = 95
    opex_wave_per_kW: float = 150
    opex_battery_per_kWh: float = 10

    # Fixed Charge Rates
    fcr_wind: float = 0.051
    fcr_wave: float = 0.108
    fcr_battery: float = 0.096