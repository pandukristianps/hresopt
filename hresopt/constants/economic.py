from dataclasses import dataclass

@dataclass
class EconomicParams:
    # Capital Expenditures ($/kW or $/kWh)
    capex_wind_per_kW: float = 4973
    capex_wave_per_kW: float = 7404
    capex_geo_per_kW: float = 5115
    capex_battery_per_kWh: float = 362

    # Operational Expenditures ($/kW/year or $/kWh/year)
    opex_wind_per_kW: float = 86
    opex_wave_per_kW: float = 430
    opex_geo_per_kW: float = 110
    opex_battery_per_kWh: float = 8

    # Fixed Charge Rates
    fcr_wind: float = 0.048
    fcr_wave: float = 0.108
    fcr_geo: float = 0.061
    fcr_battery: float = 0.096