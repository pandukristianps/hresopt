from dataclasses import dataclass

@dataclass
class EquipmentParams:
    # Rated Power (kW)
    P_rated_wind: float = 12000
    P_rated_wave: float = 400

    # System Efficiencies
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    eta_generation: float = 0.95

    #SOC Limits
    SOC_min: float = 0.2
    SOC_max: float = 0.8