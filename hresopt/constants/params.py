from dataclasses import dataclass, field
from .economic import EconomicParams
from .equipment import EquipmentParams
from .physical import PhysicalParams

@dataclass
class SystemParams:
    economic: EconomicParams = field(default_factory=EconomicParams)
    equipment: EquipmentParams = field(default_factory=EquipmentParams)
    physical: PhysicalParams = field(default_factory=PhysicalParams)