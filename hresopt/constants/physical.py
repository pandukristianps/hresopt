from dataclasses import dataclass

@dataclass
class PhysicalParams:
    # Wind Profile for Height Correction
    shear_exponent: float = 1/7
