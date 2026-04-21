from dataclasses import dataclass

@dataclass
class PhysicalParams:
    # Wind Profile for Height Correction
    shear_exponent: float = 0.5 #1/7