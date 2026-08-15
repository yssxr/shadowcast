"""Validation against ground truth: the part that makes this more than a visualisation."""

from shadowcast.validate.fog_oracle import FogAgreement, validate_fog

__all__ = ["FogAgreement", "validate_fog"]
