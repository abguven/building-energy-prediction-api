from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class EnergyInput(BaseModel):    
    # Features sous forme de texte
    LargestPropertyUseType: str
    ListOfAllPropertyUseTypes: str = Field(..., description="Liste des usages séparés par une virgule, ex: 'Office, Parking'")

    # Features Numériques
    YearBuilt: int = Field(..., ge=1900, le=2025)
    NumberofFloors: int = Field(..., ge=1)
    PropertyGFAParking: int = Field(..., ge=0)
    PropertyGFABuilding_s_: int = Field(..., ge=1)
    ENERGYSTARScore: Optional[float] = Field(default=None, ge=1, le=100)

    # Validateurs génériques
    @field_validator('PropertyGFABuilding_s_', 'PropertyGFAParking', 'NumberofFloors')
    @classmethod
    def check_strictly_positive(cls, v, info):
        """Vérifie que ces champs sont strictement positifs (sauf parking)."""
        if info.field_name == 'PropertyGFAParking':
            if v < 0:
                raise ValueError(f"La valeur pour '{info.field_name}' doit être positive ou nulle.")
        elif v <= 0:
            raise ValueError(f"La valeur pour '{info.field_name}' doit être strictement positive.")
        return v
    
    @field_validator("LargestPropertyUseType", "ListOfAllPropertyUseTypes")
    @classmethod
    def check_not_empty_and_strip(cls, v, info):
        """Vérifie que les champs texte ne sont pas vides et normalise."""
        if not v or not v.strip():
            raise ValueError(f"'{info.field_name}' ne peut pas être vide.")
        # Normalise en Title Case
        return v.strip()