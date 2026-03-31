from __future__ import annotations

import bentoml
from validation import EnergyInput

with bentoml.importing():
    import pandas as pd    

# Replaces the legacy bentofile.yml — defines the Docker image inline
runtime_image = (
            bentoml.images.Image(
                base_image="python:3.11-slim"    
            )
            .python_packages(
                "pandas>=2.3.3,<3.0.0",
                "scikit-learn>=1.7.2,<2.0.0",
                "pydantic>=2.12.3,<3.0.0",
            )
)

@bentoml.service(
    image=runtime_image,
    resources={"cpu": "2"},
    traffic={"timeout": 30}
)
class SeattleEnergyPredictor:  # Nom cohérent avec ton modèle
    model_ref = bentoml.models.BentoModel("seattle_energy_regressor:latest")    
    
    def __init__(self):

        self.model = bentoml.sklearn.load_model(self.model_ref)
        self.cat_cols = self.model_ref.custom_objects.get("categorical_features", [])
        # Median value used to impute missing ENERGYSTARScore at inference time
        self.median_energy_star_score = self.model_ref.custom_objects.get("median_energy_star_score")
    
    @bentoml.api
    def predict(self, data: EnergyInput) -> dict:

        warning_messages = []
        input_features = self.model_ref.custom_objects.get("input_features")
        
        # --- On-the-fly feature engineering ---
        input_data = data.model_dump()
                
        # Calcul de number_of_uses
        input_data['number_of_uses'] = len(input_data.get('ListOfAllPropertyUseTypes', '').split(','))

        # Calcul de has_parking
        input_data["has_parking"] = input_data.get("PropertyGFAParking", 0) > 0       
        
        # --- ENERGYSTARScore imputation ---
        if input_data['ENERGYSTARScore'] is None:
            input_data['ENERGYSTARScore'] = self.median_energy_star_score
            input_data['has_energy_score'] = 0
            warning_messages.append("ENERGYSTARScore not provided, replaced with median. Prediction may be less accurate.")
        else:
            input_data['has_energy_score'] = 1        
        
        # Drop raw column not used as a model feature
        del input_data['ListOfAllPropertyUseTypes']
        
        # Prédiction
        df = pd.DataFrame([input_data])
        df = df.rename(columns={'PropertyGFABuilding_s_': 'PropertyGFABuilding(s)'})        
        df[self.cat_cols] = df[self.cat_cols].astype(str)

        # Ensure the correct feature set is selected
        if input_features:
            df = df[input_features]        
    
        pred = self.model.predict(df)
        value = round(float(pred[0]), 2)

        response = {
            "predicted_SiteEnergyUse(kBtu)": value,
            "formatted": f"{value:_.2f}"
        }

        if warning_messages: 
            for w in warning_messages:  
                print(f"[USER_WARNING] {w}")
            response["warning_messages"] = warning_messages

        return response
    
    @bentoml.api
    def health(self) -> dict:
        return {"status": "ok"}