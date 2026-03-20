import pandas as pd
import json
from pycaret.classification import load_model, predict_model

model = load_model("best_model_automl")

file_path = input().strip()

with open(file_path) as f:
    input_data = json.load(f)

input_df = pd.DataFrame([input_data])

result = predict_model(model, data=input_df)

print(result[['prediction_label', 'prediction_score']])
