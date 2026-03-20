import numpy as np
import pandas as pd

from pycaret.classification import *

df = pd.read_csv("income-dataset.csv")

df.head()

df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

df['income'] = df['income'].str.replace('.', '', regex=False)
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

df_encoded = pd.get_dummies(df, drop_first=True)
corr = df_encoded.corr()[['income']].sort_values(by='income', ascending=False)
print(corr.head(15))

clf = setup(data=df, target='income', session_id=42, normalize=True, verbose=False)

best_model = compare_models()

final_model = finalize_model(best_model)

save_model(final_model, "best_model_automl")

