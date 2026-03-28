import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2

df = pd.read_csv("/content/income-dataset.csv")

df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

df['income'] = df['income'].str.replace('.', '', regex=False)
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

X = df.drop('income', axis=1)
y = df['income']

X_encoded = pd.get_dummies(X, drop_first=True)

corr_matrix = X_encoded.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
X_encoded = X_encoded.drop(columns=to_drop)

selector = SelectKBest(score_func=chi2, k=15)
X_selected = selector.fit_transform(X_encoded, y)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

with open("model_fg.pkl", "wb") as f:
    pickle.dump((model, scaler, selector), f)
