# Adult Income Prediction using Machine Learning

## 📌 Overview

This project predicts whether a person earns more than 50K per year using a Logistic Regression model.

It includes:

* Data preprocessing
* Model training and evaluation
* Feature correlation analysis
* SHAP-based explainability
* Prediction using JSON input file

---

## 📁 Files

* `model.py` → trains the model and performs EDA
* `predict.py` → loads model and predicts using JSON input
* `model.pkl` → saved trained model
* `income-dataset.csv` → dataset
* `requirements.txt` → dependencies

---

## ⚙️ Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Step 1: Train the model

```bash
python model.py
```

### Step 2: Run prediction

```bash
python predict.py
```

Enter the JSON file path when prompted.

---

## 📄 Input Format (JSON)

Example:

```json
{
  "age": 37,
  "workclass": "Private",
  "fnlwgt": 284582,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital-gain": 0,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}
```

---

## 📊 Model Performance

* Accuracy: ~84%
* Evaluation metrics printed in `model.py`

---

## 🔍 Explainability

SHAP (SHapley Additive exPlanations) is used to explain the contribution of each feature to the prediction.

---

## 🧠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SHAP

---

## 📎 Author

Your Name
