# Adult Income Prediction using Machine Learning

## 📌 Overview

This project predicts whether a person earns more than $50K per year using Machine Learning.

The project includes:

* Manual model (Logistic Regression)
* AutoML using PyCaret (multiple models comparison)
* Data preprocessing and cleaning
* Feature correlation analysis
* SHAP-based explainability
* Prediction using JSON input file

---

## 📁 Project Structure

* `model.py` → manual model training (Logistic Regression + SHAP)
* `predict.py` → prediction with SHAP explanation
* `model_automl.py` → AutoML using PyCaret
* `predict_automl.py` → prediction using AutoML model
* `income-dataset.csv` → dataset
* `requirements.txt` → dependencies

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Manual Model (with SHAP)

```bash
python model.py
python predict.py
```

---

### 🔹 AutoML Model 

```bash
python model_automl.py
python predict_automl.py
```

⚠️ Run `model_automl.py` first to generate the trained model.

---

## 📄 Input Format (JSON)

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

* Logistic Regression Accuracy: ~84%
* AutoML selects best model automatically (Random Forest, XGBoost, etc.)

---

## 🔍 Feature Analysis

* Correlation analysis performed using encoded dataset
* Helps identify important features

---

## 🧠 Explainability using SHAP

SHAP (SHapley Additive exPlanations) is used to explain model predictions.

* Shows contribution of each feature to the prediction
* Helps understand model behavior
* Provides both magnitude and direction of impact

### Key Insight:

> Features like education, marital-status, and age have strong influence on income prediction.

⚠️ Note:

* SHAP is applied to the **manual Logistic Regression model**
* For AutoML models, SHAP is not directly used due to model compatibility differences

---

## ⚙️ Approach and Implementation

### 📌 Approach

1. Data cleaning (handling missing values, removing spaces)
2. Feature preprocessing (encoding + scaling)
3. Model training (Logistic Regression)
4. Model comparison using AutoML (PyCaret)
5. Model evaluation
6. Explainability using SHAP
7. Prediction using JSON input

---

### 🛠️ Implementation

* Used Pipeline for preprocessing + model
* Used OneHotEncoding for categorical features
* Used StandardScaler for numerical features
* Used PyCaret for AutoML
* Used Pickle for saving models
* Built prediction system using JSON input

---

## 🤖 AutoML using PyCaret

* Trains multiple models automatically
* Compares performance
* Selects best model
* Improves accuracy and reduces manual effort

---

## 📘 Feature Description

### 🔢 Numerical Features

| Feature        | Description           | Range   |
| -------------- | --------------------- | ------- |
| age            | Age of the individual | 17 – 90 |
| fnlwgt         | Census weight         | > 0     |
| education-num  | Years of education    | 1 – 16  |
| capital-gain   | Capital gain          | ≥ 0     |
| capital-loss   | Capital loss          | ≥ 0     |
| hours-per-week | Working hours         | 1 – 99  |

---

### 🔤 Categorical Features

| Feature        | Description     | Values                            |
| -------------- | --------------- | --------------------------------- |
| workclass      | Employment type | Private, Self-emp, Govt, etc.     |
| education      | Education level | Bachelors, HS-grad, Masters, etc. |
| marital-status | Marital status  | Married, Single, Divorced, etc.   |
| occupation     | Job type        | Tech, Sales, Exec, etc.           |
| relationship   | Family relation | Husband, Wife, etc.               |
| race           | Race category   | White, Black, Asian, etc.         |
| sex            | Gender          | Male, Female                      |
| native-country | Country         | US, India, etc.                   |

---

### 🎯 Target Variable

| Feature | Description  | Values      |
| ------- | ------------ | ----------- |
| income  | Income level | <=50K, >50K |

---

## 🧠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* PyCaret
* SHAP

---

## 📎 Author

Joyprakash Hela
