# Adult Income Prediction using Machine Learning

## 📌 Overview

This project predicts whether a person earns more than $50K per year using Machine Learning.

The project includes:

* Manual model (Logistic Regression)
* AutoML using PyCaret (multiple models comparison)
* Feature Engineering model (correlation + SelectKBest)
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
* `model_fg.py` → feature engineering + feature selection model
* `predict_fg.py` → prediction using FG model
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

---

### 🔹 Feature Engineering Model (FG)

```bash
python model_featuregeneration.py
python predict_featuregeneration.py
```  


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
* AutoML selects best model automatically
* Feature Engineering model improves efficiency and reduces redundancy

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

---

## ⚙️ Approach and Implementation

### 📌 Approach

1. Data cleaning (handling missing values, removing spaces)
2. Feature preprocessing (encoding + scaling)
3. Model training (Logistic Regression)
4. Model comparison using AutoML (PyCaret)
5. Feature engineering (FG model)
6. Model evaluation
7. Explainability using SHAP
8. Prediction using JSON input

---

### 🛠️ Implementation

* Used Pipeline for preprocessing + model
* Used OneHotEncoding for categorical features
* Used StandardScaler for numerical features
* Used PyCaret for AutoML
* Used SelectKBest for feature selection
* Used Pickle for saving models
* Built prediction system using JSON input

---


## 🤖 AutoML using PyCaret

* Trains multiple models automatically
* Compares performance
* Selects best model
* Improves accuracy and reduces manual effort

---


## 🔧 Feature Engineering Model (FG)

This model focuses on improving performance using feature selection techniques.

---


### Techniques:
* Correlation-based feature removal (>0.9)
* SelectKBest (Chi-square)

### Advantages:
* Reduced features
* Faster training
* Better generalization

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


