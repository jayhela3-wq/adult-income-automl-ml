# Adult Income Prediction using Machine Learning

## 📌 Overview

This project predicts whether a person earns more than $50K per year using Machine Learning.

The project includes:

* Manual model (Logistic Regression)
* AutoML using PyCaret (multiple models comparison)
* Data preprocessing and cleaning
* Feature correlation analysis
* Prediction using JSON input file

---

## 📁 Project Structure

* `model.py` → manual model training (Logistic Regression)
* `predict.py` → prediction using manual model
* `model_automl.py` → AutoML using PyCaret
* `predict_automl.py` → prediction using AutoML model
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

### 🔹 Manual Model

```bash
python model.py
python predict.py
```

---

### 🔹 AutoML Model (Recommended)

```bash
python model_automl.py
python predict_automl.py
```

⚠️ Run `model_automl.py` first to generate the trained model file before prediction.

---

## 📄 Input Format (JSON)

Example input file:

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
* AutoML selects best model (Random Forest, XGBoost, etc.)
* Performance comparison done automatically using PyCaret

---

## 🔍 Feature Analysis

* Correlation analysis performed using encoded dataset
* Helps understand relationships between features and target variable

### Key Insight:

> Features with high correlation often have strong influence, but AutoML captures more complex patterns.

---

## ⚙️ Approach and Implementation

### 📌 Approach

The project follows a structured Machine Learning workflow:

1. Data cleaning and preprocessing
2. Feature handling (categorical + numerical)
3. Model training and evaluation
4. Model comparison using AutoML
5. Prediction using saved model

---

### 🛠️ Implementation

* Cleaned dataset (handled missing values, removed spaces)
* Used OneHotEncoding and scaling (manual model)
* Used Pipeline for structured workflow
* Used PyCaret for:

  * Automatic preprocessing
  * Training multiple models
  * Selecting best model
* Saved model using Pickle
* Built prediction system using JSON input

---

## 🤖 AutoML using PyCaret

PyCaret is used to automate model selection:

* Trains multiple models (Logistic Regression, Random Forest, XGBoost, etc.)
* Compares performance automatically
* Selects best model
* Reduces manual effort and improves accuracy

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

---

## 📎 Author

Joyprakash Hela
