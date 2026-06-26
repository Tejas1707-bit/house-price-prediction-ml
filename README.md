# 🏠 House Price Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue?logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

This project predicts house prices using Machine Learning techniques on the California Housing Dataset. It demonstrates a complete machine learning workflow including data preprocessing, feature engineering, model training, evaluation, and prediction using Scikit-Learn pipelines.

The project compares multiple regression algorithms and uses cross-validation to evaluate model performance.

---

## 🚀 Features

- Data preprocessing using Scikit-Learn Pipelines
- Missing value handling using SimpleImputer
- Feature scaling using StandardScaler
- One-Hot Encoding for categorical features
- Stratified Train-Test Split
- Model comparison using:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Cross-validation for model evaluation
- Prediction pipeline for inference

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib

---

## 📂 Project Structure

```text
house-price-prediction-ml/
│
├── data/
│   ├── housing.csv
│   ├── input.csv
│   └── output.csv
│
├── images/
│
├── models/
│   └── pipeline.pkl
│
├── src/
│   ├── main.py
│   └── model_comparison.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/house-price-prediction-ml.git
```

Go to the project directory:

```bash
cd house-price-prediction-ml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Train the model and generate predictions:

```bash
python src/main.py
```

To compare machine learning models:

```bash
python src/model_comparison.py
```

---

## 📊 Machine Learning Models

The following regression models are implemented:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Model performance is evaluated using **10-Fold Cross Validation** with **Root Mean Squared Error (RMSE)**.

---

## 📈 Dataset Features

The dataset contains features such as:

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

Target Variable:

- Median House Value

---

## 📌 Future Improvements

- Hyperparameter tuning using GridSearchCV
- Feature importance visualization
- Model deployment using Flask or FastAPI
- Interactive web interface with Streamlit
- Docker containerization

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Tejas Shinde**

- GitHub: https://github.com/Tejas1707-bit

If you found this project useful, consider giving it a ⭐ on GitHub.
