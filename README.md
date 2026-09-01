      HEAD
# California House Price Predictor:- 🏠

A machine learning web app that predicts median house value for a California
district using a scikit-learn `RandomForestRegressor`, deployed with Streamlit.

**Live demo:**(https://house-price-prediction-ml-f4yn9qguxv8klzwytqj6eh.streamlit.app/)
---
## Features:-
- Interactive form for all 8 raw inputs (location, age, rooms, bedrooms,
  population, households, income, ocean proximity)
- One-hot encoding handled automatically to match the model's training format
- Instant price prediction with an interactive map showing the district location
- Cached model loading for fast repeat predictions

## Tech Stack:-
- **Model:**scikit-learn `RandomForestRegressor`
- **Frontend/App:**Streamlit
- **Data:** California Housing Prices dataset (Kaggle)

## Project Structure:-
```
streamlit_app/
├── app.py              # Streamlit application
├── model.pkl           # Trained model (loaded at runtime)
├── requirements.txt    # Python dependencies
├── shrink_model.py      # Optional: retrain a smaller model
└── README.md
=======
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
>>>>>>> d1f9837d3eb6c0186796672473c32c4ba63f4a93
```

---

HEAD
## 1. Run Locally

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Streamlit will open automatically at `http://localhost:8501`.

---

## 2. Push to GitHub

Your `model.pkl` is **144MB**, and GitHub blocks any file over 100MB pushed
with normal `git add`/`git push`. You have two options — pick one.

### Option A: Git LFS (keeps your exact trained model)

```bash
# Install Git LFS once (https://git-lfs.com)
git lfs install

cd streamlit_app
git init
git lfs track "*.pkl"
git add .gitattributes
git add .
git commit -m "Initial commit: California house price predictor"

# Create a new repo on GitHub first, then:
git remote add origin https://github.com/<your-username>/california-house-price-predictor.git
git branch -M main
git push -u origin main
```

Note: GitHub's free tier includes 1GB of LFS storage and 1GB/month of
bandwidth, which is enough for one 144MB model with normal usage.

### Option B: Shrink the model (no LFS needed)

Run `shrink_model.py` locally (update the `TRAIN_CSV` and `TARGET_COLUMN`
variables to point at your original training CSV first). Capping `max_depth`
typically brings a Random Forest like this down to a few MB with a small,
often negligible, accuracy trade-off. Then just `git add`/`git push` normally.

---

## 3. Deploy on Streamlit Community Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository, branch (`main`), and main file path (`app.py`).
4. Click **"Deploy"**.
5. Wait a few minutes for the build — Streamlit installs `requirements.txt`
   and starts the app. You'll get a public URL like:
   `https://<your-app-name>.streamlit.app`

That URL is what you share on your resume/portfolio.

---

## 4. Resume Framing

**Project bullet points (pick 2-3):**
- Built and deployed an end-to-end ML web app predicting California house
  prices using a scikit-learn Random Forest Regressor and Streamlit
- Designed an interactive UI with real-time predictions and geographic
  visualization, deployed on Streamlit Community Cloud
- Handled categorical feature encoding (one-hot) and model serialization
  (joblib) to serve a trained model in a production-style app

**Resume line:**
> California House Price Predictor — Streamlit app serving a trained Random
> Forest model to predict median house values from location, income, and
> housing attributes; deployed publicly on Streamlit Cloud. [GitHub] [Live Demo]

**Possible interview questions to prep for:**
- Why Random Forest over Linear Regression here?
- How did you handle the categorical `ocean_proximity` feature?
- How would you monitor/retrain this model if data drifted over time?
- What's the trade-off between model size (max_depth) and accuracy?
- How would you turn this into a REST API (FastAPI) instead of Streamlit?
=======
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

## ▶️ Run the Project:-

Train the model and generate predictions

```bash
python src/main.py
```

To compare machine learning models:-

```bash
python src/model_comparison.py
```

---

## 📊 Machine Learning Models:-

The following regression models are implemented:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Model performance is evaluated using **10-Fold Cross Validation** with **Root Mean Squared Error (RMSE)**.

---

## 📈 Dataset Features:-

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

## 📌 Future Improvements:-

- Hyperparameter tuning using GridSearchCV
- Feature importance visualization
- Model deployment using Flask or FastAPI
- Interactive web interface with Streamlit
- Docker containerization

---

## 📄 License:-

This project is licensed under the MIT License.

---

## 👨‍💻 Author:-

**Tejas Shinde**

- GitHub: https://github.com/Tejas1707-bit

If you found this project useful, consider giving it a ⭐ on GitHub.
>>>>>>> d1f9837d3eb6c0186796672473c32c4ba63f4a93
