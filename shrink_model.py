"""
OPTIONAL: Run this locally in VS Code if you'd rather avoid Git LFS.

Your current model.pkl is ~144MB because it was trained with max_depth=None
(fully grown trees). This script re-trains a lighter RandomForestRegressor
with capped depth/estimators, which usually shrinks the file to a few MB
with only a small accuracy trade-off. Point TRAIN_CSV at your original
training data (the CSV that has the target column, e.g. median_house_value).
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

TRAIN_CSV = "your_training_data.csv"  # <-- update this path
TARGET_COLUMN = "median_house_value"  # <-- update if different

df = pd.read_csv(TRAIN_CSV).dropna()
df = pd.get_dummies(df, columns=["ocean_proximity"])

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,        # caps tree size -> much smaller file
    min_samples_leaf=5,  # further reduces tree size
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

print("R2 on test set:", r2_score(y_test, model.predict(X_test)))

joblib.dump(model, "model.pkl", compress=3)
print("Saved smaller model.pkl")
