import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ==========================================================
# 1. Load Dataset
# ==========================================================

housing = pd.read_csv("data/housing.csv")

# ==========================================================
# 2. Create Stratified Train-Test Split
# ==========================================================

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

# ==========================================================
# 3. Separate Features and Labels
# ==========================================================

housing = strat_train_set.copy()

housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# ==========================================================
# 4. Numerical and Categorical Columns
# ==========================================================

num_attributes = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attributes = ["ocean_proximity"]

# ==========================================================
# 5. Data Preprocessing Pipeline
# ==========================================================

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", cat_pipeline, cat_attributes)
])

housing_prepared = full_pipeline.fit_transform(housing)

# ==========================================================
# 6. Linear Regression
# ==========================================================

lin_reg = LinearRegression()

lin_scores = -cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

print("\nLinear Regression")
print(pd.Series(lin_scores).describe())

# ==========================================================
# 7. Decision Tree Regressor
# ==========================================================

tree_reg = DecisionTreeRegressor(random_state=42)

tree_scores = -cross_val_score(
    tree_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

print("\nDecision Tree")
print(pd.Series(tree_scores).describe())

# ==========================================================
# 8. Random Forest Regressor
# ==========================================================

forest_reg = RandomForestRegressor(random_state=42)

forest_scores = -cross_val_score(
    forest_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

print("\nRandom Forest")
print(pd.Series(forest_scores).describe())