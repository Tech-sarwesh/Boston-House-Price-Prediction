import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from data_preprocessing import (
    load_data,
    split_features_target,
    split_train_test
)



# 1. Load dataset


df = load_data("data/raw/HousingData.csv")



# 2. Separate features and target


X, y = split_features_target(df)



# 3. Train-test split


X_train, X_test, y_train, y_test = split_train_test(X, y)



# 4. Create ML pipeline


pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "model",
        GradientBoostingRegressor(
            random_state=42
        )
    )
])


# 5. Hyperparameter grid


param_grid = {
    "model__n_estimators": [100, 150, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__min_samples_split": [2, 5]
}



# 6. GridSearchCV


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)



# 7. Train


grid_search.fit(X_train, y_train)



# 8. Get best model


best_model = grid_search.best_estimator_



# 9. Save model


joblib.dump(
    best_model,
    "models/best_model.pkl"
)


# 10. Display results

print("Training completed successfully!")

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest CV Score:")
print(grid_search.best_score_)

print("\nModel saved to:")
print("models/best_model.pkl")