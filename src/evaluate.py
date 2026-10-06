import joblib
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_preprocessing import (
    load_data,
    split_features_target,
    split_train_test
)



# 1. Load dataset

df = load_data("data/raw/HousingData.csv")


# 2. Separate features and target

X, y = split_features_target(df)


# 3. Same train-test split

X_train, X_test, y_train, y_test = split_train_test(X, y)


# 4. Load trained model

model = joblib.load("models/best_model.pkl")


# 5. Make predictions

y_pred = model.predict(X_test)


# 6. Calculate metrics

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


# 7. Display results

print("Final Model Evaluation")
print("----------------------")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)