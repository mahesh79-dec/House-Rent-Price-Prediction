"""
Train a house rent prediction model.
Dataset included in data/ is SYNTHETIC demonstration data, not actual market listings.
Replace it with a verified real-world dataset before making real rental decisions.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "house_rent_data.csv"
MODEL_PATH = ROOT / "models" / "rent_model.joblib"

def main():
    data = pd.read_csv(DATA_PATH)
    target = "Rent"
    X = data.drop(columns=[target])
    y = data[target]

    categorical = ["City", "Locality", "Furnishing", "Property_Type"]
    numerical = ["BHK", "Size_sqft", "Bathrooms", "Floor"]
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numerical", "passthrough", numerical),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=250, max_depth=18, min_samples_leaf=2, random_state=42
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved to:", MODEL_PATH)
    print(f"Test MAE: ₹{mae:,.0f}")
    print(f"Test RMSE: ₹{rmse:,.0f}")
    print(f"Test R²: {r2:.3f}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("\nImportant: metrics describe the included synthetic demo dataset only.")

if __name__ == "__main__":
    main()
