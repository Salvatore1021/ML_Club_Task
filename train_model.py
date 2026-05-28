"""
Train and tune the used-car auction price model.

Outputs:
    model_Daksh.pkl
    encoders_Daksh.pkl
    metrics_Daksh.csv
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# Import our custom preprocessor and feature list
from preprocessing import AuctionPreprocessor, RAW_FEATURES

# Define simple text strings for where to save and load files
DATA_PATH = "C:\\Users\\Daksh\\Downloads\\ML Recruitment\\ML_Club_Task\\car_auction_train.csv"
MODEL_PATH = "model_Daksh.pkl"
ENCODER_PATH = "encoders_Daksh.pkl"
METRICS_PATH = "metrics_Daksh.csv"
RANDOM_STATE = 42

def clean_training_frame(df):
    """Clean the raw data before splitting it."""
    
    # 1. Make sure all the columns we expect are actually in the file
    keep_cols = RAW_FEATURES + ["sellingprice"]
    
    for col in keep_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # 2. Extract just the columns we need
    out = df[keep_cols].copy()
    
    # 3. Clean up the target variable (sellingprice)
    out["sellingprice"] = pd.to_numeric(out["sellingprice"], errors="coerce")
    out = out.dropna(subset=["sellingprice"])
    out = out[out["sellingprice"] > 0]

    # 4. Remove extreme outliers (top and bottom 0.5%)
    low_cutoff = out["sellingprice"].quantile(0.005)
    high_cutoff = out["sellingprice"].quantile(0.995)
    
    out = out[(out["sellingprice"] >= low_cutoff) & (out["sellingprice"] <= high_cutoff)]
    
    return out.reset_index(drop=True)

def evaluate_model(name, y_true, y_pred):
    """Helper function to calculate error metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    results = {
        "model": name,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
    }
    return results

# =====================================================================
# Main Execution Steps
# (Written top-to-bottom like a standard data science script)
# =====================================================================

try:
    # 1. Load the raw data
    raw_data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: {DATA_PATH} was not found. Place it beside this script and rerun.")
    exit()

# 2. Clean the data
df = clean_training_frame(raw_data)

# 3. Separate features (X) and target (y)
X = df[RAW_FEATURES]
y = df["sellingprice"]

# 4. Split into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# 5. Initialize and run our custom data encoders
encoders = AuctionPreprocessor(min_category_frequency=25, max_categories_per_column=45)
X_train_enc = encoders.fit_transform(X_train)
X_valid_enc = encoders.transform(X_valid)

# 6. Train a basic default model first
print("Training default Random Forest...")
default_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
default_model.fit(X_train_enc, y_train)

# 7. Set up the hyperparameter tuning (Random Search)
print("Starting Random Search to tune the model (this may take a while)...")
base_for_tuning = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [12, 18, 26, None],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features": ["sqrt", 0.5, 0.8],
}

search = RandomizedSearchCV(
    estimator=base_for_tuning,
    param_distributions=param_grid,
    n_iter=24,
    scoring="neg_mean_absolute_error",
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

# Run the search and extract the winner
search.fit(X_train_enc, y_train)
tuned_model = search.best_estimator_

# 8. Evaluate both models using the validation data
print("Evaluating models...")
default_predictions = default_model.predict(X_valid_enc)
tuned_predictions = tuned_model.predict(X_valid_enc)

default_report = evaluate_model("default_random_forest", y_valid, default_predictions)
tuned_report = evaluate_model("tuned_random_forest", y_valid, tuned_predictions)

# Put the reports into a DataFrame
metrics = pd.DataFrame([default_report, tuned_report])

# Add a column to show the best settings we found
metrics["best_params"] = ["", str(search.best_params_)]

# Save the metrics to a CSV
metrics.to_csv(METRICS_PATH, index=False)

# 9. Save the finished model and the encoders to disk
with open(MODEL_PATH, "wb") as f:
    pickle.dump(tuned_model, f)
    
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(encoders, f)

# 10. Print the final results
print("\n--- Final Metrics ---")
print(metrics.to_string(index=False))
print(f"\nSaved {MODEL_PATH} and {ENCODER_PATH} successfully!")