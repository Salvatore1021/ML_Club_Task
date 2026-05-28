import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_FEATURES = [
    "year",
    "make",
    "model",
    "trim",
    "body",
    "transmission",
    "state",
    "condition",
    "odometer",
    "color",
    "interior",
]

NUMERIC_FEATURES = [
    "year",
    "condition",
    "odometer",
    "car_age",
    "miles_per_year",
    "condition_x_age",
]

CATEGORICAL_FEATURES = [
    "make",
    "model",
    "trim",
    "body",
    "transmission",
    "state",
    "color",
    "interior",
    "body_transmission",
]

def _as_dataframe(x):
    if isinstance(x, pd.DataFrame):
        return x.copy()
    elif isinstance(x, dict):
        return pd.DataFrame([x])
    else:
        return pd.DataFrame(list(x))

def features(df, reference_year=None):
    out = _as_dataframe(df)
    for col in RAW_FEATURES:
        if col not in out.columns:
            out[col] = np.nan

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["condition"] = pd.to_numeric(out["condition"], errors="coerce")
    out["odometer"] = pd.to_numeric(out["odometer"], errors="coerce")

    # If we don't have a reference year, find the newest car year
    if reference_year is None:
        valid_year = out["year"].dropna()
        if len(valid_year) > 0:
            reference_year = int(valid_year.max())
        else:
            reference_year = 2015

    # Calculate car age and make sure its > 0
    out["car_age"] = reference_year - out["year"]
    out["car_age"] = out["car_age"].clip(lower=0)
    
    # more features
    out["miles_per_year"] = out["odometer"] / (out["car_age"] + 1.0)
    out["condition_x_age"] = out["condition"] * out["car_age"]

    body = out["body"].astype(str)
    body = body.fillna("unknown")

    transmission = out["transmission"].astype(str)
    transmission = transmission.fillna("unknown")

    out["body_transmission"] = body + "__" + transmission
    return out


class Auction(BaseEstimator, TransformerMixin):
    def __init__(self, min_category_frequency=25, max_categories_per_column=45, numeric_features=None, categorical_features=None):
        self.min_category_frequency = min_category_frequency
        self.max_categories_per_column = max_categories_per_column
        if numeric_features is None:
            self.numeric_features = NUMERIC_FEATURES.copy()
        else:
            self.numeric_features = numeric_features

        if categorical_features is None:
            self.categorical_features = CATEGORICAL_FEATURES.copy()
        else:
            self.categorical_features = categorical_features

    def fit(self, x, y=None):
        df = features(_as_dataframe(x))
        self.reference_year_ = int(pd.to_numeric(df["year"], errors="coerce").max())
        df = features(df, self.reference_year_)

        self.numeric_medians_ = {}
        self.numeric_caps_ = {}
    
        for col in self.numeric_features:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                self.numeric_medians_[col] = float(series.median())
                q01 = float(series.quantile(0.01))
                q99 = float(series.quantile(0.99))
            else:
                self.numeric_medians_[col] = 0.0
                q01 = 0.0
                q99 = 0.0
            self.numeric_caps_[col] = (q01, q99)

        self.categorical_modes_ = {}
        self.category_levels_ = {}
        
        for col in self.categorical_features:
            series = df[col].astype(str).fillna("unknown").str.lower().str.strip()
            mode = series.mode(dropna=True)
            if len(mode) > 0:
                self.categorical_modes_[col] = str(mode.iloc)
            else:
                self.categorical_modes_[col] = "unknown"
            counts = series.value_counts()
            frequent = counts[counts >= self.min_category_frequency].head(self.max_categories_per_column)
            levels = sorted(frequent.index.astype(str).tolist())
            
            if "other" not in levels:
                levels.append("other")
            self.category_levels_[col] = levels

        self.feature_names_ = self.numeric_features.copy()
        for col in self.categorical_features:
            for level in self.category_levels_[col]:
                self.feature_names_.append(f"{col}={level}")
                
        return self

    def transform(self, x):
        df = features(_as_dataframe(x), self.reference_year_)
        numeric_blocks = []
        for col in self.numeric_features:
            values = pd.to_numeric(df[col], errors="coerce")
            values = values.fillna(self.numeric_medians_[col])

            low = self.numeric_caps_[col]
            high = self.numeric_caps_[col]
            values_clipped = values.clip(lower=low, upper=high)
            numeric_blocks.append(values_clipped.to_numpy(dtype=float).reshape(-1, 1))

        categorical_blocks = []
        for col in self.categorical_features:
            values = df[col].astype(str)
            values = values.fillna(self.categorical_modes_[col])
            values = values.str.lower().str.strip()

            levels = self.category_levels_[col]
            known = set(levels)
            values = values.where(values.isin(known), "other")
            encoded = np.zeros((len(values), len(levels)), dtype=float)
            level_index = {}
            
            for idx, level in enumerate(levels):
                level_index[level] = idx
            for row_idx, value in enumerate(values):
                col_idx = level_index.get(str(value), level_index["other"])
                encoded[row_idx, col_idx] = 1.0
            categorical_blocks.append(encoded)
        return np.hstack(numeric_blocks + categorical_blocks)

    def get_feature_names_out(self):
        return np.array(self.feature_names_, dtype=object)