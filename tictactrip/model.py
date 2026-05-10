"""
model.py
--------
XGBoost price prediction model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


FEATURES: List[str] = [
    "distance_km",
    "duration_h",
    "transport_enc",
    "dep_hour",
    "dep_dow",
    "dep_month",
    "days_advance",
    "is_weekend",
    "o_city_enc",
    "d_city_enc",
]

TARGET = "price_eur"

XGBOOST_PARAMS: Dict = {
    "n_estimators":     500,
    "max_depth":        7,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "verbosity":        0,
    "n_jobs":           -1,
}


@dataclass
class EvaluationResult:
    mae:                   float
    rmse:                  float
    r2:                    float
    mape:                  float
    mae_by_transport:      pd.Series
    mae_by_distance_range: pd.Series
    y_test:                pd.Series
    y_pred:                np.ndarray

    def summary(self) -> str:
        lines = [
            "=" * 45,
            "  XGBOOST EVALUATION RESULTS",
            "=" * 45,
            f"  MAE   (mean absolute error)  {self.mae:>8.2f} EUR",
            f"  RMSE  (root mean sq. error)  {self.rmse:>8.2f} EUR",
            f"  R2    (variance explained)   {self.r2:>8.4f}",
            f"  MAPE  (mean abs pct error)   {self.mape:>7.1f} %",
            "",
            "  MAE by transport:",
        ]
        for t, v in self.mae_by_transport.items():
            lines.append(f"    {t:<15} {v:.2f} EUR")
        lines.append("")
        lines.append("  MAE by distance range:")
        for r, v in self.mae_by_distance_range.items():
            lines.append(f"    {str(r):<15} {v:.2f} EUR")
        return "\n".join(lines)


class PricePredictor:
    """
    XGBoost-based price regression model for Tictactrip ticket data.

    Usage
    -----
    >>> predictor = PricePredictor()
    >>> predictor.fit(df)
    >>> result = predictor.evaluate()
    >>> print(result.summary())
    """

    def __init__(self) -> None:
        self._model        = XGBRegressor(**XGBOOST_PARAMS)
        self._le_transport = LabelEncoder()
        self._le_o_city    = LabelEncoder()
        self._le_d_city    = LabelEncoder()
        self._X_test:  Optional[pd.DataFrame] = None
        self._y_test:  Optional[pd.Series]    = None
        self._ml_df:   Optional[pd.DataFrame] = None
        self._is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> "PricePredictor":
        """Prepare features, encode categoricals, and train the model."""
        ml = df[
            (df["days_advance"] >= 0) & (df["duration_h"] <= 48)
        ].copy()

        ml["transport_enc"] = self._le_transport.fit_transform(
            ml["transport_type"].fillna("unknown")
        )
        ml["o_city_enc"] = self._le_o_city.fit_transform(
            ml["o_city_name"].fillna("unknown")
        )
        ml["d_city_enc"] = self._le_d_city.fit_transform(
            ml["d_city_name"].fillna("unknown")
        )
        ml["is_weekend"] = ml["is_weekend"].astype(int)

        X = ml[FEATURES].copy()
        y = ml[TARGET].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self._model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        self._X_test    = X_test
        self._y_test    = y_test
        self._ml_df     = ml
        self._is_fitted = True
        return self

    def evaluate(self) -> EvaluationResult:
        """Compute test-set metrics and segment-level MAE breakdowns."""
        self._assert_fitted()

        y_pred = self._model.predict(self._X_test)
        y_true = self._y_test

        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = float(r2_score(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true.values - y_pred) / y_true.values)) * 100)

        results = self._X_test.copy()
        results["y_true"]         = y_true.values
        results["y_pred"]         = y_pred
        results["abs_error"]      = np.abs(y_true.values - y_pred)
        results["transport_type"] = self._le_transport.inverse_transform(
            results["transport_enc"]
        )
        results["distance_range"] = pd.cut(
            self._ml_df.loc[self._X_test.index, "distance_km"],
            bins=[0, 200, 800, 2000, float("inf")],
            labels=["0-200 km", "201-800 km", "801-2000 km", "2000+ km"],
        )

        return EvaluationResult(
            mae=mae, rmse=rmse, r2=r2, mape=mape,
            mae_by_transport=results.groupby("transport_type")["abs_error"].mean().round(2),
            mae_by_distance_range=results.groupby("distance_range", observed=True)["abs_error"].mean().round(2),
            y_test=y_true,
            y_pred=y_pred,
        )

    def predict_price(
        self,
        distance_km:  float,
        duration_h:   float,
        transport:    str,
        dep_hour:     int,
        dep_dow:      int,
        dep_month:    int,
        days_advance: float,
        is_weekend:   bool,
        o_city:       str,
        d_city:       str,
    ) -> float:
        """Predict the price of a single trip in EUR."""
        self._assert_fitted()
        t_enc = self._safe_encode(self._le_transport, transport)
        o_enc = self._safe_encode(self._le_o_city,    o_city)
        d_enc = self._safe_encode(self._le_d_city,    d_city)
        X = np.array([[
            distance_km, duration_h, t_enc,
            dep_hour, dep_dow, dep_month,
            days_advance, int(is_weekend),
            o_enc, d_enc,
        ]])
        return float(self._model.predict(X)[0])

    @property
    def feature_importances(self) -> pd.Series:
        self._assert_fitted()
        return pd.Series(self._model.feature_importances_, index=FEATURES)

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before using this method.")

    @staticmethod
    def _safe_encode(encoder: LabelEncoder, value: str) -> int:
        return int(encoder.transform([value])[0]) if value in list(encoder.classes_) else 0
