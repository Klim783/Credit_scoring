# app/transformers.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FinancialRatiosTransformer(BaseEstimator, TransformerMixin):
    """
    Computes domain-specific financial ratios for Home Credit data.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # Avoid division by zero using safe replacement
        income = X_out["AMT_INCOME_TOTAL"].replace(0, np.nan)
        credit = X_out["AMT_CREDIT"].replace(0, np.nan)
        annuity = X_out["AMT_ANNUITY"].replace(0, np.nan)
        goods = X_out["AMT_GOODS_PRICE"].replace(0, np.nan) if "AMT_GOODS_PRICE" in X_out.columns else np.nan

        X_out["CREDIT_INCOME_PERCENT"] = X_out["AMT_CREDIT"] / income
        X_out["ANNUITY_INCOME_PERCENT"] = X_out["AMT_ANNUITY"] / income
        X_out["CREDIT_TERM"] = X_out["AMT_ANNUITY"] / credit
        X_out["GOODS_PRICE_RATIO"] = X_out["AMT_CREDIT"] / goods

        if "DAYS_BIRTH" in X_out.columns and "DAYS_EMPLOYED" in X_out.columns:
            days_birth = X_out["DAYS_BIRTH"].replace(0, np.nan)
            X_out["DAYS_EMPLOYED_PERCENT"] = X_out["DAYS_EMPLOYED"] / days_birth

        if "DAYS_BIRTH" in X_out.columns:
            X_out["AGE_YEARS"] = -X_out["DAYS_BIRTH"] / 365.0

        return X_out