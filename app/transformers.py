from sklearn.base import BaseEstimator, TransformerMixin


class FinancialRatiosTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["CREDIT_INCOME_PERCENT"] = X["AMT_CREDIT"] / (
            X["AMT_INCOME_TOTAL"] + 1e-6
        )
        X["ANNUITY_INCOME_PERCENT"] = X["AMT_ANNUITY"] / (
            X["AMT_INCOME_TOTAL"] + 1e-6
        )
        X["CREDIT_TERM"] = X["AMT_CREDIT"] / (X["AMT_ANNUITY"] + 1e-6)
        X["GOODS_PRICE_RATIO"] = X["AMT_CREDIT"] / (
            X["AMT_GOODS_PRICE"] + 1e-6
        )

        if "DAYS_BIRTH" in X.columns:
            X["AGE_YEARS"] = -X["DAYS_BIRTH"] / 365.0

        return X