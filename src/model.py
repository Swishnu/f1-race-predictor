"""Predict each driver's finishing position from the features.

Why predict finishing position instead of "did they win"?  There have only been
14 winners in 2026, which is 14 positive examples. Every driver's finishing
position gives ~300 rows to learn from. The Monte Carlo step (simulate.py) turns
predicted positions into win probabilities.

Why Ridge (linear) instead of something fancier?  With ~300 rows, a linear model
is hard to overfit, and its coefficients tell you exactly what it learned
("one grid slot is worth X positions at the finish"), which is what you want
to be able to explain.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES


def usable_features(df):
    """FEATURES that actually have data (lets the pipeline run before every
    feature is implemented)."""
    cols = [f for f in FEATURES if df[f].notna().any()]
    missing = sorted(set(FEATURES) - set(cols))
    if missing:
        print(f"  [model] skipping unimplemented features: {missing}")
    return cols


def fit(train, cols):
    train = train[train["finish_pos"].notna()]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[cols], train["finish_pos"])
    return model


def predict(model, race, cols):
    return model.predict(race[cols])


def coefficients(model, cols):
    """Effect of +1 standard deviation of each feature on predicted finishing
    position (negative = helps you finish higher)."""
    ridge = model[-1]
    return dict(zip(cols, np.round(ridge.coef_, 2)))
