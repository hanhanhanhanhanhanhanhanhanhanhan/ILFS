import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils import check_random_state
from xgboost import XGBRegressor

from .constants import EPS_INPUTS

SEED = 178
N_CANDIDATES = 30
INPUTS = EPS_INPUTS["LGPS-type"]


def loco_predict(make, d):
    p = np.full(len(d), np.nan)
    for tr, te in LeaveOneGroupOut().split(d, groups=d.comp):
        p[te] = make().fit(d.iloc[tr].reset_index(drop=True)).predict(d.iloc[te])
    return p


class OLS:
    def __init__(self, columns):
        self.columns = columns

    def fit(self, d):
        self.m = LinearRegression().fit(self.columns(d), d.target.values)
        return self

    def predict(self, d):
        return self.m.predict(self.columns(d))


def inv_t(d):
    return 1 / d[["T"]].values


def inputs_inv_t(d):
    return np.column_stack([d[INPUTS].values, 1 / d["T"].values])


def ml_features(d):
    return np.column_stack([d[INPUTS].astype(float).values, 1000.0 / d["T"].astype(float).values])


class HiddenLayers:
    def rvs(self, random_state=None):
        rng = check_random_state(random_state)
        return tuple(10 * int(u) for u in rng.randint(1, 10, size=rng.randint(1, 4)))


def search(est, space):
    return RandomizedSearchCV(est, space, n_iter=N_CANDIDATES, cv=LeaveOneGroupOut(),
                              scoring="neg_mean_squared_error", random_state=SEED)


def searches():
    rf = RandomForestRegressor(random_state=SEED, n_jobs=1)
    xgb = XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0)
    svr = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
    mlp = make_pipeline(StandardScaler(), MLPRegressor(solver="adam", activation="relu", early_stopping=True,
                                                       validation_fraction=0.1, max_iter=2000, shuffle=True,
                                                       random_state=SEED))
    return {
        "RF": lambda: search(rf, {"n_estimators": randint(1, 101), "max_depth": randint(1, 20),
                                  "max_features": randint(1, 10), "min_samples_leaf": randint(1, 21),
                                  "min_samples_split": randint(2, 22),
                                  "criterion": ["squared_error", "friedman_mse", "absolute_error"]}),
        "XGBoost": lambda: search(xgb, {"n_estimators": randint(1, 200), "max_depth": randint(1, 30),
                                        "learning_rate": uniform(0.01, 0.98), "subsample": uniform(0.01, 0.99),
                                        "gamma": uniform(0, 1), "reg_alpha": uniform(0, 9.5),
                                        "reg_lambda": uniform(0, 9.5), "colsample_bytree": uniform(0.5, 0.49),
                                        "colsample_bylevel": uniform(0.5, 0.49), "colsample_bynode": uniform(0.5, 0.49)}),
        "SVM": lambda: search(svr, {"svr__C": loguniform(1e-3, 1e3), "svr__gamma": loguniform(1e-3, 1e1),
                                    "svr__epsilon": loguniform(1e-3, 1e1)}),
        "MLP": lambda: search(mlp, {"mlpregressor__hidden_layer_sizes": HiddenLayers(),
                                    "mlpregressor__alpha": loguniform(1e-6, 1e0),
                                    "mlpregressor__learning_rate_init": loguniform(1e-5, 1e-1),
                                    "mlpregressor__tol": loguniform(1e-5, 1e-1),
                                    "mlpregressor__beta_1": uniform(0.1, 0.8),
                                    "mlpregressor__beta_2": uniform(0.001, 0.998),
                                    "mlpregressor__batch_size": randint(1, 300),
                                    "mlpregressor__n_iter_no_change": randint(1, 41)}),
    }


def tune(name, X, y, groups, n_jobs=-1):
    m = searches()[name]().set_params(n_jobs=n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m.fit(X, y, groups=groups)
    return m


def ml_loco(best, X, y, groups, n_jobs=-1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return cross_val_predict(clone(best), X, y, groups=groups, cv=LeaveOneGroupOut(), n_jobs=n_jobs)


def composition_mae(p, d):
    return pd.Series(np.abs(p - d.target.values)).groupby(d.comp.values).mean()


def bootstrap_ci(x, rng, n, weights=None):
    idx = rng.integers(0, len(x), (n, len(x)))
    s = x[idx].mean(axis=1) if weights is None else (x[idx] * weights[idx]).sum(axis=1) / weights[idx].sum(axis=1)
    return np.percentile(s, [2.5, 97.5])
