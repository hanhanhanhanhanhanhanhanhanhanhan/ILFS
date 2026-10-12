import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from .constants import EPS_INPUTS, FAMILIES
from .metrics import regression


def fit(s1):
    return {f: LinearRegression().fit(g[EPS_INPUTS[f]].values, g.ILFS.values) for f, g in s1.groupby("family")}


def performance(s1):
    rows = []
    for f in FAMILIES:
        g = s1[s1.family == f]
        X, y = g[EPS_INPUTS[f]].values, g.ILFS.values
        p_in = LinearRegression().fit(X, y).predict(X)
        p_lo = cross_val_predict(LinearRegression(), X, y, groups=g.comp, cv=LeaveOneGroupOut())
        rows.append({"family": f, "structures": len(g), "compositions": g.comp.nunique(), "evaluation": "in-sample",
                     **regression(y, p_in)})
        rows.append({"family": f, "structures": len(g), "compositions": g.comp.nunique(), "evaluation": "LOCO",
                     **regression(y, p_lo)})
    return pd.DataFrame(rows)


def predict(models, family, rows):
    return models[family].predict(np.asarray(rows[EPS_INPUTS[family]], float))
