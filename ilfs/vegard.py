from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from .constants import EPS_INPUTS, LGSSPS

ELEMENTS = ["Li", "Ge", "Si", "Sn", "P", "Sb", "O", "S", "Cl", "Br"]
V_REF = 903.25                 # V_perfect of the pure-S LGPS-type reference (A^3)


def lgps_structures(s1):
    g = s1[s1.family == "LGPS-type"].copy()
    g["eps LOCO"] = cross_val_predict(LinearRegression(), g[EPS_INPUTS["LGPS-type"]].values, g.ILFS.values,
                                      groups=g.comp, cv=LeaveOneGroupOut())
    g["V/Z"] = g["Cell volume"] / g["Cell formula units Z"]
    return g


def select(g, li=None, p=None, **n):
    """Room-temperature (290-300 K) all-sulfide structures; n: element -> (min, max) or 0."""
    m = g["T"].between(290, 300) & g.S.eq(12) & g[["O", "Cl", "Br", "Sb"]].fillna(0).eq(0).all(axis=1)
    if li is not None:
        m &= g.Li.eq(li)
    if p is not None:
        m &= g.P.eq(p)
    for el, v in n.items():
        m &= g[el].between(*v) if isinstance(v, tuple) else g[el].fillna(0).eq(v)
    return g[m]


def ge_sn_series(g):
    """Li10Ge1-xSnxP2S12: interpolation between the room-temperature end members."""
    ge = select(g, 10, 2, Ge=1, Si=0, Sn=0).loc[:, "V/Z"].mean()
    sn = select(g, 10, 2, Sn=1, Ge=0, Si=0).loc[:, "V/Z"].mean()
    m = select(g, 10, 2, Ge=(1e-6, 0.999), Sn=(1e-6, 0.999), Si=0)
    x = m.Sn / (m.Ge + m.Sn)
    vegard = ((1 - x) * ge + x * sn) * m["Cell formula units Z"] / m.V_perfect - 1
    return pd.DataFrame({"structures": [len(m)], "Vegard MAE (pp)": [100 * (vegard - m.ILFS).abs().mean()],
                         "eps-predictor LOCO MAE (pp)": [100 * (m["eps LOCO"] - m.ILFS).abs().mean()]})


def linear_composition(g):
    """ILFS linear in the elemental composition vs the eps-predictor, leave-one-composition-out."""
    p = cross_val_predict(LinearRegression(), g[ELEMENTS].fillna(0).values, g.ILFS.values, groups=g.comp,
                          cv=LeaveOneGroupOut())
    return pd.DataFrame({"structures": [len(g)], "Vegard-type LOCO R2": [r2_score(g.ILFS, p)],
                         "eps-predictor LOCO R2": [r2_score(g.ILFS, g["eps LOCO"])]})


def lgssps_ternary(g, lgssps, eps_p):
    """Ge-Si-Sn interpolation; the Li10SiP2S12 volume is estimated as
    V(Li10.35Si1.35P1.65S12) - [V(Li10.35Ge1.35P1.65S12) - V(Li10GeP2S12)]."""
    ge = select(g, 10, 2, Ge=1, Si=0, Sn=0)["V/Z"].values
    sn = select(g, 10, 2, Sn=1, Ge=0, Si=0)["V/Z"].values
    ge135 = select(g, Ge=(1.3, 1.4), Si=0, Sn=0)["V/Z"].values
    si135 = select(g, Si=(1.3, 1.4), Ge=0, Sn=0)["V/Z"].mean()
    f = lgssps[["Ge", "Si", "Sn"]].values
    exp = LGSSPS.eps_exp.values
    mae = []
    for a, b, c in product(ge, sn, ge135):
        pred = 2 * (f @ np.array([a, si135 - (c - a), b])) / V_REF - 1
        mae.append(100 * np.abs(pred - exp).mean())
    return pd.DataFrame({"end-member combinations": [len(mae)], "Vegard MAE min (pp)": [min(mae)],
                         "Vegard MAE max (pp)": [max(mae)], "eps-predictor MAE (pp)": [100 * np.abs(eps_p - exp).mean()],
                         "Li10GeP2S12 ILFS min (%)": [100 * (2 * ge.min() / V_REF - 1)],
                         "Li10GeP2S12 ILFS max (%)": [100 * (2 * ge.max() / V_REF - 1)],
                         "LGSSPS ILFS min (%)": [100 * exp.min()], "LGSSPS ILFS max (%)": [100 * exp.max()]})
