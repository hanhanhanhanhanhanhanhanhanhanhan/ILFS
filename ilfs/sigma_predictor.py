import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.special import gammaln
from sklearn.linear_model import LinearRegression

from .constants import EV, G, HBAR, KB, LGPS_REF, POISSON, T25

BP = 4.0                       # B0' of the Birch-Murnaghan energy
THIO_TERMS = ["eps", "eps2*V0/T", "ln(1+eps)", "eps*ln(1+eps)", "eps2", "eps2*ln(1+eps)", "eps*V0/T", "Smix/kB"]


def arrhenius_table(d):
    """Per composition with >= 3 temperatures: ln(sigma T) = A - Ea/(kB T)."""
    rows = []
    for c, g in d.groupby("comp"):
        if g["T"].nunique() >= 3:
            slope, icpt = np.polyfit(1 / g["T"].values, g.target.values, 1)
            rows.append({"comp": c, "Ea": -KB * slope, "A": icpt, "eps": g.eps.mean(), "V0": g.V0.mean(),
                         "Smix/kB": g["Smix/kB"].mean(), "atoms": g.atoms.iloc[0]})
    return pd.DataFrame(rows)


def poisson_ratio(b, g):
    return (3 * b - 2 * g) / (2 * (3 * b + g))


def debye_energy(B0, poisson=POISSON, ref=LGPS_REF):
    B = B0 * 1e9
    rho = ref["density_g_cm3"] * 1000
    Gs = 3 * B * (1 - 2 * poisson) / (2 * (1 + poisson))
    v_l, v_t = np.sqrt((B + 4 * Gs / 3) / rho), np.sqrt(Gs / rho)
    v_m = (3 / (1 / v_l ** 3 + 2 / v_t ** 3)) ** (1 / 3)
    k_d = (6 * np.pi ** 2 * ref["atoms"] / (ref["volume_A3"] * 1e-30)) ** (1 / 3)
    return HBAR * v_m * k_d / EV


def lattice_energy(form, B0, V0, e):
    if form == "second-order":
        return 0.5 * B0 * V0 * e ** 2
    y = (1 + e) ** (-2 / 3)
    x = y - 1
    return 9 * V0 * B0 / 16 * (x ** 3 * BP + x ** 2 * (6 - 4 * y))


def excitation_entropy(form, n0, gamma, N, e):
    n = n0 * (1 + e) ** gamma
    if form == "Stirling":
        return n * (np.log(N / n0) - gamma * np.log1p(e))
    return gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1)


class SigmaPredictor:
    """ln(sigma T) = C + dS_mix/kB + dS_M(eps)/kB - [Ea0 - P0 V0 eps + 1/2 B0 V0 eps^2]/(kB T)."""

    def __init__(self, hw="debye", N="reference", poisson=POISSON, hform="second-order", sform="Stirling", tmax=None):
        self.hw_spec, self.N_spec, self.poisson = hw, N, poisson
        self.hform, self.sform, self.tmax = hform, sform, tmax

    def n_exc(self, rows):
        if self.N_spec == "reference":
            return 3 * LGPS_REF["formula_atoms"] * LGPS_REF["Z"]
        return 6 * rows.atoms.values

    def ea(self, rows, B0=None):
        b0, p0, ea0 = self.ea_par
        b0 = b0 if B0 is None else B0
        e, V0 = rows.eps.values, rows.V0.values
        return ea0 + G * (lattice_energy(self.hform, b0, V0, e) - p0 * V0 * e)

    def ln_sigma0(self, rows):
        return self.C + rows["Smix/kB"].values + excitation_entropy(self.sform, self.n0, self.gamma, self.n_exc(rows),
                                                                    rows.eps.values)

    def fit(self, d):
        c = arrhenius_table(d if self.tmax is None else d[d["T"] <= self.tmax])
        best = None
        for z0 in ((10.0, 0.5, 0.3), (30.0, 1.5, 0.5), (60.0, 3.0, 0.8)):
            r = least_squares(lambda z: z[2] + G * (lattice_energy(self.hform, z[0], c.V0.values, c.eps.values)
                                                    - z[1] * c.V0.values * c.eps.values) - c.Ea.values,
                              z0, bounds=([0, 0, 0], [100, 100, 3]))
            if best is None or r.cost < best.cost:
                best = r
        self.ea_par = best.x
        self.hw = debye_energy(self.ea_par[0], self.poisson) if self.hw_spec == "debye" else self.hw_spec
        self.n0 = self.ea_par[2] / self.hw
        N = self.n_exc(c)
        self.L = np.log(N / self.n0)
        best = None
        for g0 in (0.5, 1.5, 3.0, 8.0):
            r = least_squares(lambda z: z[0] + c["Smix/kB"].values
                              + excitation_entropy(self.sform, self.n0, z[1], N, c.eps.values) - c.A.values,
                              (0.0, g0), bounds=([-np.inf, 0.0], [np.inf, 100.0]))
            if best is None or r.cost < best.cost:
                best = r
        self.C, self.gamma = best.x
        return self

    def predict(self, rows, B0=None):
        return self.ln_sigma0(rows) - self.ea(rows, B0) / (KB * rows["T"].values)

    def params(self):
        B0, P0, Ea0 = self.ea_par
        return {"B0 (GPa)": B0, "P0 (GPa)": P0, "Ea0 (eV)": Ea0, "eps* (%)": 100 * P0 / B0, "hw0 (meV)": 1000 * self.hw,
                "gamma": self.gamma, "C": self.C, "n0": self.n0, "ln(N/n0)": float(np.mean(self.L))}


def sigma_mScm(ln_st, T=T25):
    return np.exp(ln_st) / T * 1000


def effective_b0(model, rows, sigma_exp, k):
    """B0 for which the prediction of row k equals sigma_exp[k]; all other parameters fixed."""
    return brentq(lambda b: np.log10(sigma_mScm(model.predict(rows, B0=b)[k])) - np.log10(sigma_exp[k]), 0.0, 100.0)


def thio_terms(d):
    e, ln1, v0t = d.eps.values, np.log1p(d.eps.values), d.V0.values / d["T"].values
    return pd.DataFrame({"eps": e, "eps2*V0/T": e ** 2 * v0t, "ln(1+eps)": ln1, "eps*ln(1+eps)": e * ln1, "eps2": e ** 2,
                         "eps2*ln(1+eps)": e ** 2 * ln1, "eps*V0/T": e * v0t, "Smix/kB": d["Smix/kB"].values})


class ThioSigmaPredictor:
    def fit(self, d):
        self.m = LinearRegression().fit(thio_terms(d).values, d.target.values)
        return self

    def predict(self, d):
        return self.m.predict(thio_terms(d).values)
