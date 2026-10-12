from pathlib import Path

import numpy as np
import pandas as pd

from .features import anion_radius, v_perfect

DATA = Path(__file__).resolve().parents[1] / "data"
S1_ELEMENTS = ["Li", "Ge", "Si", "Sn", "P", "Sb", "O", "Se", "S", "Cl", "Br"]
S2_ELEMENTS = ["Li", "Ge", "Si", "Sn", "P", "Sb", "Al", "Zn", "Ga", "S", "Se"]
S1_INPUTS = {"ΔX^2 (M)": "dX2_M", "ΔX^2 (P)": "dX2_P", "TΔS": "TdS_mix", "Δr^2(M)": "dr2_M", "Δr^2(P)": "dr2_P",
             "U(cation)": "U_cation", "U(anion)": "U_anion", "Mixing enthalpy": "dH_mix"}
EXT_ELEMENTS = ["Li", "Ge", "Si", "Sn", "P", "S", "O", "Br"]
S2_COLUMNS = {"Structural family": "family", "T (K)": "T", "ln(σT)": "target", "ε_p": "eps", "V_perfect (Å³)": "V0",
              "ΔS_mix/k_B": "Smix/kB", "ΔX²(M)": "dX2_M", "ΔX²(P)": "dX2_P", "TΔS_mix (kJ mol⁻¹)": "TdS_mix",
              "Δr²(M) (Å²)": "dr2_M", "Δr²(P) (Å²)": "dr2_P", "U(cation)": "U_cation", "U(anion)": "U_anion",
              "ΔH_mix (kJ mol⁻¹)": "dH_mix", "E_a (eV)": "Ea", "ln σ_o": "ln_sigma_o"}


def comp_key(df, elements):
    return df[elements].astype(float).fillna(0).round(3).astype(str).agg("-".join, axis=1)


def data_s1(path=DATA / "Data S1.xlsx"):
    d = pd.read_excel(path, sheet_name="ILFS dataset")
    d.columns = [c.strip().replace(" )", ")") for c in d.columns]
    d = d.rename(columns={**S1_INPUTS, "Structural family": "family", "Temperature": "T"})
    for c in S1_INPUTS.values():
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["V_perfect"] = [v_perfect(f, anion_radius(r)) for f, r in zip(d.family, d[S1_ELEMENTS].to_dict("records"))]
    d["ILFS"] = d["Cell volume"] / d.V_perfect - 1
    d["comp"] = comp_key(d, S1_ELEMENTS)
    return d


def li_sites(path=DATA / "Data S1.xlsx"):
    return pd.read_excel(path, sheet_name="Li sites")


def data_s2(path=DATA / "Data S2.xlsx"):
    d = pd.read_excel(path, sheet_name="SSE dataset", header=1)
    d.columns = [str(c).strip() for c in d.columns]
    d = d.rename(columns=S2_COLUMNS)
    d["comp"] = comp_key(d, S2_ELEMENTS)
    d["atoms"] = d[S2_ELEMENTS].astype(float).fillna(0).sum(axis=1)
    return d


def external(series, path=DATA / "Data S2.xlsx"):
    """LGSSPS or LGSSPSBrO at 25 C."""
    d = pd.read_excel(path, sheet_name="LGSSPS and LGSSPSBrO").rename(columns=S2_COLUMNS)
    d = d[d.Series == series].reset_index(drop=True)
    d["atoms"] = d[EXT_ELEMENTS].sum(axis=1)
    return d


def records(family, path=DATA / "Data S2.xlsx"):
    d = data_s2(path)
    return d[d.family == family].reset_index(drop=True)


def formula(row, elements=("Li", "Ge", "Si", "Sn", "P", "Sb", "S", "Se")):
    return "".join(el + ("" if row[el] == 1 else f"{row[el]:g}") for el in elements if row[el] and np.isfinite(row[el]))
