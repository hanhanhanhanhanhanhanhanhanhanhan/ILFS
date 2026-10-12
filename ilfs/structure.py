import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

V_ANION = "V_distorted/anion (Å³)"
DESCRIPTORS = {"Percolation radius (Å)": "Percolation radius", "Maximum Li-site size (Å)": "Maximum Li-site size",
               "d3D (Å)": "d3D", "Baur distortion index D (XY4)": "Baur distortion index, D",
               "Bond-angle variance σ² (XY4, deg²)": "Bond-angle variance, σ2"}


def fixed_anion(s1):
    return s1[s1["Fixed-anion structure"] == "Yes"].reset_index(drop=True)


def loco_ols(g, x, y):
    p = np.full(len(g), np.nan)
    for c in g.comp.unique():
        te = (g.comp == c).values
        p[te] = LinearRegression().fit(g.loc[~te, [x]], g.loc[~te, y]).predict(g.loc[te, [x]])
    return p


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def table_s6(s1, n_boot=5000, seed=3):
    f = fixed_anion(s1)
    fam = {k: g for k, g in f.groupby("family")}
    rows = [{"descriptor": "Refined lattice volume, V/anion", "n": len(f),
             **{f"Spearman rho, {k}": spearmanr(g.ILFS, g[V_ANION])[0] for k, g in fam.items()}}]
    rng = np.random.default_rng(seed)
    for col, name in DESCRIPTORS.items():
        g = f.dropna(subset=[col]).reset_index(drop=True)
        y = g[col].values
        p_e, p_v = loco_ols(g, "ILFS", col), loco_ols(g, V_ANION, col)
        comps = g.comp.unique()
        idx = {c: np.flatnonzero(g.comp.values == c) for c in comps}
        d = []
        for _ in range(n_boot):
            ii = np.concatenate([idx[c] for c in rng.choice(comps, len(comps))])
            d.append(r2(y[ii], p_e[ii]) - r2(y[ii], p_v[ii]))
        rows.append({"descriptor": name, "n": len(g),
                     **{f"Spearman rho, {k}": spearmanr(*h.dropna(subset=[col])[["ILFS", col]].T.values)[0]
                        for k, h in fam.items()},
                     "LOCO R2, ILFS": r2(y, p_e), "LOCO R2, V/anion": r2(y, p_v),
                     "dR2 95% CI low": np.quantile(d, 0.025), "dR2 95% CI high": np.quantile(d, 0.975),
                     "compositions": len(comps)})
    return pd.DataFrame(rows)


def reference_volumes(s1):
    """Fig. S11b: V_perfect per anion by reference group."""
    g = s1.groupby(["family", "Anion reference group"])["V_perfect/anion (Å³)"]
    return g.agg(["size", "min", "max"]).reset_index()


def li_site_correlations(s1, sites):
    f = fixed_anion(s1)
    L = sites[sites.Entry.isin(f.Entry)].copy()
    L["refined"] = L["Occupancy refined"].eq("Yes")
    li = L["Wyckoff position"].str.extract(r"(\d+)")[0].astype(int) * L.Occupancy
    L["share"] = li / li.groupby(L.Entry).transform("sum")
    a = L.groupby(["Entry", "Structural family", "Wyckoff position"]).agg(
        share=("share", "sum"), occupancy=("Occupancy", "mean"),
        refined=("refined", "all")).reset_index().merge(f[["Entry", "ILFS"]], on="Entry")
    rows = []
    for (fam, w), g in a.groupby(["Structural family", "Wyckoff position"]):
        for label, s in (("all", g), ("refined only", g[g.refined])):
            if len(s) >= 6 and s.share.nunique() > 2:
                rows.append({"family": fam, "Li site": w, "subset": label, "n": len(s),
                             "rho(ILFS, Li fraction on site)": spearmanr(s.ILFS, s.share)[0],
                             "rho(ILFS, occupancy)": spearmanr(s.ILFS, s.occupancy)[0]})
    return pd.DataFrame(rows)
