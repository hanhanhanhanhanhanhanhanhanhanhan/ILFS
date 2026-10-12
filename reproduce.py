import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

from ilfs import benchmarks as B
from ilfs import data, eps_predictor, structure
from ilfs.constants import DFT_B, DFT_G, EPS_INPUTS, G, KB, LGSSPS, LGSSPSBRO, T25
from ilfs.metrics import pairs, ranking, regression
from ilfs.sigma_predictor import SigmaPredictor, ThioSigmaPredictor, arrhenius_table, debye_energy, effective_b0, \
    poisson_ratio, sigma_mScm

warnings.simplefilter("ignore")
OUT = Path(__file__).resolve().parent / "results"
SI_LINE = [0] + list(range(1, 11))          # Si parent and Si1-xGex
SN_LINE = [0] + list(range(11, 17))         # Si parent and Si1-ySny
OTHERS = list(range(1, 17))


def load():
    s1 = data.data_s1()
    em = eps_predictor.fit(s1)
    lg = data.records("LGPS-type")
    th = data.records("thio-LISICON")
    lgssps = data.external("LGSSPS")
    bro = data.external("LGSSPSBrO")
    return s1, em, lg, th, lgssps, bro


def tuned_models(lg):
    path = OUT / "ml_models.joblib"
    if path.exists():
        return joblib.load(path)
    X, y, g = B.ml_features(lg), lg.target.values, lg.comp.values
    models = {name: B.tune(name, X, y, g) for name in B.searches()}
    joblib.dump(models, path)
    return models


def eps_results(s1, em, lg, th, lgssps, bro):
    perf = eps_predictor.performance(s1)
    eps_p = eps_predictor.predict(em, "LGPS-type", lgssps)
    t3 = LGSSPS[["composition"]].assign(eps_exp_pct=100 * LGSSPS.eps_exp, eps_p_pct=100 * eps_p)
    t3["relative error (%)"] = 100 * (t3.eps_p_pct - t3.eps_exp_pct).abs() / t3.eps_exp_pct
    stats = pd.DataFrame([{"set": "LGSSPS (Table S3)", **regression(LGSSPS.eps_exp, eps_p)}])
    ranges = pd.DataFrame([{"set": k, "eps_p min (%)": 100 * v.min(), "eps_p max (%)": 100 * v.max()} for k, v in (
        ("LGPS-type records", lg.eps), ("thio-LISICON records", th.eps), ("LGSSPS", lgssps.eps), ("LGSSPSBrO", bro.eps))])
    return {"eps_predictor": perf, "table_s3": t3, "table_s3_stats": stats, "eps_ranges": ranges}


def sigma_results(s1, em, lg, th, lgssps, bro):
    m = SigmaPredictor().fit(lg)
    folds = []
    p_loco = np.full(len(lg), np.nan)
    for tr, te in LeaveOneGroupOut().split(lg, groups=lg.comp):
        mt = SigmaPredictor().fit(lg.iloc[tr].reset_index(drop=True))
        p_loco[te] = mt.predict(lg.iloc[te])
        folds.append(mt.params())
    F = pd.DataFrame(folds)
    p = m.params()
    t1 = pd.DataFrame({"parameter": list(p), "value": list(p.values()),
                       "LOCO min": F.min().values, "LOCO max": F.max().values})
    t1 = pd.concat([t1, pd.DataFrame([
        {"parameter": "N", "value": 3 * 25 * 2},
        {"parameter": "Ea minimum (eV)", "value": p["Ea0 (eV)"] - G * p["P0 (GPa)"] ** 2 * 903.25 / (2 * p["B0 (GPa)"])},
        {"parameter": "hw0 from DFT moduli (meV)", "value": 1000 * debye_energy(DFT_B, poisson_ratio(DFT_B, DFT_G))},
        {"parameter": "compositions in the fit", "value": len(arrhenius_table(lg))},
        {"parameter": "mean dS_mix/kB of the fitted compositions", "value": arrhenius_table(lg)["Smix/kB"].mean()}])])
    thio_in = ThioSigmaPredictor().fit(th).predict(th)
    thio_loco = B.loco_predict(ThioSigmaPredictor, th)
    perf = pd.DataFrame([
        {"family": "LGPS-type", "records": len(lg), "compositions": lg.comp.nunique(),
         "in-sample R2": r2_score(lg.target, m.predict(lg)), "LOCO R2": r2_score(lg.target, p_loco),
         "LOCO MAE": mean_absolute_error(lg.target, p_loco)},
        {"family": "thio-LISICON", "records": len(th), "compositions": th.comp.nunique(),
         "in-sample R2": r2_score(th.target, thio_in), "LOCO R2": r2_score(th.target, thio_loco),
         "LOCO MAE": mean_absolute_error(th.target, thio_loco)}])
    return {"table1": t1, "sigma_predictor_performance": perf}


def table_s2():
    d = data.data_s2()
    f = d[d.Panels.notna()]
    rows = []
    for fam, g in [(k, f[f.family == k]) for k in ("LGPS-type", "thio-LISICON")] + [("Pooled", f)]:
        for prop, col in (("ln(sigma T)", "target"), ("Ea", "Ea"), ("ln sigma_o", "ln_sigma_o")):
            s = g.dropna(subset=[col])
            r, pr = pearsonr(s.eps, s[col])
            rho, ps = spearmanr(s.eps, s[col])
            rows.append({"family": fam, "property": prop, "n": len(s), "Pearson r": r, "P": pr,
                         "Spearman rho": rho, "P ": ps})
    return {"table_s2": pd.DataFrame(rows)}


def table_s4(s1, em, lg, th, lgssps, bro):
    m = SigmaPredictor().fit(lg)
    sp = sigma_mScm(m.predict(lgssps))
    pel = LGSSPS[["sigma_pellet1", "sigma_pellet2"]].values
    rho = 100 * LGSSPS[["density_pellet1", "density_pellet2"]].values / LGSSPS[["density_theoretical"]].values
    exp = pel.mean(axis=1)
    t = pd.DataFrame({"composition": LGSSPS.composition, "relative density mean (%)": rho.mean(axis=1),
                      "relative density sd (%)": rho.std(axis=1, ddof=1), "sigma exp mean (mS/cm)": exp,
                      "sigma exp sd (mS/cm)": pel.std(axis=1, ddof=1), "sigma pred (mS/cm)": sp,
                      "dlog10 sigma": np.log10(sp) - np.log10(exp)})
    st = pd.DataFrame([{**ranking(sp, exp), "factor": 10 ** np.mean(np.log10(sp / exp))}])
    return {"table_s4": t, "table_s4_stats": st}


def table_s5(s1, em, lg, th, lgssps, bro):
    m = SigmaPredictor().fit(lg)
    exp = LGSSPSBRO.sigma_exp.values
    base = sigma_mScm(m.predict(bro))
    b0 = effective_b0(m, bro, exp, 0)
    cal = sigma_mScm(m.predict(bro, B0=b0))
    shift = base * exp[0] / base[0]
    mlp = sigma_mScm(tuned_models(lg)["MLP"].predict(B.ml_features(bro)))
    mlp_shift = mlp * exp[0] / mlp[0]
    t = LGSSPSBRO[["label"]].assign(eps_p=bro.eps, sigma_exp=exp, sigma_pred=base,
                                    dlog10=np.log10(base / exp), sigma_calibrated=cal,
                                    dlog10_calibrated=np.log10(cal / exp), sigma_MLP=mlp,
                                    dlog10_MLP=np.log10(mlp / exp))
    o = OTHERS

    def slope(pred):
        return np.polyfit(100 * bro.eps.values[o], np.log10(pred[o] / exp[o]), 1)[0]

    st = pd.DataFrame([
        {"model": "sigma-predictor", **ranking(base, exp)},
        {"model": "sigma-predictor, 16 non-anchor", **ranking(base[o], exp[o]), "slope per pp eps_p": slope(base)},
        {"model": f"sigma-predictor, B0' = {b0:.2f} GPa (Si anchor)", **ranking(cal[o], exp[o]),
         "slope per pp eps_p": slope(cal)},
        {"model": "sigma-predictor, uniform shift (Si anchor)", **ranking(shift[o], exp[o]),
         "slope per pp eps_p": slope(shift)},
        {"model": "MLP", **ranking(mlp, exp)},
        {"model": "MLP, uniform shift (Si anchor)", **ranking(mlp_shift[o], exp[o]),
         "slope per pp eps_p": slope(mlp_shift)}])
    return {"table_s5": t, "table_s5_stats": st}


def lgssps_bro_entropy(s1, em, lg, th, lgssps, bro):
    m = SigmaPredictor().fit(lg)
    exp = LGSSPSBRO.sigma_exp.values
    s_mix = bro["Smix/kB"].values
    s_m = m.ln_sigma0(bro) - m.C - s_mix
    rows = []
    for x, v in (("dS_mix", s_mix), ("eps_p", bro.eps.values)):
        for line, idx in (("Si-Ge", SI_LINE), ("Si-Sn", SN_LINE), ("all 17", list(range(17)))):
            rows.append({"x": x, "line": line, "Spearman rho (sigma_exp)": spearmanr(v[idx], exp[idx])[0]})
    ratio = pd.DataFrame([{"dS_M/dS_mix min": (s_m / s_mix).min(), "dS_M/dS_mix max": (s_m / s_mix).max(),
                           "dS_mix share min (%)": 100 * (s_mix / (s_mix + s_m)).min(),
                           "dS_mix share max (%)": 100 * (s_mix / (s_mix + s_m)).max()}])
    return {"lgssps_bro_spearman": pd.DataFrame(rows), "lgssps_bro_entropy": ratio}


def table_s7(s1, em, lg, th, lgssps, bro):
    y = lg.target.values
    exp = LGSSPSBRO.sigma_exp.values
    makers = {"sigma-predictor": SigmaPredictor, "Arrhenius only": lambda: B.OLS(B.inv_t),
              "Linear regression": lambda: B.OLS(B.inputs_inv_t)}
    pred = {k: B.loco_predict(mk, lg) for k, mk in makers.items()}
    full = {k: mk().fit(lg) for k, mk in makers.items()}
    bro_pred = {k: v.predict(bro) for k, v in full.items()}
    bro_pred["Arrhenius only"] = np.zeros(len(bro))
    cm = {k: B.composition_mae(p, lg) for k, p in pred.items()}
    n_rec = lg.groupby("comp").size().reindex(cm["sigma-predictor"].index).values.astype(float)
    rng = np.random.default_rng(7)
    ci, ci_rw = {}, {}
    for k in ("Arrhenius only", "Linear regression"):
        dm = (cm[k] - cm["sigma-predictor"]).values
        ci[k] = (dm.mean(), *B.bootstrap_ci(dm, rng, 10000))
        ci_rw[k] = (np.average(dm, weights=n_rec), *B.bootstrap_ci(dm, rng, 10000, weights=n_rec))
    X, g = B.ml_features(lg), lg.comp.values
    models = tuned_models(lg)
    for name, m in models.items():
        p = B.ml_loco(m.best_estimator_, X, y, g)
        pred[name] = p
        dm = (B.composition_mae(p, lg) - cm["sigma-predictor"]).values
        ci[name] = (dm.mean(), *B.bootstrap_ci(dm, np.random.default_rng(2026), 5000))
        bro_pred[name] = m.predict(B.ml_features(bro))
    rows = []
    for k, p in pred.items():
        lo, rw = ci.get(k, (np.nan,) * 3), ci_rw.get(k, (np.nan,) * 3)
        rows.append({"model": k, "LOCO R2": r2_score(y, p), "LOCO MAE": mean_absolute_error(y, p),
                     "dMAE": lo[0], "dMAE 95% CI low": lo[1], "dMAE 95% CI high": lo[2],
                     "dMAE (record-weighted)": rw[0], "95% CI low (rw)": rw[1], "95% CI high (rw)": rw[2],
                     "LGSSPSBrO pairs (of 136)": pairs(bro_pred[k], exp)})
    params = pd.DataFrame([{"model": k, "hyperparameters": str(m.best_params_)} for k, m in models.items()])
    return {"table_s7": pd.DataFrame(rows), "table_s7_hyperparameters": params}


def table_s8(s1, em, lg, th, lgssps, bro):
    hw_dft = debye_energy(DFT_B, poisson_ratio(DFT_B, DFT_G))
    variants = {"Default": {}, "Poisson ratio 0.35": {"poisson": 0.35}, "hw0 from DFT moduli": {"hw": hw_dft},
                "Birch-Murnaghan enthalpy": {"hform": "Birch-Murnaghan"}, "Exact entropy": {"sform": "exact"},
                "N per composition": {"N": "per-composition"}, "Arrhenius fits at T <= 400 K": {"tmax": 400.0}}
    exp = LGSSPSBRO.sigma_exp.values
    rows = []
    for name, kw in variants.items():
        m = SigmaPredictor(**kw).fit(lg)
        p = B.loco_predict(lambda: SigmaPredictor(**kw), lg)
        rows.append({"variant": name, "hw0 (meV)": 1000 * m.hw, "B0 (GPa)": m.ea_par[0], "gamma": m.gamma,
                     "LOCO R2": r2_score(lg.target, p), "LOCO MAE": mean_absolute_error(lg.target, p),
                     "LGSSPSBrO pairs (of 136)": pairs(m.predict(bro), exp)})
    return {"table_s8": pd.DataFrame(rows)}


def table_s9(s1, em, lg, th, lgssps, bro):
    rows = []
    for c, g in lg.groupby("comp"):
        idx = g.index.values
        for b in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            s = lg.loc[b]
            if s["T"].nunique() >= 3 and s["T"].min() <= 310 and s["T"].max() >= 290:
                k, a = np.polyfit(1 / s["T"], s.target, 1)
                rows.append({"comp": c, "composition": data.formula(s.iloc[0]),
                             "sigma25": sigma_mScm(a + k / T25), "Ea": -k * KB})
    S = pd.DataFrame(rows).groupby("comp").filter(lambda g: len(g) >= 2)
    S["log10 sigma25"] = np.log10(S.sigma25)
    t = S.groupby("composition").agg(series=("sigma25", "size"), sigma_min=("sigma25", "min"),
                                     sigma_max=("sigma25", "max"), sd_log10=("log10 sigma25", lambda v: v.std(ddof=1)),
                                     Ea_min=("Ea", "min"), Ea_max=("Ea", "max")).reset_index()
    t["max/min"] = t.sigma_max / t.sigma_min
    dof = S.groupby("comp").size() - 1
    pooled = np.sqrt((S.groupby("comp")["log10 sigma25"].var(ddof=1) * dof).sum() / dof.sum())
    t = pd.concat([t, pd.DataFrame([{"composition": "pooled within-composition SD", "sd_log10": pooled,
                                     "max/min": 10 ** pooled}])])
    return {"table_s9": t}


def pooled_families(s1, em, lg, th, lgssps, bro):
    lg2, th2 = lg.assign(comp="LGPS:" + lg.comp), th.assign(comp="thio:" + th.comp)
    d = pd.concat([lg2, th2], ignore_index=True)
    inputs = {"LGPS-type": EPS_INPUTS["LGPS-type"], "thio-LISICON": EPS_INPUTS["thio-LISICON"]}
    models = {"sigma-predictor": lambda f: SigmaPredictor() if f == "LGPS-type" else ThioSigmaPredictor(),
              "Arrhenius, per family": lambda f: B.OLS(B.inv_t),
              "Linear regression, per family": lambda f: B.OLS(lambda r, c=inputs[f]: np.column_stack(
                  [r[c].values, 1 / r["T"].values]))}
    pred = {k: np.full(len(d), np.nan) for k in models}
    for c in d.comp.unique():
        te = (d.comp == c).values
        f = d.family[te].iloc[0]
        tr = d[~te & (d.family == f).values].reset_index(drop=True)
        for k, mk in models.items():
            pred[k][te] = mk(f).fit(tr).predict(d[te])
    return {"pooled_families": pd.DataFrame([{"model": k, "records": len(d), "compositions": d.comp.nunique(),
                                              "LOCO R2": r2_score(d.target, p)} for k, p in pred.items()])}


def calibration(s1, em, lg, th, lgssps, bro):
    m = SigmaPredictor().fit(lg)
    exp = LGSSPSBRO.sigma_exp.values
    X0, y0, g0 = B.ml_features(lg), lg.target.values, lg.comp.values
    Xb = B.ml_features(bro)
    rows = []
    for k in range(len(exp)):
        o = [i for i in range(len(exp)) if i != k]
        b0 = effective_b0(m, bro, exp, k)
        cal = sigma_mScm(m.predict(bro, B0=b0))
        mlp = B.tune("MLP", np.vstack([X0, Xb[[k]]]), np.append(y0, np.log(exp[k] / 1000 * T25)),
                     np.append(g0, f"LGSSPSBrO {k}"))
        p = sigma_mScm(mlp.predict(Xb))
        rows.append({"single sample": LGSSPSBRO.label[k], "B0' (GPa)": b0,
                     "sigma-predictor pairs (of 120)": pairs(cal[o], exp[o]), "MLP + sample pairs (of 120)": pairs(p[o], exp[o])})
    t = pd.DataFrame(rows)
    summary = t.drop(columns="single sample").agg(["median", "min", "max"])
    return {"calibration_all_anchors": t, "calibration_summary": summary.reset_index()}


def structures(s1, *_):
    return {"table_s6": structure.table_s6(s1), "fig_s11b_reference_volumes": structure.reference_volumes(s1),
            "li_sites": structure.li_site_correlations(s1, data.li_sites())}


TASKS = {"eps": eps_results, "sigma": sigma_results, "table_s2": lambda *a: table_s2(), "table_s4": table_s4,
         "table_s5": table_s5, "entropy": lgssps_bro_entropy, "table_s6": structures, "table_s7": table_s7,
         "table_s8": table_s8, "table_s9": table_s9, "pooled": pooled_families, "calibration": calibration}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", help=", ".join(TASKS))
    args = ap.parse_args()
    unknown = set(args.tasks) - set(TASKS)
    if unknown:
        ap.error(f"unknown task(s): {', '.join(sorted(unknown))}")
    OUT.mkdir(exist_ok=True)
    inputs = load()
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 60)
    for name in args.tasks or list(TASKS):
        for key, df in TASKS[name](*inputs).items():
            df.to_csv(OUT / f"{key}.csv", index=False)
            print(f"\n== {key}\n{df.to_string(index=False, float_format=lambda v: f'{v:.4g}')}", flush=True)


if __name__ == "__main__":
    main()
