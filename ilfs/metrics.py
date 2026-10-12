from itertools import combinations

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def regression(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"R2": r2_score(y, p), "RMSE": mean_squared_error(y, p) ** 0.5, "MAE": mean_absolute_error(y, p),
            "MAPE": mean_absolute_percentage_error(y, p)}


def pairs(pred, exp):
    """Concordant pairwise orderings; tied predictions count one half."""
    s = np.array([np.sign(pred[i] - pred[j]) * np.sign(exp[i] - exp[j]) for i, j in combinations(range(len(exp)), 2)])
    return float((s > 0).sum() + 0.5 * (s == 0).sum())


def ranking(pred, exp):
    """pred, exp: conductivities."""
    pred, exp = np.asarray(pred, float), np.asarray(exp, float)
    dl = np.log10(pred) - np.log10(exp)
    return {"n": len(exp), "Pearson r": pearsonr(pred, exp)[0], "Spearman rho": spearmanr(pred, exp)[0],
            "Kendall tau": kendalltau(pred, exp)[0], "Pairs": pairs(pred, exp), "of": len(exp) * (len(exp) - 1) // 2,
            "mean dlog10": dl.mean(), "MAE log10": np.abs(dl).mean(), "RMSE log10": np.sqrt(np.mean(dl ** 2)),
            "within 2x": int((np.abs(dl) <= np.log10(2)).sum())}
