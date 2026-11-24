"""Mono-backend oracles: how far can a per-image global map of raw depth get?

All fits are evaluator-only and reuse cached predictions + GT; no VLM calls.
Hypotheses tested per image (AbsRel-optimal parameters, positive-depth constraint):
  scale        z = a * d
  affine       z = a * d + b
  inverse      z = 1 / (a * d + b)   (checks whether raw output is inverse-depth/disparity)
A monotone piecewise-linear-in-log map is available for the case where affine is poor.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar


def _pix(pred, gt, mask, max_n, seed):
    m = mask & np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    p, g = pred[m].astype(np.float64), gt[m].astype(np.float64)
    if max_n is not None and max_n > 0 and p.size > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(p.size, max_n, replace=False)
        p, g = p[idx], g[idx]
    return p, g


def _absrel(z, g):
    return float(np.mean(np.abs(z - g) / g))


def fit_scale(pred, gt, mask, max_n=None, seed=0):
    p, g = _pix(pred, gt, mask, max_n, seed)
    if p.size == 0:
        return {"kind": "scale", "ok": False, "a": 1.0, "b": 0.0, "abs_rel": float("nan")}
    r = minimize_scalar(lambda a: _absrel(a * p, g), bounds=(1e-4, 1e4), method="bounded")
    return {"kind": "scale", "ok": bool(r.success), "a": float(r.x), "b": 0.0,
            "abs_rel": float(r.fun), "clip_frac": 0.0}


def fit_affine(pred, gt, mask, max_n=None, seed=0, med=None):
    p, g = _pix(pred, gt, mask, max_n, seed)
    if p.size == 0:
        return {"kind": "affine", "ok": False, "a": 1.0, "b": 0.0, "abs_rel": float("nan")}
    med = float(np.median(p)) if med is None else med

    def loss(t):
        a = float(np.exp(t[0])); b = float(t[1] * med)
        z = a * p + b
        if np.any(z <= 0):
            return 10.0 + float(np.mean(np.maximum(0.0, -z) / max(med, 1e-9)))
        return _absrel(z, g)

    best = None
    for la in (np.log(0.2), 0.0, np.log(5.0)):
        for bh in (-1.0, 0.0, 1.0):
            r = minimize(loss, np.array([la, bh]), method="L-BFGS-B",
                         bounds=[(np.log(1e-4), np.log(1e4)), (-20, 20)],
                         options={"maxiter": 200})
            if best is None or r.fun < best.fun:
                best = r
    a = float(np.exp(best.x[0])); b = float(best.x[1] * med)
    return {"kind": "affine", "ok": bool(best.success), "a": a, "b": b,
            "abs_rel": float(best.fun), "clip_frac": 0.0}


def fit_inverse(pred, gt, mask, max_n=None, seed=0, med=None):
    """z = 1/(a p + b); tests the 'raw output is inverse depth' hypothesis."""
    p, g = _pix(pred, gt, mask, max_n, seed)
    if p.size == 0:
        return {"kind": "inverse", "ok": False, "a": 1.0, "b": 0.0, "abs_rel": float("nan")}
    med = float(np.median(p)) if med is None else med

    def loss(t):
        a = float(np.exp(t[0])); b = float(t[1] * med)
        den = a * p + b
        if np.any(den <= 1e-9):
            return 10.0
        return _absrel(1.0 / den, g)

    best = None
    for la in (np.log(0.2), 0.0, np.log(5.0)):
        for bh in (-1.0, 0.0, 1.0):
            r = minimize(loss, np.array([la, bh]), method="L-BFGS-B",
                         bounds=[(np.log(1e-6), np.log(1e6)), (-20, 20)],
                         options={"maxiter": 200})
            if best is None or r.fun < best.fun:
                best = r
    a = float(np.exp(best.x[0])); b = float(best.x[1] * med)
    return {"kind": "inverse", "ok": bool(best.success), "a": a, "b": b,
            "abs_rel": float(best.fun)}


def fit_monotone_loglin(pred, gt, mask, train_frac=0.5, knots=4, max_n=None,
                        seed=0, affine_abs_rel=None):
    """Piecewise-linear map of log z vs log d with `knots` interior knots.

    Fitted on half the valid pixels (deterministic split), evaluated on the held-out
    half. Reported only as a bounded-DOF check; not a deployment method.
    """
    m = mask & np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
    p, g = pred[m].astype(np.float64), gt[m].astype(np.float64)
    if p.size < 1000:
        return {"kind": "monotone_loglin", "ok": False, "abs_rel_holdout": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(p.size)
    n = p.size if (max_n is None or max_n <= 0) else min(p.size, max_n)
    idx = idx[:n]
    split = n // 2
    tr, ho = idx[:split], idx[split:]
    lp = np.log(p); lg = np.log(g)
    qs = np.linspace(0, 1, knots + 2)
    knot_x = np.quantile(lp[tr], qs)

    def design(lp_vals):
        X = np.zeros((lp_vals.size, knot_x.size))
        for j in range(knot_x.size):
            X[:, j] = np.clip(lp_vals - knot_x[j], 0.0, None)
        return X

    def fit_eval(train_ids, eval_ids):
        X = design(lp[train_ids])
        Xe = design(lp[eval_ids])

        def f(w):
            s = np.log1p(np.exp(-np.abs(w))) + np.maximum(w, 0.0)   # softplus, >=0
            c = lg[train_ids].mean() - X.mean(0) @ s
            pred_log = X @ s + c
            return float(np.mean(np.abs(np.exp(pred_log) - g[train_ids]) / g[train_ids]))

        w0 = np.zeros(knot_x.size)
        r = minimize(f, w0, method="L-BFGS-B")
        s = np.log1p(np.exp(-np.abs(r.x))) + np.maximum(r.x, 0.0)
        c = lg[train_ids].mean() - X.mean(0) @ s
        pred_log = Xe @ s + c
        hold = float(np.mean(np.abs(np.exp(pred_log) - g[eval_ids]) / g[eval_ids]))
        return hold

    hold = fit_eval(tr, ho)
    rev = fit_eval(ho, tr)
    return {"kind": "monotone_loglin", "ok": True, "knots": knots,
            "abs_rel_holdout": float((hold + rev) / 2.0)}


def mono_oracle_report(pred, gt, mask, seed=0, run_monotone=False):
    med = float(np.median(pred[np.isfinite(pred) & (pred > 0)])) if np.any(pred > 0) else 1.0
    out = {
        "scale": fit_scale(pred, gt, mask, seed=seed),
        "affine": fit_affine(pred, gt, mask, seed=seed, med=med),
        "inverse": fit_inverse(pred, gt, mask, seed=seed, med=med),
    }
    if run_monotone:
        out["monotone_loglin"] = fit_monotone_loglin(pred, gt, mask, seed=seed)
    # clipping diagnostics for the affine map over *all* finite pixels
    a, b = out["affine"]["a"], out["affine"]["b"]
    fin = np.isfinite(pred)
    z = a * pred[fin] + b
    out["affine_clip_frac_all"] = float(np.mean(z <= 0))
    return out
