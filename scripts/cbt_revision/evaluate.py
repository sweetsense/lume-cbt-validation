#!/usr/bin/env python3
"""
CBT revision, step 3: evaluation harness.

Rung 0 reproduces the submitted paper's numbers from its own paired table, using a port
of scripts/cbt-model-eval.js (ridge Tobit by EM, per-sensor fixed effects and mon2
slopes, leave-one-out, Youden cutpoints, class-weighted logistic). Matching
paper_data.json is the check that the harness is right before anything is changed.

Usage: python3 scripts/cbt_revision/evaluate.py
"""
import json
import math
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# macOS Accelerate BLAS raises spurious matmul warnings on finite inputs; results are checked
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(all='ignore')
REV = Path(__file__).resolve().parents[2] / 'data' / 'cbt_revision'

_SUB = Path(__file__).resolve().parents[2] / 'data' / 'cbt_revision' / 'submitted'   # the submitted paper's files
_LOCAL_PAPER = Path('/Users/ethomas/Dropbox/Claude/overleaf-6974e5991c24cbfce9bb651a')
PAPER_DIR = Path(os.environ.get('CBT_PAPER_DIR', _LOCAL_PAPER if _LOCAL_PAPER.exists() else Path(__file__).resolve().parents[2] / 'paper'))
OVERLEAF = _SUB   # rungs 0-1 and the IA comparison read the submitted paper's files
CBT_DL = 100
DL_LOG = math.log10(CBT_DL + 1)
CBT_SE = 0.65 / 1.96
DIFF_CI_HALF = 1.96 * math.sqrt(2 * CBT_SE ** 2)
CENS_MIN_YHAT = DL_LOG - DIFF_CI_HALF
REFERENCE = '50065'


# ---- numerics, matching cbt-model-eval.js -------------------------------------------
def erf(x):
    t = 1 / (1 + 0.3275911 * abs(x))
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * math.exp(-x * x)
    return y if x >= 0 else -y


norm_cdf = np.vectorize(lambda x: 0.5 * (1 + erf(x / math.sqrt(2))))
norm_pdf = lambda x: np.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def ols_beta(X, y, lam):
    XtX = X.T @ X
    XtX[1:, 1:] += lam * np.eye(X.shape[1] - 1)
    return np.linalg.solve(XtX, X.T @ y)


def fit_tobit(X, y, cens, lam, dl=DL_LOG):
    b = ols_beta(X, y, lam)
    sig = math.sqrt(np.mean((y - X @ b) ** 2)) or 0.5
    for _ in range(200):
        mu = X @ b
        al = (dl - mu) / sig
        z = np.where(cens, mu + sig * norm_pdf(al) / np.maximum(1e-9, 1 - norm_cdf(al)), y)
        nb = ols_beta(X, z, lam)
        m = X @ nb
        al2 = (dl - m) / sig
        sse = np.where(cens, sig * sig * (1 + al2 * norm_pdf(al2) / np.maximum(1e-9, 1 - norm_cdf(al2))), (y - m) ** 2).sum()
        nsig = math.sqrt(sse / len(y)) or sig
        d = max(np.max(np.abs(nb - b)), abs(nsig - sig))
        b, sig = nb, nsig
        if d < 1e-7:
            break
    return b, sig


def zstat(v):
    v = np.asarray(v, float)
    sd = v.std() or 1.0
    return v.mean(), sd


def youden(scores, ybin):
    """Threshold sweep exactly as the JS: 0, each unique score, and midpoints; first max J."""
    u = np.unique(scores)
    cuts = [0.0]
    for i, s in enumerate(u):
        cuts.append(s)
        if i < len(u) - 1:
            cuts.append((u[i] + u[i + 1]) / 2)
    npos, nneg = ybin.sum(), (1 - ybin).sum()
    best = None
    for c in cuts:
        pred = scores >= c
        tp = int((pred & (ybin == 1)).sum()); fn = int((~pred & (ybin == 1)).sum())
        fp = int((pred & (ybin == 0)).sum()); tn = int((~pred & (ybin == 0)).sum())
        sens, spec = tp / npos, tn / nneg
        J = sens + spec - 1
        if best is None or J > best['J']:
            best = dict(J=J, cut=float(c), ba=(sens + spec) / 2, sens=sens, spec=spec, tp=tp, fp=fp, tn=tn, fn=fn)
    return best


def logistic_gd(X, y, w, lr, iters, lam):
    """Two-class softmax by full-batch gradient descent, as fitMultinomialLogistic."""
    W = np.zeros((X.shape[1], 2))
    Y = np.stack([1 - y, y], axis=1)
    wsum = w.sum()
    for _ in range(iters):
        L = X @ W
        L -= L.max(axis=1, keepdims=True)
        P = np.exp(L); P /= P.sum(axis=1, keepdims=True)
        G = X.T @ ((P - Y) * w[:, None])
        W -= lr * (G / wsum + lam * W)
    return W


def logistic_prob(X, W):
    L = X @ W
    L -= L.max(axis=1, keepdims=True)
    P = np.exp(L)
    return P[:, 1] / P.sum(axis=1)


# ---- the paper's feature construction --------------------------------------------------
def paper_features(d):
    """Per-sensor baselines: median over the sensor's CBT=0 points (p10 of all if none)."""
    d = d.copy()
    for col, raw in (('mon2n', 'mon2'), ('tofn', 'tof')):
        base = {}
        for bc, g in d.groupby('bc'):
            clean = g[g.cbt == 0][raw]
            base[bc] = float(clean.median()) if len(clean) else float(np.sort(g[raw])[int(len(g) * 0.1)])
        d[col] = d[raw] - d.bc.map(base)
        d.attrs[col + '_baseline'] = base
    return d


def design(d, stats, fe_sensors, slopes=True):
    mz = (d.mon2n - stats['m'][0]) / stats['m'][1]
    tz = (d.temp - stats['t'][0]) / stats['t'][1]
    fz = (d.tofn - stats['f'][0]) / stats['f'][1]
    cols = [np.ones(len(d)), mz, tz, fz]
    cols += [(d.bc == bc).astype(float).values for bc in fe_sensors]
    if slopes:
        cols += [np.where(d.bc == bc, mz, 0.0) for bc in fe_sensors]
    return np.column_stack(cols)


def stats_of(d):
    return dict(m=zstat(d.mon2n), t=zstat(d.temp), f=zstat(d.tofn))


def rung0():
    raw = pd.read_csv(OVERLEAF / 'paired_observations.csv')
    paper = json.load(open(OVERLEAF / 'paper_data.json'))
    d = pd.DataFrame(dict(bc=raw.barcode.astype(str), cbt=raw.cbt_ecoli_cfu.astype(float),
                          cens=raw.cbt_censored.astype(str).str.upper().eq('TRUE') | (raw.cbt_ecoli_cfu >= CBT_DL),
                          mon2=raw.sensor_mon2_raw.astype(float), tof=raw.sensor_tof_raw.astype(float),
                          temp=raw.water_temp_c.astype(float)))
    d = paper_features(d)
    fe = sorted(b for b in d.bc.unique() if b != REFERENCE)
    y = np.log10(d.cbt.values + 1)
    cens = d.cens.values
    lam = 0.1

    # production (in-sample) fit
    X = design(d, stats_of(d), fe)
    beta, sigma = fit_tobit(X, y, cens, lam)
    yhat = X @ beta
    yplot = np.where(cens & (yhat > DL_LOG), DL_LOG, yhat)
    r2_in = 1 - ((y - yplot) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae_in = np.abs(y - yplot).mean()

    # leave-one-out, z-scores re-derived per fold, baselines global (as the JS)
    loo = np.zeros(len(d))
    for i in range(len(d)):
        tr = np.arange(len(d)) != i
        st = stats_of(d[tr])
        b, _ = fit_tobit(design(d[tr], st, fe), y[tr], cens[tr], lam)
        p = (design(d.iloc[[i]], st, fe) @ b)[0]
        loo[i] = DL_LOG if (cens[i] and p > DL_LOG) else p
    agree = np.where(cens, loo >= CENS_MIN_YHAT, np.abs(loo - y) <= DIFF_CI_HALF)
    r2_loo = 1 - ((y - loo) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae_loo = np.abs(y - loo).mean()

    tobit_bin = {thr: youden(10 ** loo - 1, (d.cbt.values >= thr).astype(int)) for thr in (1, 10)}

    # class-weighted logistic, leave-one-out, Youden on the LOO probabilities
    def logit_design(dd, st):
        return design(dd, st, fe)
    logi = {}
    for thr in (1, 10):
        yb = (d.cbt.values >= thr).astype(int)
        pr = np.zeros(len(d))
        for i in range(len(d)):
            tr = np.arange(len(d)) != i
            st = stats_of(d[tr])
            Xt = logit_design(d[tr], st)
            ytr = yb[tr]; n1 = len(ytr)
            c1, c0 = ytr.sum(), n1 - ytr.sum()
            w = np.where(ytr == 1, n1 / (2 * c1), n1 / (2 * c0))
            W = logistic_gd(Xt, ytr.astype(float), w, 0.03, 5000, 0.02)
            pr[i] = logistic_prob(logit_design(d.iloc[[i]], st), W)[0]
        logi[thr] = youden(pr, yb)

    # the paper's paper_data.json takes its logistic figures from fitLogisticInSample
    logi_in = {}
    Xall = design(d, stats_of(d), fe)
    for thr in (1, 10):
        yb = (d.cbt.values >= thr).astype(int)
        n1, c1 = len(yb), yb.sum()
        w = np.where(yb == 1, n1 / (2 * c1), n1 / (2 * (n1 - c1)))
        p = logistic_prob(Xall, logistic_gd(Xall, yb.astype(float), w, 0.03, 10000, 0.02))
        tp = fp = 0; bestJ = -9; tuned = 0
        for i in np.argsort(-p, kind='stable'):
            tp += yb[i]; fp += 1 - yb[i]
            J = tp / c1 - fp / (n1 - c1)
            if J > bestJ:
                bestJ, tuned = J, p[i]
        yh = p >= tuned
        logi_in[thr] = ((yh & (yb == 1)).sum() / c1 + (~yh & (yb == 0)).sum() / (n1 - c1)) / 2

    ours = {
        'coefficients': beta.tolist(), 'sigma': sigma, 'r2_in': r2_in, 'mae_in': mae_in,
        'loocv_r2': r2_loo, 'loocv_mae': mae_loo, 'loocv_agree': int(agree.sum()),
        'tobit_ge1_ba': tobit_bin[1]['ba'], 'tobit_ge10_ba': tobit_bin[10]['ba'], 'tobit_ge10_cut': tobit_bin[10]['cut'],
        'logistic_ge1_ba': logi_in[1], 'logistic_ge10_ba': logi_in[10],
        'baselines_mon2': d.attrs['mon2n_baseline'], 'baselines_tof': d.attrs['tofn_baseline'],
    }
    ours_extra = {'logistic_ge1_ba_LOO': logi[1]['ba'], 'logistic_ge10_ba_LOO': logi[10]['ba']}
    pm = paper['model']
    ref = {
        'coefficients': pm['coefficients'], 'sigma': pm['sigma'], 'r2_in': pm['r2'], 'mae_in': pm['mae'],
        'loocv_r2': pm['loocv_r2'], 'loocv_mae': pm['loocv_mae'], 'loocv_agree': pm['loocv_agree'],
        'tobit_ge1_ba': paper['binary_classification']['ge1']['balanced_accuracy'],
        'tobit_ge10_ba': paper['binary_classification']['ge10']['balanced_accuracy'],
        'tobit_ge10_cut': paper['binary_classification']['ge10']['cutpoint'],
        'logistic_ge1_ba': paper['logistic_classification']['ge1']['balanced_accuracy'],
        'logistic_ge10_ba': paper['logistic_classification']['ge10']['balanced_accuracy'],
        'baselines_mon2': paper['normalization']['sensor_baselines_mon2c'],
        'baselines_tof': paper['normalization']['sensor_baselines_tof'],
    }
    print('RUNG 0: reproduce the submitted paper from its own paired table (n=%d)' % len(d))
    for k in ours:
        a, b = ours[k], ref[k]
        if isinstance(a, list):
            diff = max(abs(x - y) for x, y in zip(a, b))
            print(f'  {k:18s} max |diff| {diff:.2e}')
        elif isinstance(a, dict):
            print(f'  {k:18s} ours {a}  paper {b}')
        else:
            print(f'  {k:18s} ours {a:.6f}  paper {b:.6f}  diff {a - b:+.2e}')
    print('  (paper logistic figures are the in-sample fit; the same model leave-one-out gives',
          {k: round(v, 4) for k, v in ours_extra.items()}, ')')
    return dict(ours, **ours_extra)


# ---- rung 1: the protocol, one element at a time, on the paper's own pairs ---------------
def baselines_fit(tr):
    base = {}
    for col, raw in (('mon2n', 'mon2'), ('tofn', 'tof')):
        base[col] = {}
        for bc, g in tr.groupby('bc'):
            clean = g[g.cbt == 0][raw]
            base[col][bc] = float(clean.median()) if len(clean) else float(np.sort(g[raw])[int(len(g) * 0.1)])
    return base


def baselines_apply(d, base):
    d = d.copy()
    for col, raw in (('mon2n', 'mon2'), ('tofn', 'tof')):
        d[col] = d[raw] - d.bc.map(base[col])
    return d


def cv_predict(d, y, cens, folds, fe, lam=0.1, infold_baseline=True, infold_cut=True, global_base=None):
    n = len(d)
    pred = np.full(n, np.nan)
    cut = {1: np.full(n, np.nan), 10: np.full(n, np.nan)}
    for f in pd.unique(folds):
        te = folds == f
        tr = ~te
        base = baselines_fit(d[tr]) if infold_baseline else global_base
        dtr, dte = baselines_apply(d[tr], base), baselines_apply(d[te], base)
        st = stats_of(dtr)
        Xtr = design(dtr, st, fe)
        b, _ = fit_tobit(Xtr, y[tr], cens[tr], lam)
        p = design(dte, st, fe) @ b
        pred[te] = np.where(cens[te] & (p > DL_LOG), DL_LOG, p)
        if infold_cut:
            ptr = Xtr @ b
            for thr in (1, 10):
                cut[thr][te] = youden(10 ** ptr - 1, (d.cbt.values[tr] >= thr).astype(int))['cut']
    if not infold_cut:
        for thr in (1, 10):
            cut[thr][:] = youden(10 ** pred - 1, (d.cbt.values >= thr).astype(int))['cut']
    return pred, cut


def score(y, pred, cens, cbt, cut):
    """Continuous and classification metrics. Classification uses each row's own cut."""
    out = {}
    agree = np.where(cens, pred >= CENS_MIN_YHAT, np.abs(pred - y) <= DIFF_CI_HALF)
    out['agree_pct'] = 100 * agree.mean()
    out['r2'] = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    out['mae'] = np.abs(y - pred).mean()
    for thr in (1, 10):
        obs = cbt >= thr
        pos = (10 ** pred - 1) >= cut[thr]
        tp, fp = int((pos & obs).sum()), int((pos & ~obs).sum())
        tn, fn = int((~pos & ~obs).sum()), int((~pos & obs).sum())
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        out[f'ge{thr}'] = dict(tp=tp, fp=fp, tn=tn, fn=fn, sens=sens, spec=spec, ba=(sens + spec) / 2,
                                ppv=tp / (tp + fp) if tp + fp else np.nan, npv=tn / (tn + fn) if tn + fn else np.nan)
    return out


def flat(s):
    f = {k: v for k, v in s.items() if not isinstance(v, dict)}
    for thr in (1, 10):
        for k in ('sens', 'spec', 'ba', 'ppv', 'npv'):
            f[f'ge{thr}_{k}'] = s[f'ge{thr}'][k]
    return f


def day_bootstrap(groups, fn, B=2000, seed=7):
    """95% percentile intervals, resampling whole sampling days with replacement."""
    rng = np.random.default_rng(seed)
    days = pd.unique(groups)
    idx_by_day = {g: np.where(groups == g)[0] for g in days}
    draws = []
    for _ in range(B):
        pick = rng.choice(days, size=len(days), replace=True)
        idx = np.concatenate([idx_by_day[g] for g in pick])
        draws.append(flat(fn(idx)))
    D = pd.DataFrame(draws)
    return {k: (float(np.nanpercentile(D[k], 2.5)), float(np.nanpercentile(D[k], 97.5))) for k in D.columns}


def sample_level(d, y, pred, cens, cut, groups):
    """One row per sample group: observed = mean log10(CBT+1) over its records, predicted =
    mean over its pairs, censored if every record is censored."""
    g = pd.DataFrame(dict(grp=groups, bag=d.bag.values, y=y, pred=pred, cens=cens, c1=cut[1], c10=cut[10], day=d.day.values))
    bag = g.groupby(['grp', 'bag']).agg(y=('y', 'first'), cens=('cens', 'first')).reset_index()
    obs = bag.groupby('grp').agg(y=('y', 'mean'), cens=('cens', 'all'))
    # pairs of one sample can come from different folds (new-sensor scheme) with different cuts;
    # the sample's decision is the mean of each pair's margin above its own cut, i.e. mean
    # prediction against the mean of log10(cut + 1). Identical when all pairs share a fold.
    logmean_cut = lambda v: 10 ** np.log10(np.asarray(v) + 1).mean() - 1
    pr = g.groupby('grp').agg(pred=('pred', 'mean'), c1=('c1', logmean_cut), c10=('c10', logmean_cut), day=('day', 'first'))
    s = obs.join(pr)
    s['cbt'] = 10 ** s.y - 1
    return s


def load_paper_pairs():
    raw = pd.read_csv(OVERLEAF / 'paired_observations.csv')
    d = pd.DataFrame(dict(bc=raw.barcode.astype(str), cbt=raw.cbt_ecoli_cfu.astype(float),
                          cens=raw.cbt_censored.astype(str).str.upper().eq('TRUE') | (raw.cbt_ecoli_cfu >= CBT_DL),
                          mon2=raw.sensor_mon2_raw.astype(float), tof=raw.sensor_tof_raw.astype(float),
                          temp=raw.water_temp_c.astype(float), day=raw.sample_date.str[:10],
                          local=raw.sample_date.str[:16]))
    # map each paper row to its mWater record and sample group from step 2
    bags = pd.read_csv(REV / 'bags.csv')
    bags['local'] = bags.sample_local.str[:16]
    key = bags.set_index('local')
    d['bag'] = d.local.map(key.bag_id)
    d['grp'] = d.local.map(key.sample_group)
    d['grp'] = d.grp.fillna('solo-' + d.bag.astype(str))
    return d


def rung1():
    d = load_paper_pairs()
    fe = sorted(b for b in d.bc.unique() if b != REFERENCE)
    y = np.log10(d.cbt.values + 1)
    cens = d.cens.values
    loo = np.arange(len(d))
    days = d.day.values
    gbase = baselines_fit(d)
    rows = {}

    def run(name, folds, infold_baseline, infold_cut):
        pred, cut = cv_predict(d, y, cens, folds, fe, infold_baseline=infold_baseline, infold_cut=infold_cut, global_base=gbase)
        s = score(y, pred, cens, d.cbt.values, cut)
        rows[name] = dict(flat(s), ge10_counts={k: s['ge10'][k] for k in ('tp', 'fp', 'tn', 'fn')})
        return pred, cut

    run('0  paper protocol (LOO, global baselines, global cut)', loo, False, False)
    run('1a + baselines fitted in fold', loo, True, False)
    run('1b + cutpoint chosen in fold', loo, True, True)
    pred, cut = run('1c + leave one sampling day out', days, True, True)

    # CIs for 1c (pair level), resampling days
    ci_pair = day_bootstrap(days, lambda idx: score(y[idx], pred[idx], cens[idx], d.cbt.values[idx], {t: cut[t][idx] for t in (1, 10)}))

    # 1d: score one prediction per sample group
    s = sample_level(d, y, pred, cens, cut, d.grp.values)
    sy, sp, sc, scbt = s.y.values, s.pred.values, s.cens.values, s.cbt.values
    scut = {1: s.c1.values, 10: s.c10.values}
    ss = score(sy, sp, sc, scbt, scut)
    rows['1d + scored per sample group'] = dict(flat(ss), ge10_counts={k: ss['ge10'][k] for k in ('tp', 'fp', 'tn', 'fn')}, n=len(s))
    ci_sample = day_bootstrap(s.day.values, lambda idx: score(sy[idx], sp[idx], sc[idx], scbt[idx], {t: scut[t][idx] for t in (1, 10)}))

    print('\nRUNG 1: protocol changes on the paper\'s own %d pairs (%d days, %d sample groups)' % (len(d), len(pd.unique(days)), len(s)))
    cols = ['agree_pct', 'r2', 'mae', 'ge10_ba', 'ge10_sens', 'ge10_spec', 'ge10_ppv', 'ge1_ba']
    T = pd.DataFrame(rows).T
    print(T[cols].astype(float).round(3).to_string())
    print('ge10 counts:'); print(T['ge10_counts'].to_string())
    print('\n95% day-bootstrap intervals, 1c (pairs):', {k: tuple(round(v, 3) for v in ci_pair[k]) for k in cols})
    print('95% day-bootstrap intervals, 1d (sample groups):', {k: tuple(round(v, 3) for v in ci_sample[k]) for k in cols})
    json.dump(dict(rows=rows, ci_1c=ci_pair, ci_1d=ci_sample), open(REV / 'ladder_rung1.json', 'w'), indent=2, default=float)
    return rows


# ---- rungs 2-3: new pairing, then the corrections, one at a time -----------------------
def load_new_pairs():
    P = pd.read_csv(REV / 'pairs.csv')
    B = pd.read_csv(REV / 'bags.csv')[['bag_id', 'sample_local', 'sample_group']]
    P = P.merge(B, on='bag_id', suffixes=('', '_b'))
    grp = P['sample_group'] if 'sample_group' in P else P['sample_group_b']
    return pd.DataFrame(dict(bc=P.barcode.astype(str), cbt=P.cbt.astype(float),
                             cens=P.censored.astype(str).str.upper().eq('TRUE') | (P.cbt >= CBT_DL),
                             mon2=P.mon2.astype(float), tof=P.tof.astype(float), temp=P.temp_diag.astype(float),
                             sig=P.sig.astype(float), sig_T=P.sig_T.astype(float), sig_TD=P.sig_TD.astype(float),
                             day=P.sample_local.str[:10], bag=P.bag_id, grp=grp.values,
                             treated=P.water_type.astype(str).str.startswith('Treated').values))


def gen_baselines(tr, cols):
    base = {}
    for col in cols:
        base[col] = {}
        for bc, g in tr.groupby('bc'):
            clean = g[g.cbt == 0][col]
            base[col][bc] = float(clean.median()) if len(clean) else float(np.sort(g[col])[int(len(g) * 0.1)])
    return base


def gen_apply(d, base):
    d = d.copy()
    for col, b in base.items():
        d[col + '_n'] = d[col] - d.bc.map(b)
    return d


def gen_design(d, st, feats, fe, slope_feat):
    z = {f: (d[f] - st[f][0]) / st[f][1] for f in feats}
    cols = [np.ones(len(d))] + [z[f].values for f in feats]
    cols += [(d.bc == bc).astype(float).values for bc in fe]
    if slope_feat:
        cols += [np.where(d.bc == bc, z[slope_feat], 0.0) for bc in fe]
    return np.column_stack(cols)


def cv_generic(d, y, cens, folds, fe, spec, lam=0.1):
    n = len(d)
    pred = np.full(n, np.nan)
    cut = {1: np.full(n, np.nan), 10: np.full(n, np.nan)}
    for f in pd.unique(folds):
        te = folds == f
        tr = ~te
        base = gen_baselines(d[tr], spec['baseline'])
        dtr, dte = gen_apply(d[tr], base), gen_apply(d[te], base)
        st = {f_: zstat(dtr[f_]) for f_ in spec['feats']}
        Xtr = gen_design(dtr, st, spec['feats'], fe, spec['slope'])
        b, _ = fit_tobit(Xtr, y[tr], cens[tr], lam)
        p = gen_design(dte, st, spec['feats'], fe, spec['slope']) @ b
        pred[te] = np.where(cens[te] & (p > DL_LOG), DL_LOG, p)
        ptr = Xtr @ b
        for thr in (1, 10):
            cut[thr][te] = youden(10 ** ptr - 1, (d.cbt.values[tr] >= thr).astype(int))['cut']
    return pred, cut


SPECS = {
    '2  new pairing (paper features)':           dict(baseline=['mon2', 'tof'], feats=['mon2_n', 'temp', 'tof_n'], slope='mon2_n'),
    '3a + log signal ln(mon2-170)':              dict(baseline=['sig', 'tof'], feats=['sig_n', 'temp', 'tof_n'], slope='sig_n'),
    '3b + batch temperature correction':         dict(baseline=['sig_T', 'tof'], feats=['sig_T_n', 'tof_n'], slope='sig_T_n'),
    '3b-check  (corrected, free temp term kept)': dict(baseline=['sig_T', 'tof'], feats=['sig_T_n', 'temp', 'tof_n'], slope='sig_T_n'),
    '3c + drift reference (treated water)':      dict(baseline=['tof'], feats=['sig_TD', 'tof_n'], slope='sig_TD'),
}


def rung23():
    d = load_new_pairs()
    fe = sorted(b for b in d.bc.unique() if b != REFERENCE)
    y = np.log10(d.cbt.values + 1)
    cens = d.cens.values
    days = d.day.values
    rows, cis = {}, {}
    for name, spec in SPECS.items():
        pred, cut = cv_generic(d, y, cens, days, fe, spec)
        sp_pair = score(y, pred, cens, d.cbt.values, cut)
        s = sample_level(d, y, pred, cens, cut, d.grp.values)
        sy, spd, sc, scbt = s.y.values, s.pred.values, s.cens.values, s.cbt.values
        scut = {1: s.c1.values, 10: s.c10.values}
        ss = score(sy, spd, sc, scbt, scut)
        rows[name] = dict(flat(ss), pair_ge10_ba=sp_pair['ge10']['ba'], n_samples=len(s), n_pairs=len(d),
                          ge10_counts={k: ss['ge10'][k] for k in ('tp', 'fp', 'tn', 'fn')})
        cis[name] = day_bootstrap(s.day.values, lambda idx: score(sy[idx], spd[idx], sc[idx], scbt[idx], {t: scut[t][idx] for t in (1, 10)}))
    print('\nRUNGS 2-3: rebuilt pairs (%d pairs, %d sample groups, %d days); leave-one-day-out, in-fold baselines and cuts, scored per sample group'
          % (len(d), d.grp.nunique(), len(pd.unique(days))))
    cols = ['agree_pct', 'r2', 'mae', 'ge10_ba', 'ge10_sens', 'ge10_spec', 'ge10_ppv', 'ge1_ba', 'pair_ge10_ba']
    T = pd.DataFrame(rows).T
    print(T[cols].astype(float).round(3).to_string())
    print('ge10 counts (sample groups):'); print(T['ge10_counts'].to_string())
    for name in rows:
        print(f'  95% CI {name}: ge10_ba {tuple(round(v, 3) for v in cis[name]["ge10_ba"])}  r2 {tuple(round(v, 3) for v in cis[name]["r2"])}')
    json.dump(dict(rows=rows, ci=cis), open(REV / 'ladder_rung23.json', 'w'), indent=2, default=float)
    return rows


# ---- step 4: model form, candidates fixed before running --------------------------------
CANDIDATES = {
    'A  per-sensor intercepts + per-sensor F slopes': dict(feats=['sig_TD', 'tof_n'], fe=True, unit_slopes=['sig_TD']),
    'B  per-sensor intercepts, shared F slope':       dict(feats=['sig_TD', 'tof_n'], fe=True),
    'C  fully shared (no per-sensor terms)':          dict(feats=['sig_TD', 'tof_n'], fe=False),
    'D  B + T and FxT (method paper Eq. 2 terms)':    dict(feats=['sig_TD', 'Tc', 'FT', 'tof_n'], fe=True),
    'E  Eq. 2 fitted per unit':                      dict(feats=['sig_TD', 'Tc', 'FT', 'tof_n'], fe=True, per_unit_all=True),
}
LOSO = {
    'C  fully shared': dict(feats=['sig_TD', 'tof_n'], fe=False),
    'D0 Eq. 2 terms, shared, no per-sensor terms': dict(feats=['sig_TD', 'Tc', 'FT', 'tof_n'], fe=False),
}


def design2(d, st, spec, units, fe_units):
    z = {f: ((d[f] - st[f][0]) / st[f][1]).values for f in spec['feats']}
    cols = [np.ones(len(d))]
    if spec.get('fe'):
        cols += [(d.bc == u).astype(float).values for u in fe_units]
    if spec.get('per_unit_all'):
        for u in units:
            cols += [np.where(d.bc == u, z[f], 0.0) for f in spec['feats']]
    else:
        cols += [z[f] for f in spec['feats']]
        for f in spec.get('unit_slopes', []):
            cols += [np.where(d.bc == u, z[f], 0.0) for u in fe_units]
    return np.column_stack(cols)


def fit_predict(dtr, ytr, ctr, dte, spec, units, fe_units, lam=0.1, new_unit_rows=None):
    base = gen_baselines(dtr, ['tof'])
    # a unit absent from training gets a ToF baseline without its CBT results: the median
    # ToF of its own treated-water records, chosen by water type
    if new_unit_rows is not None:
        for u, g in new_unit_rows.groupby('bc'):
            if u not in base['tof']:
                t = g[g.treated]
                base['tof'][u] = float((t if len(t) else g).tof.median())
    dtr, dte = gen_apply(dtr, base), gen_apply(dte, base)
    st = {f: zstat(dtr[f]) for f in spec['feats']}
    Xtr = design2(dtr, st, spec, units, fe_units)
    b, _ = fit_tobit(Xtr, ytr, ctr, lam)
    return design2(dte, st, spec, units, fe_units) @ b, Xtr @ b


def step4():
    d = load_new_pairs()
    P = pd.read_csv(REV / 'pairs.csv')
    d['Tc'] = P.Tch.values - TREF_C
    d['FT'] = d.sig_TD * d.Tc
    units = sorted(d.bc.unique())
    fe_units = [u for u in units if u != REFERENCE]
    y = np.log10(d.cbt.values + 1)
    cens = d.cens.values
    days = d.day.values
    rows, cis = {}, {}
    for name, spec in CANDIDATES.items():
        pred = np.full(len(d), np.nan)
        cut = {1: np.full(len(d), np.nan), 10: np.full(len(d), np.nan)}
        for g in pd.unique(days):
            te = days == g
            p, ptr = fit_predict(d[~te], y[~te], cens[~te], d[te], spec, units, fe_units)
            pred[te] = np.where(cens[te] & (p > DL_LOG), DL_LOG, p)
            for thr in (1, 10):
                cut[thr][te] = youden(10 ** ptr - 1, (d.cbt.values[~te] >= thr).astype(int))['cut']
        s = sample_level(d, y, pred, cens, cut, d.grp.values)
        sy, spd, sc, scbt = s.y.values, s.pred.values, s.cens.values, s.cbt.values
        scut = {1: s.c1.values, 10: s.c10.values}
        ss = score(sy, spd, sc, scbt, scut)
        rows[name] = dict(flat(ss), ge10_counts={k: ss['ge10'][k] for k in ('tp', 'fp', 'tn', 'fn')})
        cis[name] = day_bootstrap(s.day.values, lambda idx: score(sy[idx], spd[idx], sc[idx], scbt[idx], {t: scut[t][idx] for t in (1, 10)}))
        s.assign(model=name.split()[0]).to_csv(REV / f'heldout_samples_{name.split()[0]}.csv')

    # new-unit test: hold out one sensor AND one day; train on the other sensors, other days
    loso = {}
    for name, spec in LOSO.items():
        for u in units:
            pred = np.full(len(d), np.nan)
            cut = {1: np.full(len(d), np.nan), 10: np.full(len(d), np.nan)}
            for g in pd.unique(days[d.bc.values == u]):
                te = (d.bc.values == u) & (days == g)
                tr = (d.bc.values != u) & (days != g)
                p, ptr = fit_predict(d[tr], y[tr], cens[tr], d[te], spec, units, [], new_unit_rows=d[d.bc.values == u])
                pred[te] = np.where(cens[te] & (p > DL_LOG), DL_LOG, p)
                for thr in (1, 10):
                    cut[thr][te] = youden(10 ** ptr - 1, (d.cbt.values[tr] >= thr).astype(int))['cut']
            m = d.bc.values == u
            s = sample_level(d[m], y[m], pred[m], cens[m], {t: cut[t][m] for t in (1, 10)}, d.grp.values[m])
            ss = score(s.y.values, s.pred.values, s.cens.values, s.cbt.values, {1: s.c1.values, 10: s.c10.values})
            loso[f'{name} | held out {u}'] = dict(flat(ss), n=len(s), ge10_counts={k: ss['ge10'][k] for k in ('tp', 'fp', 'tn', 'fn')})

    cols = ['agree_pct', 'r2', 'mae', 'ge10_ba', 'ge10_sens', 'ge10_spec', 'ge10_ppv', 'ge1_ba']
    print('\nSTEP 4: model form (leave-one-day-out, scored per sample group, n=%d)' % d.grp.nunique())
    T = pd.DataFrame(rows).T
    print(T[cols].astype(float).round(3).to_string())
    print(T['ge10_counts'].to_string())
    for name in rows:
        print(f'  95% CI {name}: ge10_ba {tuple(round(v, 3) for v in cis[name]["ge10_ba"])}  r2 {tuple(round(v, 3) for v in cis[name]["r2"])}')
    print('\nNEW-UNIT TEST (sensor and day held out together), scored per sample group within the held-out sensor')
    L = pd.DataFrame(loso).T
    print(L[['n'] + cols].astype(float).round(3).to_string())
    print(L['ge10_counts'].to_string())
    json.dump(dict(rows=rows, ci=cis, loso=loso), open(REV / 'step4_models.json', 'w'), indent=2, default=float)


TREF_C = 20.0


# ---- index of agreement, as in the TLF methods paper (scripts/recompute_ia_table.py) -----
ND_SUB = 0.1   # a zero (non-detect) enters the all-pairs rows as 0.1, as in the methods paper


def ia_r2(x, y):
    """Willmott d on log10 pairs, x = reference, y = candidate; both denominator terms use the
    reference mean (EPA-820-R-14-011 App. E). R^2 = squared Pearson r. RMSE in log10."""
    lx, ly = np.log10(np.asarray(x, float)), np.log10(np.asarray(y, float))
    mx = lx.mean()
    d = 1 - ((lx - ly) ** 2).sum() / ((np.abs(lx - mx) + np.abs(ly - mx)) ** 2).sum()
    r = np.corrcoef(lx, ly)[0, 1] if len(lx) > 2 else np.nan
    return dict(n=len(lx), ia=float(d), r2=float(r * r), rmse=float(np.sqrt(((lx - ly) ** 2).mean())))


def ia_rows(obs, pred, cens, day, B=2000):
    """All pairs (non-detects as 0.1, censored at the 100 MPN ceiling) and the rows within the
    quantification limits of both assays, at a lower limit of 1 and of 10 MPN/100 mL."""
    o = np.where(obs <= 0, ND_SUB, obs)
    p = np.maximum(pred, ND_SUB)
    masks = {
        'all pairs': np.ones(len(o), bool),
        'both within limits, >=1 (CBT 1-99, uncensored)': (obs >= 1) & (obs < CBT_DL) & ~cens & (pred >= 1),
        'both within limits, >=10 (CBT 10-99, uncensored)': (obs >= 10) & (obs < CBT_DL) & ~cens & (pred >= 10),
    }
    out = {}
    rng = np.random.default_rng(7)
    days = pd.unique(day)
    for k, m in masks.items():
        if m.sum() < 3:
            out[k] = dict(n=int(m.sum())); continue
        r = ia_r2(o[m], p[m])
        draws = []
        for _ in range(B):
            pick = rng.choice(days, size=len(days), replace=True)
            idx = np.concatenate([np.where((day == g) & m)[0] for g in pick])
            if len(idx) >= 3:
                draws.append(ia_r2(o[idx], p[idx])['ia'])
        r['ia_ci'] = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
        out[k] = r
    return out


def step_ia():
    res = {}
    for mdl in 'ABCDE':
        s = pd.read_csv(REV / f'heldout_samples_{mdl}.csv')
        res[f'model {mdl} (held out, per sample)'] = ia_rows(s.cbt.values, 10 ** s.pred.values - 1,
                                                             s.cens.values.astype(bool), s.day.values)
    # the submitted paper's own leave-one-out predictions, per pair
    paper = json.load(open(OVERLEAF / 'paper_data.json'))['paired_points']
    pp = pd.DataFrame(paper)
    raw = pd.read_csv(OVERLEAF / 'paired_observations.csv')
    res['submitted paper (LOO, per pair)'] = ia_rows(raw.cbt_ecoli_cfu.values.astype(float), 10 ** pp.predicted_loo_log.values - 1,
                                                     raw.cbt_censored.astype(str).str.upper().eq('TRUE').values | (raw.cbt_ecoli_cfu.values >= CBT_DL),
                                                     raw.sample_date.str[:10].values)
    # reference against reference: consecutive replicate CBTs in one sample group
    B_ = pd.read_csv(REV / 'bags.csv').dropna(subset=['sample_group'])
    B_['t'] = pd.to_datetime(B_.t_utc, utc=True)
    xs, ys, cs, ds = [], [], [], []
    for _, g in B_.sort_values('t').groupby('sample_group'):
        for i in range(1, len(g)):
            xs.append(g.cbt.iloc[i - 1]); ys.append(g.cbt.iloc[i])
            cs.append(bool(g.censored.iloc[i - 1] or g.censored.iloc[i])); ds.append(str(g.sample_local.iloc[i])[:10])
    xs, ys = np.array(xs, float), np.array(ys, float)
    rep = ia_rows(xs, ys, np.array(cs), np.array(ds))
    res['CBT vs replicate CBT (reference bar)'] = rep
    print('\nINDEX OF AGREEMENT (Willmott, log10, reference mean in both terms); 95% CI by resampling days')
    for name, rows in res.items():
        print(name)
        for k, r in rows.items():
            if 'ia' in r:
                print(f"   {k:52s} n={r['n']:3d}  IA {r['ia']:.3f} {tuple(round(v, 3) for v in r['ia_ci'])}  R2 {r['r2']:.3f}  RMSE {r['rmse']:.3f}")
            else:
                print(f"   {k:52s} n={r['n']:3d}  (too few)")
    print('   replicate pairs: identical', int((xs == ys).sum()), 'of', len(xs), '| both zero', int(((xs == 0) & (ys == 0)).sum()))
    json.dump(res, open(REV / 'step_ia.json', 'w'), indent=2, default=float)


# ---- deployable approach: shared coefficients, daily clean-water baseline only ------------
DEPLOY_MODELS = {
    'C  shared: F, B': ['F', 'B'],
    'D0 shared Eq. 2 terms: F, B, T, FxT': ['F', 'B', 'Tc', 'FT'],
}


BRIDGE = 'nearest'   # 'interp' after the no-CBT bridging comparison (step_bridge)


def daily_baseline(P, bridge=None):
    """Per unit and day, the clean-water baseline is the median over that day's treated-water
    buckets (chosen by water type), excluding the bucket being scored. Days with none take the
    nearest day ('nearest') or interpolate between the nearest earlier and later days with one
    ('interp'; one-sided gaps fall back to nearest). Applied to temperature-corrected
    fluorescence (sig_T) and to ToF."""
    bridge = bridge or BRIDGE
    P = P.copy()
    P['day'] = P.sample_local.str[:10]
    bk = P[P.water_type.str.startswith('Treated')].groupby(['barcode', 'soak_id']).agg(
        day=('day', 'first'), sig_T=('sig_T', 'median'), tof=('tof', 'median')).reset_index()
    ref_s, ref_b, age = [], [], []
    for _, r in P.iterrows():
        c = bk[(bk.barcode == r.barcode) & (bk.soak_id != r.soak_id)]
        dm = c.groupby('day')[['sig_T', 'tof']].median().reset_index()
        dm['gap'] = [(pd.Timestamp(x) - pd.Timestamp(r.day)).days for x in dm.day]
        same = dm[dm.gap == 0]
        before, after = dm[dm.gap < 0], dm[dm.gap > 0]
        if len(same):
            s_, b_ = same.sig_T.iloc[0], same.tof.iloc[0]
        elif bridge == 'interp' and len(before) and len(after):
            b1, a1 = before.loc[before.gap.idxmax()], after.loc[after.gap.idxmin()]
            w = -b1.gap / (a1.gap - b1.gap)
            s_, b_ = b1.sig_T + w * (a1.sig_T - b1.sig_T), b1.tof + w * (a1.tof - b1.tof)
        else:
            best = dm[dm.gap.abs() == dm.gap.abs().min()]
            s_, b_ = best.sig_T.mean(), best.tof.mean()
        ref_s.append(s_); ref_b.append(b_); age.append(int(dm.gap.abs().min()))
    P['F'] = P.sig_T - np.array(ref_s)
    P['B'] = P.tof - np.array(ref_b)
    P['Blog'] = np.log(P.tof / np.array(ref_b))     # ToF as a log ratio to the day's baseline
    P['Tc'] = P.Tch - TREF_C
    P['FT'] = P.F * P.Tc
    P['base_age'] = age
    return P


def shared_fit(tr, feats, lam=0.1):
    st = {f: zstat(tr[f]) for f in feats}
    X = np.column_stack([np.ones(len(tr))] + [((tr[f] - st[f][0]) / st[f][1]).values for f in feats])
    b, _ = fit_tobit(X, np.log10(tr.cbt.values + 1), tr.cens.values, lam)
    return lambda d: np.column_stack([np.ones(len(d))] + [((d[f] - st[f][0]) / st[f][1]).values for f in feats]) @ b


def nested_cut(tr, feats, thr):
    """Cutpoint from held-out predictions inside the training data (leave one training day out)."""
    pr = np.full(len(tr), np.nan)
    for g in pd.unique(tr.day):
        m = (tr.day == g).values
        if m.all():
            continue
        pr[m] = shared_fit(tr[~m], feats)(tr[m])
    ok = ~np.isnan(pr)
    return youden(10 ** pr[ok] - 1, (tr.cbt.values[ok] >= thr).astype(int))['cut']


TOF_VARIANTS = {
    'D0a ToF as is: F, B, T, FxT':         ['F', 'B', 'Tc', 'FT'],
    'D0b ToF log ratio: F, Blog, T, FxT':  ['F', 'Blog', 'Tc', 'FT'],
    'D0c no ToF: F, T, FxT':               ['F', 'Tc', 'FT'],
    'Ca ToF as is: F, B':                  ['F', 'B'],
    'Cb ToF log ratio: F, Blog':           ['F', 'Blog'],
    'Cc no ToF: F':                        ['F'],
}


def step_deploy(models=None, outfile='step_deploy.json'):
    models = models or DEPLOY_MODELS
    P = pd.read_csv(REV / 'pairs.csv').merge(pd.read_csv(REV / 'bags.csv')[['bag_id', 'sample_local']], on='bag_id')
    P = daily_baseline(P)
    P['cens'] = P.censored.astype(str).str.upper().eq('TRUE') | (P.cbt >= CBT_DL)
    P['bc'] = P.barcode.astype(str)
    P['grp'] = P.sample_group
    P['bag'] = P.bag_id
    y = np.log10(P.cbt.values + 1)
    out, ia_out = {}, {}
    for name, feats in models.items():
        for scheme in ('day held out', 'new sensor + new day'):
            pred = np.full(len(P), np.nan)
            cut = {1: np.full(len(P), np.nan), 10: np.full(len(P), np.nan)}
            keys = [(None, g) for g in pd.unique(P.day)] if scheme == 'day held out' else \
                   [(u, g) for u in P.bc.unique() for g in pd.unique(P.day[P.bc == u])]
            for u, g in keys:
                te = (P.day == g).values & ((P.bc == u).values if u else True)
                tr = (P.day != g).values & ((P.bc != u).values if u else True)
                p = shared_fit(P[tr], feats)(P[te])
                pred[te] = np.where(P.cens.values[te] & (p > DL_LOG), DL_LOG, p)
                for thr in (1, 10):
                    cut[thr][te] = nested_cut(P[tr].reset_index(drop=True), feats, thr)
            for subset in (('all',) if models is not DEPLOY_MODELS else ('all', 'same-day baseline only')):
                m = np.ones(len(P), bool) if subset == 'all' else (P.base_age.values == 0)
                s = sample_level(P[m], y[m], pred[m], P.cens.values[m], {t: cut[t][m] for t in (1, 10)}, P.grp.values[m])
                sy, spd, sc, scbt = s.y.values, s.pred.values, s.cens.values, s.cbt.values
                scut = {1: s.c1.values, 10: s.c10.values}
                ss = score(sy, spd, sc, scbt, scut)
                ci = day_bootstrap(s.day.values, lambda idx: score(sy[idx], spd[idx], sc[idx], scbt[idx], {t: scut[t][idx] for t in (1, 10)}))
                key = f'{name} | {scheme} | {subset}'
                out[key] = dict(flat(ss), n=len(s), ba_ci=ci['ge10_ba'], counts={k: ss['ge10'][k] for k in ('tp', 'fp', 'tn', 'fn')})
                if subset == 'all':
                    ia_out[f'{name} | {scheme}'] = ia_rows(scbt, 10 ** spd - 1, sc.astype(bool), s.day.values)
    cols = ['n', 'ge10_ba', 'ge10_sens', 'ge10_spec', 'ge10_ppv', 'ge1_ba', 'r2', 'agree_pct']
    T = pd.DataFrame(out).T
    print('\nDEPLOYABLE APPROACH: shared coefficients; per-unit calibration = daily clean-water baseline only')
    print(T[cols].astype(float).round(3).to_string())
    for k in out:
        print(f'  {k}: ge10 counts {out[k]["counts"]}  BA 95% CI {tuple(round(v, 3) for v in out[k]["ba_ci"])}')
    print('\nIA (all pairs; non-detects as 0.1):')
    for k, r in ia_out.items():
        a = r['all pairs']
        print(f'  {k}: IA {a["ia"]:.3f} {tuple(round(v, 3) for v in a["ia_ci"])}  R2 {a["r2"]:.3f}  RMSE {a["rmse"]:.3f}')
    print('pairs using a baseline from another day:', int((P.base_age > 0).sum()), 'of', len(P))
    json.dump(dict(rows=out, ia=ia_out), open(REV / outfile, 'w'), indent=2, default=float)


# ---- air-phase baseline: the unit's own air readings each sampling day -------------------
def air_references():
    """Per unit and sampling day: median temperature-corrected ln(mon2-170) and median ToF over
    the unit's air readings inside that day's sampling window (first to last CBT record, +/-30
    min), leaving out readings within 2 min of an in-water reading and ToF > 200 kcps."""
    import pair_soaks as ps
    J = json.load(open(ps.JUDGMENTS))
    U = {bc: ps.unit_readings(bc)[0] for bc in ps.UNITS}
    ps.apply_judgments(U, J)
    bags = pd.read_csv(REV / 'bags.csv')
    bags['t'] = pd.to_datetime(bags.t_utc, utc=True)
    rows = []
    for bc, u in U.items():
        u = u.copy()
        u['sig_T'] = u.sig - ps.BATCH_RHO * (u.Tch - ps.TREF)
        wt = u.t[u.in_water].values
        near_water = np.array([np.min(np.abs(wt - t)) <= np.timedelta64(2, 'm') if len(wt) else False for t in u.t.values])
        air = u[~u.in_water & ~near_water & (u.tof <= 200)]
        mine = bags[bags.barcodes.astype(str).str.contains(str(bc))]
        for day, g in mine.groupby(mine.sample_local.str[:10]):
            lo, hi = g.t.min() - pd.Timedelta('30min'), g.t.max() + pd.Timedelta('30min')
            a = air[(air.t >= lo) & (air.t <= hi)]
            if len(a):
                rows.append(dict(barcode=bc, day=day, n_air=len(a), air_sig=float(a.sig_T.median()),
                                 air_tof=float(a.tof.median()), air_sig_iqr=float(a.sig_T.quantile(.75) - a.sig_T.quantile(.25))))
    return pd.DataFrame(rows)


def step_air():
    A = air_references()
    P = pd.read_csv(REV / 'pairs.csv').merge(pd.read_csv(REV / 'bags.csv')[['bag_id', 'sample_local']], on='bag_id')
    P = daily_baseline(P)            # treated-water baseline, for the comparison
    P['ref_water_sig'] = P.sig_T - P.F
    P['ref_water_tof'] = P.tof - P.B
    P = P.merge(A, on=['barcode', 'day'], how='left')
    print('\nAIR-PHASE REFERENCE, per unit-day'); print(A.round(3).to_string(index=False))
    # validity: does the air reference track the same-day treated-water baseline?
    same = P[P.base_age == 0].groupby(['barcode', 'day']).agg(water=('ref_water_sig', 'first'), air=('air_sig', 'first'),
                                                              water_tof=('ref_water_tof', 'first'), air_tof=('air_tof', 'first')).dropna()
    print('\nunit-days with both a same-day treated baseline and an air reference:', len(same))
    for bc, g in same.groupby(level=0):
        if len(g) >= 3:
            print(f'  {bc}: n={len(g)} corr(air, water) sig {np.corrcoef(g.air, g.water)[0,1]:.2f}  sd(water-air) {np.std(g.water-g.air):.3f}'
                  f' | tof corr {np.corrcoef(g.air_tof, g.water_tof)[0,1]:.2f} sd {np.std(g.water_tof-g.air_tof):.1f}')
    print(same.assign(diff=same.water - same.air).round(3).to_string())
    # evaluate D0 with the air phase as the daily baseline on every day
    P['F'] = P.sig_T - P.air_sig
    P['B'] = P.tof - P.air_tof
    P['FT'] = P.F * P.Tc
    P = P.dropna(subset=['F', 'B'])
    P['cens'] = P.censored.astype(str).str.upper().eq('TRUE') | (P.cbt >= CBT_DL)
    P['bc'] = P.barcode.astype(str); P['grp'] = P.sample_group; P['bag'] = P.bag_id
    y = np.log10(P.cbt.values + 1)
    out = {}
    for name, feats in DEPLOY_MODELS.items():
        for scheme in ('day held out', 'new sensor + new day'):
            pred = np.full(len(P), np.nan)
            cut = {1: np.full(len(P), np.nan), 10: np.full(len(P), np.nan)}
            keys = [(None, g) for g in pd.unique(P.day)] if scheme == 'day held out' else \
                   [(u, g) for u in P.bc.unique() for g in pd.unique(P.day[P.bc == u])]
            for u, g in keys:
                te = (P.day == g).values & ((P.bc == u).values if u else True)
                tr = (P.day != g).values & ((P.bc != u).values if u else True)
                p = shared_fit(P[tr], feats)(P[te])
                pred[te] = np.where(P.cens.values[te] & (p > DL_LOG), DL_LOG, p)
                for thr in (1, 10):
                    cut[thr][te] = nested_cut(P[tr].reset_index(drop=True), feats, thr)
            s = sample_level(P, y, pred, P.cens.values, cut, P.grp.values)
            sy, spd, sc, scbt = s.y.values, s.pred.values, s.cens.values, s.cbt.values
            scut = {1: s.c1.values, 10: s.c10.values}
            ss = score(sy, spd, sc, scbt, scut)
            ci = day_bootstrap(s.day.values, lambda idx: score(sy[idx], spd[idx], sc[idx], scbt[idx], {t: scut[t][idx] for t in (1, 10)}))
            ia = ia_rows(scbt, 10 ** spd - 1, sc.astype(bool), s.day.values)['all pairs']
            out[f'{name} | {scheme}'] = dict(flat(ss), n=len(s), ba_ci=ci['ge10_ba'], ia=ia['ia'], ia_ci=ia['ia_ci'],
                                            counts={k: ss['ge10'][k] for k in ('tp', 'fp', 'tn', 'fn')})
    T = pd.DataFrame(out).T
    print('\nDEPLOYABLE, AIR-PHASE BASELINE on every day (pairs with an air reference: %d of 212)' % len(P))
    print(T[['n', 'ge10_ba', 'ge10_sens', 'ge10_spec', 'ge10_ppv', 'ge1_ba', 'r2', 'ia']].astype(float).round(3).to_string())
    for k in out:
        print(f'  {k}: counts {out[k]["counts"]}  BA CI {tuple(round(v, 3) for v in out[k]["ba_ci"])}  IA CI {tuple(round(v, 3) for v in out[k]["ia_ci"])}')
    json.dump(dict(air=A.to_dict('records'), rows=out), open(REV / 'step_air.json', 'w'), indent=2, default=float)


# ---- choosing how to bridge days without a clean-water baseline, without CBT results -----
def step_bridge():
    """On unit-days where the clean-water (treated-bucket) baseline is known, predict it with
    that day held out, by each bridging rule, and compare errors. Uses no CBT result."""
    P = pd.read_csv(REV / 'pairs.csv').merge(pd.read_csv(REV / 'bags.csv')[['bag_id', 'sample_local']], on='bag_id')
    P['day'] = P.sample_local.str[:10]
    t = P[P.water_type.str.startswith('Treated')]
    W = t.groupby(['barcode', 'soak_id']).agg(day=('day', 'first'), s=('sig_T', 'median'), b=('tof', 'median')) \
         .groupby(['barcode', 'day']).median().reset_index()
    A = air_references().rename(columns={'air_sig': 'as', 'air_tof': 'ab'})
    W = W.merge(A[['barcode', 'day', 'as', 'ab']], on=['barcode', 'day'], how='left')
    rows = []
    for _, r in W.iterrows():
        o = W[(W.barcode == r.barcode) & (W.day != r.day)].copy()
        if o.empty:
            continue
        o['gap'] = [(pd.Timestamp(x) - pd.Timestamp(r.day)).days for x in o.day]
        rec = dict(barcode=r.barcode, day=r.day)
        for col, air in (('s', 'as'), ('b', 'ab')):
            near = o[o.gap.abs() == o.gap.abs().min()][col].mean()
            before, after = o[o.gap < 0], o[o.gap > 0]
            if len(before) and len(after):
                b1 = before.loc[before.gap.idxmax()]; a1 = after.loc[after.gap.idxmin()]
                w = -b1.gap / (a1.gap - b1.gap)
                interp = b1[col] + w * (a1[col] - b1[col])
            else:
                interp = near
            off = (o[col] - o[air]).dropna()
            airoff = r[air] + off.median() if (pd.notna(r[air]) and len(off)) else np.nan
            rec.update({f'{col}_true': r[col], f'{col}_nearest': near, f'{col}_mean_other': o[col].mean(),
                        f'{col}_interp': interp, f'{col}_air_offset': airoff})
        rows.append(rec)
    R = pd.DataFrame(rows)
    print('\nBRIDGING RULES: held-out prediction of a known clean-water baseline (%d unit-days; no CBT used)' % len(R))
    for col, label, unit in (('s', 'fluorescence, ln units', ''), ('b', 'ToF', ' kcps')):
        print(f'  {label}: RMSE (per unit n) ')
        for rule in ('nearest', 'mean_other', 'interp', 'air_offset'):
            e = (R[f'{col}_{rule}'] - R[f'{col}_true'])
            per = {bc: round(float(np.sqrt((g ** 2).mean())), 3) for bc, g in e.groupby(R.barcode)}
            print(f'    {rule:11s} pooled {np.sqrt((e ** 2).mean()):.3f}{unit}  by unit {per}  (n={int(e.notna().sum())})')
    R.to_csv(REV / 'bridge_check.csv', index=False)


# ---- step 5: final numbers for the paper ----------------------------------------------
# Aquagenx CBT MPN table (Basis of the Aquagenx MPN Table, 2021, Table 2): MPN/100 mL and the
# upper 95% confidence level for each of the 32 compartment patterns.
AQUAGENX = [(0.0, 2.87), (1.0, 5.14), (1.0, 4.74), (1.1, 5.16), (1.2, 5.64), (1.5, 7.81), (2.0, 6.32),
            (2.1, 6.85), (2.1, 6.64), (2.4, 7.81), (2.4, 8.12), (2.6, 8.51), (3.2, 8.38), (3.7, 9.70),
            (3.1, 11.36), (3.2, 11.82), (3.4, 12.53), (3.9, 10.43), (4.0, 10.94), (4.7, 22.75), (5.2, 14.73),
            (5.4, 12.93), (5.6, 17.14), (5.8, 16.87), (8.4, 21.19), (9.1, 37.04), (9.6, 37.68), (13.6, 83.06),
            (17.1, 56.35), (32.6, 145.55), (48.3, 351.91)]
D0B = ['F', 'Blog', 'Tc', 'FT']


def cbt_interval(v, censored):
    """Reference interval for a recorded (whole-number) CBT result: the rows whose MPN rounds to it,
    taking the widest upper limit; the upper limit is one-sided 95% (Aquagenx), and the lower
    limit mirrors it in log space. Returns (lower, upper, p_ge10)."""
    if censored or v >= CBT_DL:
        return CBT_DL, np.inf, 1.0
    if v <= 0:
        return 0.0, 2.87, 0.0
    rows = [u for m, u in AQUAGENX if int(math.floor(m + 0.5)) == int(round(v))] or \
           [min(AQUAGENX, key=lambda r: abs(r[0] - v))[1]]
    U = max(rows)
    sig = (math.log10(U) - math.log10(v)) / 1.645
    lower = 10 ** (2 * math.log10(v) - math.log10(U))
    p = 1 - 0.5 * (1 + math.erf((1 - math.log10(v)) / (sig * math.sqrt(2))))
    return lower, U, p


def load_final():
    B = pd.read_csv(REV / 'bags.csv')
    P = pd.read_csv(REV / 'pairs.csv').merge(B[['bag_id', 'sample_local', 'chlorine']], on='bag_id')
    P = daily_baseline(P, 'interp')
    P['cens'] = P.censored.astype(str).str.upper().eq('TRUE') | (P.cbt >= CBT_DL)
    P['bc'] = P.barcode.astype(str); P['grp'] = P.sample_group; P['bag'] = P.bag_id
    return B, P


def fit_spec(tr, spec, lam=0.1):
    """Shared or per-unit Tobit on the daily-baseline features; returns a predictor."""
    feats = spec['feats']
    units = sorted(tr.bc.unique()); fe_units = [u for u in units if u != REFERENCE]
    st = {f: zstat(tr[f]) for f in feats}
    X = design2(tr, st, spec, units, fe_units)
    b, _ = fit_tobit(X, np.log10(tr.cbt.values + 1), tr.cens.values, lam)
    return lambda d: design2(d, st, spec, units, fe_units) @ b


def nested_cut_spec(tr, spec, thr):
    pr = np.full(len(tr), np.nan)
    for g in pd.unique(tr.day):
        m = (tr.day == g).values
        if m.all():
            continue
        pr[m] = fit_spec(tr[~m], spec)(tr[m])
    ok = ~np.isnan(pr)
    return youden(10 ** pr[ok] - 1, (tr.cbt.values[ok] >= thr).astype(int))['cut']


def cv_spec(P, spec, scheme):
    pred = np.full(len(P), np.nan)
    cut = {1: np.full(len(P), np.nan), 10: np.full(len(P), np.nan)}
    keys = [(None, g) for g in pd.unique(P.day)] if scheme == 'day held out' else \
           [(u, g) for u in P.bc.unique() for g in pd.unique(P.day[P.bc == u])]
    for u, g in keys:
        te = (P.day == g).values & ((P.bc == u).values if u else True)
        tr = (P.day != g).values & ((P.bc != u).values if u else True)
        T = P[tr].reset_index(drop=True)
        p = fit_spec(T, spec)(P[te])
        pred[te] = np.where(P.cens.values[te] & (p > DL_LOG), DL_LOG, p)
        for thr in (1, 10):
            cut[thr][te] = nested_cut_spec(T, spec, thr)
    return pred, cut


def qwk(a, b, k=3):
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    W = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (W * O).sum() / (W * E).sum())


def step5():
    B, P = load_final()
    y = np.log10(P.cbt.values + 1)
    out = {}
    shared = dict(feats=D0B, fe=False)
    samples = {}
    for scheme in ('new sensor + new day', 'day held out'):
        pred, cut = cv_spec(P, shared, scheme)
        s = sample_level(P, y, pred, P.cens.values, cut, P.grp.values)
        sy, sp_, sc, scbt = s.y.values, s.pred.values, s.cens.values, s.cbt.values
        scut = {1: s.c1.values, 10: s.c10.values}
        ss = score(sy, sp_, sc, scbt, scut)
        ci = day_bootstrap(s.day.values, lambda idx: score(sy[idx], sp_[idx], sc[idx], scbt[idx], {t: scut[t][idx] for t in (1, 10)}))
        rmse = float(np.sqrt(((sy - sp_) ** 2).mean()))
        out[scheme] = dict(metrics=flat(ss), ci=ci, rmse=rmse, n=len(s),
                           counts={t: {k: ss[f'ge{t}'][k] for k in ('tp', 'fp', 'tn', 'fn')} for t in (1, 10)},
                           ia=ia_rows(scbt, 10 ** sp_ - 1, sc.astype(bool), s.day.values))
        s['pm'] = 10 ** s.pred - 1
        samples[scheme] = s
        P[f'pred_{scheme}'] = pred; P[f'cut10_{scheme}'] = cut[10]

    # reference uncertainty (methods paper 2.6): expected and reference-tolerant performance
    iv = P.drop_duplicates('bag').apply(lambda r: pd.Series(cbt_interval(r.cbt, r.cens), index=['lo', 'hi', 'p10']), axis=1)
    rec = P.drop_duplicates('bag')[['grp']].join(iv)
    ref = rec.groupby('grp').agg(p10=('p10', 'mean'), lo=('lo', 'min'), hi=('hi', 'max'))
    refunc = {}
    for scheme, s in samples.items():
        r = s.join(ref)
        yhat = (r.pm >= r.c10).astype(float).values
        p = r.p10.values
        obs = (r.cbt >= 10).values
        exp_sens = float((p * yhat).sum() / p.sum()); exp_spec = float(((1 - p) * (1 - yhat)).sum() / (1 - p).sum())
        wrong_pos = (yhat == 1) & (r.hi.values < 10)          # flagged, reference interval entirely below 10
        wrong_neg = (yhat == 0) & (r.lo.values >= 10)         # not flagged, reference interval entirely at/above 10
        tol_sens = float(1 - wrong_neg[obs].sum() / obs.sum()); tol_spec = float(1 - wrong_pos[~obs].sum() / (~obs).sum())
        refunc[scheme] = dict(expected=dict(sens=exp_sens, spec=exp_spec, ba=(exp_sens + exp_spec) / 2),
                              reference_tolerant_upper_bound=dict(sens=tol_sens, spec=tol_spec, ba=(tol_sens + tol_spec) / 2))
    out['reference_uncertainty'] = refunc

    # WHO three-level categories: <10, 10-99, >=100 (>=10 decision by the in-fold cut, >=100 at 100)
    who = {}
    for scheme, s in samples.items():
        o = np.select([s.cbt >= 100, s.cbt >= 10], [2, 1], 0)
        pr = np.select([s.pm >= 100, s.pm >= s.c10], [2, 1], 0)
        M = pd.crosstab(pd.Series(o, name='CBT'), pd.Series(pr, name='predicted')).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
        recall = [float(M.iloc[i, i] / M.iloc[i].sum()) if M.iloc[i].sum() else np.nan for i in range(3)]
        prec = [float(M.iloc[i, i] / M.iloc[:, i].sum()) if M.iloc[:, i].sum() else np.nan for i in range(3)]
        who[scheme] = dict(matrix=M.values.tolist(), recall=recall, precision=prec,
                           accuracy=float(np.trace(M.values) / M.values.sum()), macro_recall=float(np.nanmean(recall)), qwk=qwk(o, pr))
    out['who_three_level'] = who

    # reference repeatability: replicates within a bucket, repeat samples of a source in separate buckets
    Bg = B.dropna(subset=['sample_group']).copy(); Bg['t'] = pd.to_datetime(Bg.t_utc, utc=True); Bg['site'] = Bg.site.fillna('')
    Bg = Bg[Bg.bag_id.isin(P.bag)]
    reps = Bg.groupby('sample_group').cbt.agg(list); reps = reps[reps.apply(len) > 1]
    G = Bg.groupby('sample_group').agg(site=('site', 'first'), t=('t', 'min'), gm=('cbt', lambda c: 10 ** np.log10(np.array(c) + 1).mean() - 1))
    rp = []
    for site_, g in G[G.site != ''].groupby('site'):
        g = g.sort_values('t')
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if (g.t.iloc[j] - g.t.iloc[i]).total_seconds() / 60 <= 30:
                    rp.append(((g.gm.iloc[i] >= 10) == (g.gm.iloc[j] >= 10)))
    s0 = samples['new sensor + new day']
    out['reference_repeatability'] = dict(
        replicates_within_bucket=dict(groups=int(len(reps)), agree_ge10=int(sum(len({v >= 10 for v in c}) == 1 for c in reps))),
        repeats_same_source=dict(pairs=len(rp), agree_ge10=int(sum(rp))),
        sensor_vs_cbt_ge10=dict(samples=len(s0), agree=int(((s0.pm >= s0.c10) == (s0.cbt >= 10)).sum())))

    # comparison and sensitivity models on the final records
    comp = {
        'A per-unit intercepts + per-unit F slopes': dict(feats=['F', 'Blog'], fe=True, unit_slopes=['F']),
        'B per-unit intercepts, shared slopes': dict(feats=['F', 'Blog'], fe=True),
        'E Eq. 2 per unit': dict(feats=D0B, fe=True, per_unit_all=True),
        'D0a shared, ToF as is': dict(feats=['F', 'B', 'Tc', 'FT'], fe=False),
        'D0b shared, ToF log ratio (primary)': dict(feats=D0B, fe=False),
        'D0c shared, no ToF': dict(feats=['F', 'Tc', 'FT'], fe=False),
        'Cb shared: F, ToF log ratio': dict(feats=['F', 'Blog'], fe=False),
    }
    cm = {}
    for name, spec in comp.items():
        for scheme in (('day held out',) if spec['fe'] else ('day held out', 'new sensor + new day')):
            pred, cut = cv_spec(P, spec, scheme)
            s = sample_level(P, y, pred, P.cens.values, cut, P.grp.values)
            ss = score(s.y.values, s.pred.values, s.cens.values, s.cbt.values, {1: s.c1.values, 10: s.c10.values})
            ia = ia_rows(s.cbt.values, 10 ** s.pred.values - 1, s.cens.values.astype(bool), s.day.values)['all pairs']
            cm[f'{name} | {scheme}'] = dict(ba=ss['ge10']['ba'], sens=ss['ge10']['sens'], spec=ss['ge10']['spec'],
                                          ppv=ss['ge10']['ppv'], r2=ss['r2'], ia=ia['ia'])
    out['comparison_models'] = cm

    # chlorination (as the submitted paper): chlorinated records, and matched source/treated systems
    P['cl'] = pd.to_numeric(P.chlorine, errors='coerce')
    key = 'new sensor + new day'
    Ps = P.merge(samples[key][['pm', 'c10']], left_on='grp', right_index=True)
    rec_ = Ps.drop_duplicates('bag')
    chl = rec_[rec_.cl > 0]
    rec_ = rec_.assign(system=rec_.site.fillna('').astype(str).apply(lambda s: '.'.join(s.split('.')[:4]) if s.count('.') >= 3 else s),
                       src=rec_.water_type.str.startswith('Source'))
    ms = []
    for (sysn, day), g in rec_.groupby(['system', 'day']):
        so, tr_ = g[g.src & (g.cl == 0)], g[~g.src & (g.cl > 0)]
        if len(so) and len(tr_) and sysn:
            ms.append(dict(system=sysn[-40:], day=day, n_source=len(so), n_treated=len(tr_),
                           delta_log_pred=float(np.log10(so.pm.clip(lower=0.1)).mean() - np.log10(tr_.pm.clip(lower=0.1)).mean())))
    out['chlorination'] = dict(n_chlorinated=int(len(chl)), cbt_zero=int((chl.cbt == 0).sum()),
                               predicted_below_10=int((chl.pm < chl.c10).sum()), matched_systems=ms)

    # cross-sensor: samples read by two units, each unit's own held-out prediction
    two = P.groupby('grp').bc.nunique(); two = two[two > 1].index
    d = P[P.grp.isin(two)].groupby(['grp', 'bc'])[f'pred_{key}'].mean().unstack()
    diffs = []
    for g, r in d.iterrows():
        v = r.dropna().values
        if len(v) == 2:
            diffs.append(abs(v[0] - v[1]))
    diffs = np.array(diffs)
    out['cross_sensor'] = dict(samples=int(len(diffs)), mean_abs_delta_log=float(diffs.mean()), within_0p65=int((diffs <= 0.65).sum()))

    json.dump(out, open(REV / 'step5.json', 'w'), indent=2, default=float)
    for scheme, s in samples.items():
        s.to_csv(REV / f'final_samples_{scheme.replace(" ", "_").replace("+", "and")}.csv')
    P.to_csv(REV / 'final_pairs.csv', index=False)
    print(json.dumps({k: v for k, v in out.items() if k != 'comparison_models'}, indent=1, default=lambda o: round(float(o), 3))[:6000])
    print('\ncomparison models:')
    print(pd.DataFrame(cm).T.astype(float).round(3).to_string())


# ---- export for the paper: paper_data.json and paired_observations.csv ----------------
def step_export():
    o = json.load(open(REV / 'step5.json'))
    FP = pd.read_csv(REV / 'final_pairs.csv')
    sn = pd.read_csv(REV / 'final_samples_new_sensor_and_new_day.csv', index_col=0)
    sd = pd.read_csv(REV / 'final_samples_day_held_out.csv', index_col=0)
    FP['country'] = np.where(FP.day < '2026-06-01', 'Rwanda', 'Kenya')
    info = FP.groupby('grp').agg(day=('day', 'first'), country=('country', 'first'), water_type=('water_type', 'first'),
                                 site=('site', 'first'), units=('bc', lambda s: sorted(set(map(str, s)))),
                                 n_records=('bag', 'nunique'))
    samples = []
    for g, r in sn.iterrows():
        samples.append(dict(sample=g, day=r.day, country=info.loc[g, 'country'],
                            water_type='Treated' if str(info.loc[g, 'water_type']).startswith('Treated') else 'Source',
                            site=None if pd.isna(info.loc[g, 'site']) else str(info.loc[g, 'site']),
                            units=info.loc[g, 'units'], n_records=int(info.loc[g, 'n_records']),
                            cbt=float(r.cbt), censored=bool(r.cens), obs_log=float(r.y),
                            pred_log=float(r.pred), cut1=float(r.c1), cut10=float(r.c10),
                            pred_log_day=float(sd.loc[g, 'pred']), cut10_day=float(sd.loc[g, 'c10'])))
    paper = dict(
        model=dict(name='D0b', response='log10(CBT MPN/100 mL + 1), right-censored at 100 (Tobit)',
                   features=['F: temperature-corrected ln(mon2-170) minus the daily clean-water baseline',
                             'Blog: ln(ToF / daily clean-water ToF)', 'T: temperature - 20 C', 'F x T'],
                   coefficients='shared across units', ridge=0.1, per_unit_calibration='daily clean-water baseline only'),
        n_samples=len(samples), n_pairs=int(len(FP)), n_records=int(FP.bag.nunique()),
        results=o, samples=samples)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(paper, open(PAPER_DIR / 'paper_data.json', 'w'), indent=1, default=float)
    cols = dict(bag='record_id', grp='sample_group', bc='sensor', sample_local='sample_time_local', t_utc='sample_time_utc',
                site='site_name', water_type='water_type', country='country', cbt='cbt_mpn_per_100ml', censored='cbt_censored',
                chlorine='chlorine_residual_mg_l', mon2='sensor_mon2', tof='sensor_tof_kcps', Tch='sensor_temperature_c',
                T_channel='temperature_channel', F='F_corrected_fluorescence', Blog='B_log_tof_ratio', base_age='baseline_age_days',
                **{'pred_new sensor + new day': 'pred_log10_new_sensor_new_day', 'pred_day held out': 'pred_log10_day_held_out'})
    FP[[c for c in cols if c in FP.columns]].rename(columns=cols).to_csv(PAPER_DIR / 'paired_observations.csv', index=False)
    print('wrote', PAPER_DIR / 'paper_data.json', 'and paired_observations.csv |', len(samples), 'samples,', len(FP), 'pairs')


if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['0', '1', '23']
    if '0' in which: rung0()
    if '1' in which: rung1()
    if '23' in which: rung23()
    if '4' in which: step4()
    if 'ia' in which: step_ia()
    if 'deploy' in which: step_deploy()
    if 'air' in which: step_air()
    if 'bridge' in which: step_bridge()
    if 'deploy-interp' in which:
        BRIDGE = 'interp'
        step_deploy()
    if 'export' in which:
        step_export()
    if 'step5' in which:
        BRIDGE = 'interp'
        step5()
    if 'tof' in which:
        BRIDGE = 'interp'
        step_deploy(TOF_VARIANTS, 'step_tof.json')
