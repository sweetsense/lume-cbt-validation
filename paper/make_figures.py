#!/usr/bin/env python3
"""
Figures for the revised CBT fluorimeter validation paper.

Reads paper_data.json, written by lume-validation/scripts/cbt_revision/evaluate.py (export),
and draws, from held-out predictions of the primary model (a new sensor on a new day):
  - fig_cbt_histogram.pdf      CBT results by sample, WHO/UNICEF categories
  - fig_scatter.pdf            predicted vs observed, per sample
  - fig_confusion_binary.pdf   decisions at >=1 and >=10 MPN/100 mL
  - fig_confusion_3level.pdf   three-level WHO categories
and writes the LaTeX tables:
  - table_classification.tex   decisions at 1 and 10 MPN/100 mL, both validation schemes, with CIs
  - table_models.tex           comparison models

Style: Water Research, 3.5-inch single column, 300 DPI, PDF.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

OUTDIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(OUTDIR, 'paper_data.json')
COL_WIDTH, DOUBLE_COL, DPI = 3.5, 7.0, 300
SCHEME = 'new sensor + new day'

AGREE_COLOR = '#228833'
DISAGREE_COLOR = '#CC3311'
GROUPS = [  # country, water type, label, colour (same as the histogram)
    ('Rwanda', 'Source', 'Rwanda, source', '#1d4ed8'),
    ('Rwanda', 'Treated', 'Rwanda, treated', '#60a5fa'),
    ('Kenya', 'Source', 'Kenya, source', '#b45309'),
    ('Kenya', 'Treated', 'Kenya, treated', '#fbbf24'),
]
BAND = 1.96 * np.sqrt(2 * (0.65 / 1.96) ** 2)   # combined agreement band, ~0.92 log10
THR10 = np.log10(11)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7, 'figure.dpi': DPI, 'savefig.dpi': DPI, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 3, 'ytick.major.size': 3, 'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def load():
    with open(DATA_FILE) as f:
        return json.load(f)


def save(fig, name):
    out = os.path.join(OUTDIR, name)
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved: {out}')


def fig_cbt_histogram(d):
    """Samples by WHO/UNICEF category of their CBT result (geometric mean of replicate records)."""
    S = d['samples']
    bins = [('0\n(Conformity)', lambda v: v == 0), ('>0 to <10\n(Low)', lambda v: 0 < v < 10),
            ('10–99\n(Intermediate)', lambda v: 10 <= v < 100), (r'$\geq$100' + '\n(Very high)', lambda v: v >= 100)]
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.9))
    x = np.arange(len(bins)); bottom = np.zeros(len(bins))
    for country, water, label, color in GROUPS:
        yc = np.array([sum(1 for s in S if s['country'] == country and s['water_type'] == water and t(round(s['cbt'], 6)))
                       for _, t in bins])
        ax.bar(x, yc, bottom=bottom, color=color, edgecolor='black', linewidth=0.4, width=0.72, label=label)
        bottom += yc
    for xi, tot in zip(x, bottom):
        ax.text(xi, tot + bottom.max() * 0.015, str(int(tot)), ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bins])
    ax.set_xlabel('WHO/UNICEF risk category (CBT, MPN/100 mL)')
    ax.set_ylabel('Samples')
    ax.set_ylim(0, bottom.max() * 1.16)
    ax.legend(loc='upper right', frameon=False, fontsize=6.5, handlelength=1.1, labelspacing=0.3)
    save(fig, 'fig_cbt_histogram.pdf')


def fig_scatter(d):
    """Held-out predictions (a new sensor on a new day) against CBT, one point per sample."""
    S = d['samples']
    R = d['results'][SCHEME]
    obs = np.array([s['obs_log'] for s in S]); pred = np.array([s['pred_log'] for s in S])
    cens = np.array([s['censored'] for s in S])
    rng = np.random.RandomState(42)
    ox = obs + rng.uniform(-0.03, 0.03, len(obs))
    lo, hi = -0.6, 2.6
    pred_d = np.clip(pred, lo + 0.05, hi - 0.05)
    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH + 0.55))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    xx = np.linspace(lo, hi, 200)
    ax.fill_between(xx, xx - BAND, xx + BAND, color='#EEEEEE', zorder=1)
    ax.plot([lo, hi], [lo, hi], '-', color='#555555', linewidth=0.8, zorder=2)
    ax.axvline(THR10, color='#999999', linestyle=':', linewidth=0.7, zorder=2)
    ax.axhline(THR10, color='#999999', linestyle=':', linewidth=0.7, zorder=2)
    for country, water, label, color in GROUPS:
        m = np.array([s['country'] == country and s['water_type'] == water for s in S])
        for c_, mk, sz in ((False, 'o', 26), (True, '>', 30)):
            mm = m & (cens == c_)
            if mm.any():
                ax.scatter(ox[mm], pred_d[mm], s=sz, c=color, marker=mk, edgecolors='black', linewidths=0.35, zorder=3)
    ax.set_xlabel('Observed CBT (log$_{10}$(MPN/100 mL + 1))')
    ax.set_ylabel('Predicted, held out (log$_{10}$(MPN/100 mL + 1))')
    ia = R['ia']['all pairs']['ia']
    ax.text(0.04, 0.96, f"$n$ = {R['n']} samples\n$R^2$ = {R['metrics']['r2']:.2f}\n"
            f"MAE = {R['metrics']['mae']:.2f} log$_{{10}}$\nIA = {ia:.2f}",
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.95))
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markeredgecolor='black',
                      markeredgewidth=0.35, markersize=5.5, label=l) for _, _, l, c in GROUPS]
    handles.append(Line2D([0], [0], marker='>', color='w', markerfacecolor='#888888', markeredgecolor='black',
                          markeredgewidth=0.35, markersize=5.5, label=r'CBT $\geq$100 (censored)'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False,
              fontsize=6.3, handletextpad=0.3, columnspacing=0.9)
    fig.subplots_adjust(bottom=0.2)
    save(fig, 'fig_scatter.pdf')


def _cell(ax, j, i, val, total, good, maxv):
    rgba = list(to_rgba(AGREE_COLOR if good else DISAGREE_COLOR))
    rgba[3] = 0.15 + 0.5 * (val / max(maxv, 1)) if (good or val) else 0
    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=rgba if val or good else 'white', edgecolor='white', linewidth=2))
    ax.text(j + 0.5, i + 0.58, str(val), ha='center', va='center', fontsize=13, fontweight='bold',
            color='#333333' if val else '#bbbbbb')
    if val:
        ax.text(j + 0.5, i + 0.3, f'({100 * val / total:.0f}%)', ha='center', va='center', fontsize=7, color='#666666')


def fig_confusion_binary(d):
    R = d['results'][SCHEME]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, COL_WIDTH * 1.55))
    for ax, t in zip(axes, ('1', '10')):
        c = R['counts'][t]
        mat = [[c['tn'], c['fn']], [c['fp'], c['tp']]]   # rows: predicted <thr, >=thr; cols: true <thr, >=thr
        total = sum(map(sum, mat)); maxv = max(map(max, mat))
        for i in range(2):
            for j in range(2):
                _cell(ax, j, 1 - i, mat[i][j], total, i == j, maxv)
        ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect('equal')
        ax.set_xticks([0.5, 1.5]); ax.set_xticklabels([f'<{t}', f'≥{t}'])
        ax.set_yticks([0.5, 1.5]); ax.set_yticklabels([f'≥{t}', f'<{t}'])
        ax.set_xlabel('CBT (MPN/100 mL)', labelpad=6)
        if t == '1':
            ax.set_ylabel('Predicted, held out', labelpad=6)
        ax.set_title(f'≥{t} MPN/100 mL', pad=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        m, ci = R['metrics'], R['ci']
        k = f'ge{t}'
        txt = (f"Balanced accuracy {m[k + '_ba']:.2f} ({ci[k + '_ba'][0]:.2f}–{ci[k + '_ba'][1]:.2f})\n"
               f"Sensitivity {c['tp']}/{c['tp'] + c['fn']}    Specificity {c['tn']}/{c['tn'] + c['fp']}\n"
               f"PPV {m[k + '_ppv']:.2f}    NPV {m[k + '_npv']:.2f}")
        ax.text(0.5, -0.2, txt, ha='center', va='top', fontsize=7, transform=ax.transAxes)
    fig.subplots_adjust(wspace=0.4, bottom=0.28, top=0.9)
    save(fig, 'fig_confusion_binary.pdf')


def fig_confusion_3level(d):
    W = d['results']['who_three_level'][SCHEME]
    M = np.array(W['matrix']).T                       # rows: predicted, cols: true
    labels = ['<10', '10–99', '≥100']; risk = ['Low risk', 'Intermediate risk', 'High risk']
    K, total, maxv = 3, M.sum(), M.max()
    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 1.15))
    for i in range(K):
        for j in range(K):
            _cell(ax, j, K - 1 - i, int(M[i, j]), total, i == j, maxv)
    ax.set_xlim(0, K); ax.set_ylim(0, K); ax.set_aspect('equal')
    ax.set_xticks([i + 0.5 for i in range(K)]); ax.set_xticklabels(labels)
    ax.set_yticks([i + 0.5 for i in range(K)]); ax.set_yticklabels(list(reversed(labels)))
    ax.set_xlabel('CBT (MPN/100 mL)', labelpad=8); ax.set_ylabel('Predicted, held out', labelpad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    for j, rl in enumerate(risk):
        ax.text(j + 0.5, K + 0.12, rl, ha='center', va='bottom', fontsize=6, color='#888888', style='italic')
    diag = int(np.trace(M))
    fig.text(0.45, 0.03, f'Correct category: {diag}/{total} ({diag / total:.0%})    '
             f'Quadratic weighted kappa {W["qwk"]:.2f}', ha='center', va='bottom', fontsize=7.5)
    fig.subplots_adjust(bottom=0.18, top=0.88)
    save(fig, 'fig_confusion_3level.pdf')


def _num(x):
    return f'$-${-x:.2f}' if x < 0 else f'{x:.2f}'


def _table(name, caption, label, colspec, header, rows):
    body = '\n'.join('    ' + ' & '.join(r) + ' \\\\' for r in rows)
    tex = (f'\\begin{{table}}[H]\n\\centering\n\\footnotesize\n\\setlength{{\\tabcolsep}}{{4pt}}\n'
           f'\\caption{{{caption}}}\n\\label{{{label}}}\n\\resizebox{{\\linewidth}}{{!}}{{%\n\\begin{{tabular}}{{{colspec}}}\n\\hline\n'
           f'{header}\n\\hline\n{body}\n\\hline\n\\end{{tabular}}}}\n\\end{{table}}\n')
    with open(os.path.join(OUTDIR, name), 'w') as f:
        f.write(tex)
    print(f'  Saved: {os.path.join(OUTDIR, name)}')


def table_classification(d):
    R = d['results']
    cols = [(thr, scheme) for thr in (10, 1) for scheme in ('new sensor + new day', 'day held out')]
    rows = [['Detected'] + [f"{R[s]['counts'][str(t)]['tp']}/{R[s]['counts'][str(t)]['tp'] + R[s]['counts'][str(t)]['fn']}" for t, s in cols],
            ['Correct below'] + [f"{R[s]['counts'][str(t)]['tn']}/{R[s]['counts'][str(t)]['tn'] + R[s]['counts'][str(t)]['fp']}" for t, s in cols]]
    for k, lab in (('sens', 'Sensitivity'), ('spec', 'Specificity'), ('ppv', 'PPV'), ('npv', 'NPV'), ('ba', 'Balanced accuracy')):
        rows.append([lab] + [f"{_num(R[s]['metrics'][f'ge{t}_{k}'])} ({_num(R[s]['ci'][f'ge{t}_{k}'][0])}--{_num(R[s]['ci'][f'ge{t}_{k}'][1])})" for t, s in cols])
    header = ('    & \\multicolumn{2}{c}{10~MPN/100~mL} & \\multicolumn{2}{c}{1~MPN/100~mL} \\\\\n'
              '    & New sensor, new day & Day held out & New sensor, new day & Day held out \\\\')
    _table('table_classification.tex',
           'Classification at 1 and 10~MPN/100~mL for the 75 samples under both validation schemes. Detected: samples at or '
           'above the threshold that were flagged, of all samples at or above it. Correct below: samples below the threshold '
           'that were not flagged, of all samples below it. Values in parentheses are 95\\% confidence intervals from 2000 '
           'bootstrap resamples of sampling days. PPV and NPV, positive and negative predictive value.',
           'tab:classification', 'lcccc', header, rows)


MODEL_NAMES = [
    ('D0b', 'Shared: $F$, $B$, $T$, $F \\times T$ (primary)'),
    ('D0a', 'Shared: ToF as a difference from baseline'),
    ('Cb', 'Shared: $F$, $B$'),
    ('D0c', 'Shared: $F$, $T$, $F \\times T$ (no ToF)'),
    ('A', 'Per-unit intercepts and $F$ slopes: $F$, $B$'),
    ('B', 'Per-unit intercepts: $F$, $B$'),
    ('E', 'Every term per unit: $F$, $B$, $T$, $F \\times T$'),
]


def table_models(d):
    cm = d['results']['comparison_models']
    rows = []
    for key, name in MODEL_NAMES:
        first = True
        for scheme, lab in (('new sensor + new day', 'New sensor, new day'), ('day held out', 'Day held out')):
            k = next((k for k in cm if k.split(' ')[0] == key and k.endswith('| ' + scheme)), None)
            if k:
                m = cm[k]
                rows.append([name if first else '', lab] + [_num(m[x]) for x in ('ba', 'sens', 'spec', 'ppv', 'r2', 'ia')])
                first = False
    _table('table_models.tex',
           'Comparison models at 10~MPN/100~mL. Models with unit-specific terms can be evaluated only with the day held out. '
           '$F$ is corrected fluorescence and $B$ the ToF log ratio, each relative to the day\'s baseline, and $T$ is temperature. '
           'BA, balanced accuracy; PPV, positive predictive value; IA, index of agreement.',
           'tab:models', 'llcccccc',
           '    Model & Validation & BA & Sensitivity & Specificity & PPV & $R^2$ & IA \\\\', rows)


if __name__ == '__main__':
    data = load()
    print(f"{data['n_samples']} samples, {data['n_pairs']} sensor-CBT pairs")
    fig_cbt_histogram(data)
    fig_scatter(data)
    fig_confusion_binary(data)
    fig_confusion_3level(data)
    table_classification(data)
    table_models(data)
