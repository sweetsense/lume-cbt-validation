#!/usr/bin/env python3
"""
Generate publication-quality figures for the CBT fluorimeter validation paper.

Reads paper_data.json (exported by export-paper-data.js) and produces:
  - fig_scatter.pdf      (Figure 1: regression scatter)
  - fig_confusion_binary.pdf  (Figure 2: binary confusion matrices)
  - fig_temperature.pdf  (Figure 3: temperature correction)
  - fig_confusion_4level.pdf  (Figure 4: 4-level WHO confusion matrix)

Style: Water Research journal, 3.5-inch single-column, 300 DPI, PDF output.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── Configuration ──
OUTDIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(OUTDIR, 'paper_data.json')

# Water Research: single-column = 3.5 in (90 mm)
COL_WIDTH = 3.5
DOUBLE_COL = 7.0
DPI = 300

# Colorblind-safe palette (Tol bright)
SENSOR_COLORS = {
    '50045': '#4477AA',  # blue
    '50053': '#EE6677',  # rose/orange
    '50065': '#228833',  # green
}
AGREE_COLOR = '#228833'   # green
DISAGREE_COLOR = '#CC3311' # red
BAND_COLOR = '#CCEE88'    # light green for agreement band

# ── Style setup ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 0.8,
    'pdf.fonttype': 42,  # TrueType fonts in PDF
    'ps.fonttype': 42,
})


def load_data():
    with open(DATA_FILE) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════
# Figure 1: Regression scatter plot
# ════════════════════════════════════════════════════════════════════
def fig_scatter(data):
    pts = data['paired_points']
    model = data['model']
    band = model['agreement_band']  # ~0.92 log10

    obs = np.array([p['observed_log'] for p in pts])
    pred = np.array([p['predicted_log'] for p in pts])
    barcodes = [p['barcode'] for p in pts]
    censored = np.array([p['censored'] for p in pts])
    agree = np.array([p['agree'] for p in pts])

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH))

    # Axis range
    lo, hi = -0.05, 2.5
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # Agreement band (shading around 1:1 line)
    xx = np.linspace(lo, hi, 200)
    ax.fill_between(xx, xx - band, xx + band, color=BAND_COLOR, alpha=0.4,
                     zorder=1, label=f'Agreement band ($\\pm${band:.2f} log$_{{10}}$)')

    # 1:1 reference line
    ax.plot([lo, hi], [lo, hi], '--', color='#888888', linewidth=0.7, zorder=2)

    # Plot points by sensor, colored by agreement
    for bc in data['sensors']:
        mask_bc = np.array([b == bc for b in barcodes])
        color = SENSOR_COLORS.get(bc, '#999999')

        # Uncensored, agree
        m = mask_bc & agree & ~censored
        if m.any():
            ax.scatter(obs[m], pred[m], s=18, c=color, marker='o',
                       edgecolors='none', alpha=0.7, zorder=3)

        # Uncensored, disagree
        m = mask_bc & ~agree & ~censored
        if m.any():
            ax.scatter(obs[m], pred[m], s=22, c=DISAGREE_COLOR, marker='x',
                       linewidths=0.7, alpha=0.8, zorder=4)

        # Censored, agree (rightward arrow)
        m = mask_bc & agree & censored
        if m.any():
            ax.scatter(obs[m], pred[m], s=22, c=color, marker='>',
                       edgecolors='none', alpha=0.7, zorder=3)

        # Censored, disagree
        m = mask_bc & ~agree & censored
        if m.any():
            ax.scatter(obs[m], pred[m], s=22, c=DISAGREE_COLOR, marker='>',
                       edgecolors='none', alpha=0.8, zorder=4)

    # Axis labels
    ax.set_xlabel('Observed CBT (log$_{10}$(CFU + 1))')
    ax.set_ylabel('Predicted fluorimeter (log$_{10}$(CFU + 1))')

    # Inset text: metrics
    r2 = model['r2']
    mae_val = model['mae']
    n = data['n']
    textstr = f'$R^2$ = {r2:.3f}\nMAE = {mae_val:.3f} log$_{{10}}$\n$n$ = {n}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SENSOR_COLORS['50045'],
               markersize=5, label='50045'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SENSOR_COLORS['50053'],
               markersize=5, label='50053'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SENSOR_COLORS['50065'],
               markersize=5, label='50065'),
        Line2D([0], [0], marker='x', color=DISAGREE_COLOR, linestyle='None',
               markersize=5, label='Disagreement'),
        Line2D([0], [0], marker='>', color='w', markerfacecolor='#666666',
               markersize=5, label='Right-censored'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              framealpha=0.9, edgecolor='#cccccc', fontsize=6.5)

    out = os.path.join(OUTDIR, 'fig_scatter.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved: {out}')


# ════════════════════════════════════════════════════════════════════
# Figure 2: Binary confusion matrices (side-by-side)
# ════════════════════════════════════════════════════════════════════
def fig_confusion_binary(data):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, COL_WIDTH * 1.3))

    for ax_idx, key in enumerate(['ge1', 'ge10']):
        ax = axes[ax_idx]
        res = data['binary_classification'][key]
        cm = res['confusion_matrix']
        thr = res['threshold']
        # Standard layout: rows = true, cols = predicted
        # Row 0 = negative (<thr), Row 1 = positive (>=thr)
        # [[TN, FP], [FN, TP]]
        mat = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
        total = mat.sum()

        # Color map: green diagonal (correct), red off-diagonal (errors)
        colors = np.array([
            [AGREE_COLOR, DISAGREE_COLOR],
            [DISAGREE_COLOR, AGREE_COLOR],
        ])

        # Draw cells: grid spans y=0..2, with row 0 (negative) at top
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                pct = 100 * val / total
                bg = colors[i, j]
                from matplotlib.colors import to_rgba
                rgba = list(to_rgba(bg))
                max_val = mat.max()
                intensity = 0.15 + 0.45 * (val / max(max_val, 1))
                rgba[3] = intensity
                # Cell at column j, row (1-i) so row 0 is at top
                ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                             facecolor=rgba, edgecolor='white', linewidth=2))
                cy = 1.5 - i  # cell center y
                ax.text(j + 0.5, cy + 0.08, str(val),
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        color='#333333')
                ax.text(j + 0.5, cy - 0.18, f'({pct:.1f}%)',
                        ha='center', va='center', fontsize=8, color='#666666')

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels([f'<{thr}', f'\u2265{thr}'], fontsize=9)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels([f'\u2265{thr}', f'<{thr}'], fontsize=9)
        ax.set_xlabel('Predicted class', fontsize=9, labelpad=6)
        if ax_idx == 0:
            ax.set_ylabel('True class', fontsize=9, labelpad=6)

        ax.set_title(f'\u2265{thr} CFU/100 mL threshold', fontsize=9, pad=12)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(length=0)

        # Metrics below the matrix -- place with enough clearance from xlabel
        bal = res['balanced_accuracy']
        sens = res['sensitivity']
        spec = res['specificity']
        auc = res['auc']
        metrics = (f'Bal. acc. = {bal:.0%}   Sens. = {sens:.0%}\n'
                   f'Spec. = {spec:.0%}   AUC = {auc:.3f}')
        ax.text(0.5, -0.18, metrics, ha='center', va='top', fontsize=7,
                transform=ax.transAxes)

    fig.subplots_adjust(wspace=0.4, bottom=0.25, top=0.88)

    out = os.path.join(OUTDIR, 'fig_confusion_binary.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved: {out}')


# ════════════════════════════════════════════════════════════════════
# Figure 3: Temperature correction (clean water)
# ════════════════════════════════════════════════════════════════════
def fig_temperature(data):
    temp_data = data['temperature']
    clean = temp_data['clean_water_data']
    rho = temp_data['rho']
    r2 = temp_data['r2']
    n_clean = temp_data['n']

    temps = np.array([d['temp'] for d in clean])
    mon2 = np.array([d['mon2'] for d in clean])
    barcodes_c = [d['barcode'] for d in clean]

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.85))

    # Points colored by sensor
    for bc in data['sensors']:
        mask = np.array([b == bc for b in barcodes_c])
        if mask.any():
            ax.scatter(temps[mask], mon2[mask], s=18,
                       c=SENSOR_COLORS.get(bc, '#999999'), marker='o',
                       edgecolors='none', alpha=0.6, label=bc, zorder=3)

    # Fitted exponential decay curves (per sensor, shared rho)
    t_range = np.linspace(temps.min() - 1, temps.max() + 1, 200)
    per_sensor = temp_data['per_sensor']
    bcs_sorted = sorted(per_sensor.keys())
    for bc in bcs_sorted:
        intercept = per_sensor[bc]['intercept']
        # Model: log(mon2) = intercept - rho*(temp-20)
        # => mon2 = exp(intercept) * exp(-rho*(temp-20))
        fitted = np.exp(intercept) * np.exp(-rho * (t_range - 20))
        ax.plot(t_range, fitted, '-', color=SENSOR_COLORS.get(bc, '#999999'),
                linewidth=1.0, alpha=0.8, zorder=2)

    ax.set_xlabel('Water temperature (\u00b0C)')
    ax.set_ylabel('Raw mon2 fluorescence signal (a.u.)')

    # Inset text
    textstr = (f'$\\rho$ = {rho:.4f} \u00b0C$^{{-1}}$\n'
               f'$R^2$ = {r2:.3f}\n'
               f'$n$ = {n_clean} (CBT = 0 samples)')
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
            fontsize=7, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    ax.legend(loc='lower left', frameon=True, framealpha=0.9,
              edgecolor='#cccccc', fontsize=7)

    out = os.path.join(OUTDIR, 'fig_temperature.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved: {out}')


# ════════════════════════════════════════════════════════════════════
# Figure 4: 4-level WHO confusion matrix
# ════════════════════════════════════════════════════════════════════
def fig_confusion_4level(data):
    fl = data['four_level']
    mat = np.array(fl['matrix'])
    labels = fl['labels']
    who_labels = fl['who_labels']
    total = mat.sum()

    fig, ax = plt.subplots(figsize=(COL_WIDTH * 1.2, COL_WIDTH * 1.35))

    # Grid spans x=0..4, y=0..4, row 0 (top visual) = first observed category
    max_val = mat.max()
    from matplotlib.colors import to_rgba
    for i in range(4):
        for j in range(4):
            val = mat[i, j]
            pct = 100 * val / total if total > 0 else 0

            if i == j:
                intensity = 0.15 + 0.55 * (val / max(max_val, 1))
                rgba = list(to_rgba(AGREE_COLOR))
                rgba[3] = intensity
            else:
                if val == 0:
                    rgba = [1, 1, 1, 1]
                else:
                    intensity = 0.10 + 0.40 * (val / max(max_val, 1))
                    rgba = list(to_rgba(DISAGREE_COLOR))
                    rgba[3] = intensity

            # Row i at y = (3-i) to (4-i), so row 0 is at top
            ax.add_patch(plt.Rectangle((j, 3 - i), 1, 1,
                         facecolor=rgba, edgecolor='white', linewidth=1.5))

            cy = 3.5 - i
            fontcolor = '#333333' if val > 0 else '#bbbbbb'
            ax.text(j + 0.5, cy + 0.08, str(val),
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color=fontcolor)
            if val > 0:
                ax.text(j + 0.5, cy - 0.18, f'({pct:.1f}%)',
                        ha='center', va='center', fontsize=6.5, color='#666666')

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(list(reversed(labels)), fontsize=7)

    ax.set_xlabel('Predicted category', fontsize=9, labelpad=10)
    ax.set_ylabel('Observed category', fontsize=9, labelpad=8)

    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(length=0)

    # WHO risk category names on the right side
    for i, wl in enumerate(who_labels):
        ax.text(4.15, 3.5 - i, wl, ha='left', va='center', fontsize=5.5,
                color='#888888', style='italic')

    # Accuracy metrics below the xlabel
    diag_sum = sum(mat[i, i] for i in range(4))
    overall_acc = diag_sum / total if total > 0 else 0
    # Adjacent accuracy: correct or off by one category
    adj_sum = 0
    for ii in range(4):
        for jj in range(4):
            if abs(ii - jj) <= 1:
                adj_sum += mat[ii, jj]
    adj_acc = adj_sum / total if total > 0 else 0
    fig.text(0.42, 0.03,
             f'Exact: {diag_sum}/{total} = {overall_acc:.0%}     '
             f'Within 1 category: {adj_sum}/{total} = {adj_acc:.0%}',
             ha='center', va='bottom', fontsize=7)

    fig.subplots_adjust(bottom=0.24, right=0.78)

    out = os.path.join(OUTDIR, 'fig_confusion_4level.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved: {out}')


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Loading data...')
    data = load_data()
    print(f'  {data["n"]} paired points, {len(data["sensors"])} sensors')

    print('Generating figures...')
    fig_scatter(data)
    fig_confusion_binary(data)
    fig_temperature(data)
    fig_confusion_4level(data)

    print('Done.')
