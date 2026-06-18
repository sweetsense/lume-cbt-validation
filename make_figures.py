#!/usr/bin/env python3
"""
Generate publication-quality figures for the CBT fluorimeter validation paper.

Reads paper_data.json (exported by export-paper-data.js) and produces:
  - fig_scatter.pdf      (Figure 1: regression scatter)
  - fig_confusion_binary.pdf  (Figure 2: binary confusion matrices)
  - fig_temperature.pdf  (Figure 3: temperature correction)
  - fig_confusion_3level.pdf  (Figure 4: 3-level risk confusion matrix)

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

    # Add small jitter to reduce overplotting (deterministic seed)
    rng = np.random.RandomState(42)
    jitter_x = rng.uniform(-0.03, 0.03, len(obs))
    jitter_y = rng.uniform(-0.03, 0.03, len(pred))
    obs_j = obs + jitter_x
    pred_j = pred + jitter_y

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH + 0.5))

    # Axis range
    lo, hi = -0.15, 2.6
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # Agreement band (subtle shading around 1:1 line)
    xx = np.linspace(lo, hi, 200)
    ax.fill_between(xx, xx - band, xx + band, color='#E8F5E9', alpha=0.5,
                     zorder=1, label='_nolegend_')
    # Band boundary lines
    ax.plot(xx, xx - band, '--', color='#A5D6A7', linewidth=0.5, zorder=1)
    ax.plot(xx, xx + band, '--', color='#A5D6A7', linewidth=0.5, zorder=1)

    # 1:1 reference line
    ax.plot([lo, hi], [lo, hi], '-', color='#555555', linewidth=0.8, zorder=2)

    # Plot points by sensor, colored by agreement
    for bc in data['sensors']:
        mask_bc = np.array([b == bc for b in barcodes])
        color = SENSOR_COLORS.get(bc, '#999999')

        # Uncensored, agree
        m = mask_bc & agree & ~censored
        if m.any():
            ax.scatter(obs_j[m], pred_j[m], s=28, c=color, marker='o',
                       edgecolors='white', linewidths=0.4, alpha=0.85, zorder=3)

        # Uncensored, disagree
        m = mask_bc & ~agree & ~censored
        if m.any():
            ax.scatter(obs_j[m], pred_j[m], s=36, c=DISAGREE_COLOR, marker='x',
                       linewidths=0.9, alpha=0.9, zorder=4)

        # Censored, agree (rightward arrow)
        m = mask_bc & agree & censored
        if m.any():
            ax.scatter(obs_j[m], pred_j[m], s=32, c=color, marker='>',
                       edgecolors='white', linewidths=0.4, alpha=0.85, zorder=3)

        # Censored, disagree
        m = mask_bc & ~agree & censored
        if m.any():
            ax.scatter(obs_j[m], pred_j[m], s=32, c=DISAGREE_COLOR, marker='>',
                       edgecolors='white', linewidths=0.4, alpha=0.9, zorder=4)

    # Axis labels
    ax.set_xlabel('Observed CBT (log$_{10}$(CFU + 1))')
    ax.set_ylabel('Predicted fluorimeter (log$_{10}$(CFU + 1))')

    # Inset text: metrics (top-left, clear of data)
    r2 = model['r2']
    mae_val = model['mae']
    n = data['n']
    textstr = f'$R^2$ = {r2:.3f}\nMAE = {mae_val:.3f} log$_{{10}}$\n$n$ = {n}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    # Legend below the plot
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SENSOR_COLORS['50045'],
               markeredgecolor='white', markeredgewidth=0.4,
               markersize=5.5, label='50045'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SENSOR_COLORS['50053'],
               markeredgecolor='white', markeredgewidth=0.4,
               markersize=5.5, label='50053'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SENSOR_COLORS['50065'],
               markeredgecolor='white', markeredgewidth=0.4,
               markersize=5.5, label='50065'),
        Line2D([0], [0], marker='x', color=DISAGREE_COLOR, linestyle='None',
               markersize=5.5, label='Disagree'),
        Line2D([0], [0], marker='>', color='w', markerfacecolor='#666666',
               markersize=5.5, label='Censored'),
    ]
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=False,
              fontsize=6.5, handletextpad=0.3, columnspacing=1.0)

    fig.subplots_adjust(bottom=0.18)

    out = os.path.join(OUTDIR, 'fig_scatter.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved: {out}')


# ════════════════════════════════════════════════════════════════════
# Figure 2: Binary confusion matrices (side-by-side)
# ════════════════════════════════════════════════════════════════════
def fig_confusion_binary(data):
    CBT_CEILING = 0.925  # KWR/JMP CBT-Colilert threshold agreement

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, COL_WIDTH * 1.55))

    for ax_idx, key in enumerate(['ge1', 'ge10']):
        ax = axes[ax_idx]
        res = data['logistic_classification'][key]
        cm = res['confusion_matrix']
        thr = res['threshold']
        # Transpose: rows = predicted, columns = true class
        mat = np.array([[cm['tn'], cm['fn']], [cm['fp'], cm['tp']]])
        total = mat.sum()

        from matplotlib.colors import to_rgba

        colors = np.array([
            [AGREE_COLOR, DISAGREE_COLOR],
            [DISAGREE_COLOR, AGREE_COLOR],
        ])

        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                pct = 100 * val / total
                bg = colors[i, j]
                rgba = list(to_rgba(bg))
                max_val = mat.max()
                intensity = 0.15 + 0.45 * (val / max(max_val, 1))
                rgba[3] = intensity
                ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                             facecolor=rgba, edgecolor='white', linewidth=2))
                cy = 1.5 - i
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
        ax.set_xlabel('True class', fontsize=9, labelpad=6)
        if ax_idx == 0:
            ax.set_ylabel('Predicted class', fontsize=9, labelpad=6)

        ax.set_title(f'\u2265{thr} CFU/100 mL', fontsize=9, pad=10)
        ax.set_aspect('equal')
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)

        # Full metrics below each matrix (replaces Table 3)
        bal = res['balanced_accuracy']
        sens = res['sensitivity']
        spec = res['specificity']
        auc = res['auc']
        ceiling_pct = bal / CBT_CEILING
        n_pos = cm['tp'] + cm['fn']
        n_neg = cm['tn'] + cm['fp']
        metrics = (f'Bal. acc. = {bal:.0%} ({ceiling_pct:.0%} of CBT ceiling)\n'
                   f'Sens. = {sens:.0%}    Spec. = {spec:.0%}    '
                   f'AUC = {auc:.3f}\n'
                   f'$n_+$ = {n_pos}    $n_-$ = {n_neg}')
        ax.text(0.5, -0.20, metrics, ha='center', va='top', fontsize=7,
                transform=ax.transAxes)

    fig.subplots_adjust(wspace=0.4, bottom=0.28, top=0.90)

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
# Figure 4: 3-level risk confusion matrix
# ════════════════════════════════════════════════════════════════════
def fig_confusion_3level(data):
    fl = data['three_level']
    # Transpose: rows = predicted, columns = true class
    mat = np.array(fl['matrix']).T
    labels = fl['labels']
    risk_labels = fl['risk_labels']
    K = 3
    total = mat.sum()

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 1.15))

    max_val = mat.max()
    from matplotlib.colors import to_rgba
    for i in range(K):
        for j in range(K):
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

            ax.add_patch(plt.Rectangle((j, K - 1 - i), 1, 1,
                         facecolor=rgba, edgecolor='white', linewidth=1.5))

            cy = K - 0.5 - i
            fontcolor = '#333333' if val > 0 else '#bbbbbb'
            ax.text(j + 0.5, cy + 0.08, str(val),
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color=fontcolor)
            if val > 0:
                ax.text(j + 0.5, cy - 0.20, f'({pct:.1f}%)',
                        ha='center', va='center', fontsize=7, color='#666666')

    ax.set_xlim(0, K)
    ax.set_ylim(0, K)
    ax.set_xticks([i + 0.5 for i in range(K)])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(K)])
    ax.set_yticklabels(list(reversed(labels)), fontsize=8)

    ax.set_xlabel('True class', fontsize=9, labelpad=8)
    ax.set_ylabel('Predicted class', fontsize=9, labelpad=8)

    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    # Risk labels along the top
    for j, rl in enumerate(risk_labels):
        ax.text(j + 0.5, K + 0.12, rl, ha='center', va='bottom', fontsize=6,
                color='#888888', style='italic')

    # Accuracy metrics
    diag_sum = sum(mat[i, i] for i in range(K))
    overall_acc = diag_sum / total if total > 0 else 0
    adj_sum = sum(mat[ii, jj] for ii in range(K) for jj in range(K) if abs(ii - jj) <= 1)
    adj_acc = adj_sum / total if total > 0 else 0
    fig.text(0.42, 0.03,
             f'Exact: {diag_sum}/{total} = {overall_acc:.0%}     '
             f'Within 1 category: {adj_sum}/{total} = {adj_acc:.0%}',
             ha='center', va='bottom', fontsize=7.5)

    fig.subplots_adjust(bottom=0.18, top=0.88)

    out = os.path.join(OUTDIR, 'fig_confusion_3level.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved: {out}')


# ════════════════════════════════════════════════════════════════════
# Figure 5: Per-sensor scatter facets (3-panel)
# ════════════════════════════════════════════════════════════════════
def fig_per_sensor(data):
    pts = data['paired_points']
    sensors = data['sensors']
    model = data['model']
    band = model['agreement_band']
    per_sensor = data.get('per_sensor_loocv', {})

    fig, axes = plt.subplots(1, len(sensors), figsize=(DOUBLE_COL, COL_WIDTH * 0.95),
                              sharey=True)
    if len(sensors) == 1:
        axes = [axes]

    lo, hi = -0.15, 2.6

    for ax_idx, bc in enumerate(sensors):
        ax = axes[ax_idx]
        bc_pts = [p for p in pts if p['barcode'] == bc]
        obs = np.array([p['observed_log'] for p in bc_pts])
        pred = np.array([p['predicted_log'] for p in bc_pts])
        loo = np.array([p['predicted_loo_log'] for p in bc_pts])
        censored = np.array([p['censored'] for p in bc_pts])
        agree = np.array([p['agree'] for p in bc_pts])

        rng = np.random.RandomState(42 + ax_idx)
        jx = rng.uniform(-0.03, 0.03, len(obs))
        jy = rng.uniform(-0.03, 0.03, len(pred))

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        # Agreement band
        xx = np.linspace(lo, hi, 200)
        ax.fill_between(xx, xx - band, xx + band, color='#E8F5E9', alpha=0.5, zorder=1)
        ax.plot(xx, xx - band, '--', color='#A5D6A7', linewidth=0.4, zorder=1)
        ax.plot(xx, xx + band, '--', color='#A5D6A7', linewidth=0.4, zorder=1)
        ax.plot([lo, hi], [lo, hi], '-', color='#555555', linewidth=0.6, zorder=2)

        color = SENSOR_COLORS.get(bc, '#999999')

        # Agree, uncensored
        m = agree & ~censored
        if m.any():
            ax.scatter(obs[m] + jx[m], pred[m] + jy[m], s=20, c=color, marker='o',
                       edgecolors='white', linewidths=0.3, alpha=0.85, zorder=3)
        # Disagree, uncensored
        m = ~agree & ~censored
        if m.any():
            ax.scatter(obs[m] + jx[m], pred[m] + jy[m], s=28, c=DISAGREE_COLOR, marker='x',
                       linewidths=0.7, alpha=0.9, zorder=4)
        # Censored agree
        m = agree & censored
        if m.any():
            ax.scatter(obs[m] + jx[m], pred[m] + jy[m], s=24, c=color, marker='>',
                       edgecolors='white', linewidths=0.3, alpha=0.85, zorder=3)
        # Censored disagree
        m = ~agree & censored
        if m.any():
            ax.scatter(obs[m] + jx[m], pred[m] + jy[m], s=24, c=DISAGREE_COLOR, marker='>',
                       edgecolors='white', linewidths=0.3, alpha=0.9, zorder=4)

        # Per-sensor stats
        ps = per_sensor.get(bc, {})
        ps_n = ps.get('n', len(bc_pts))
        ps_agree = ps.get('agree', 0)
        ps_pct = ps.get('pct', 0)
        ps_prog = ps.get('program', '')
        ax.set_title(f'{bc} ({ps_prog})', fontsize=9, pad=6)
        ax.text(0.05, 0.95,
                f'$n$ = {ps_n}\nAgree = {ps_agree}/{ps_n} ({ps_pct:.0f}%)',
                transform=ax.transAxes, fontsize=6.5, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9))

        ax.set_xlabel('Observed CBT (log$_{10}$(CFU+1))', fontsize=8)
        if ax_idx == 0:
            ax.set_ylabel('Predicted (log$_{10}$(CFU+1))', fontsize=8)

        ax.set_aspect('equal')

    fig.subplots_adjust(wspace=0.08, bottom=0.15)

    out = os.path.join(OUTDIR, 'fig_per_sensor.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved: {out}')


# ════════════════════════════════════════════════════════════════════
# Figure 6: Chlorination matched pre/post dot plot
# ════════════════════════════════════════════════════════════════════
def fig_chlorination(data):
    chlor = data.get('chlorination', {})
    systems = chlor.get('matched_systems', [])
    if not systems:
        print('  Skipping fig_chlorination: no matched systems')
        return

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.85))

    names = [f'System {i+1}' for i in range(len(systems))]
    x = np.arange(len(names))
    width = 0.35

    source_vals = [s['source_gm_lume'] for s in systems]
    treated_vals = [s['treated_gm_lume'] for s in systems]
    # Use log scale, but min at 0.1 for display
    source_plot = [max(v, 0.1) for v in source_vals]
    treated_plot = [max(v, 0.1) for v in treated_vals]

    bars1 = ax.bar(x - width/2, source_plot, width, label='Source (pre-chlorination)',
                   color='#0ea5e9', alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, treated_plot, width, label='Treated (post-chlorination)',
                   color='#228833', alpha=0.85, edgecolor='white', linewidth=0.5)

    # WHO 10 CFU line
    ax.axhline(y=10, color='#CC3311', linewidth=1.0, linestyle='--', alpha=0.7, zorder=1)
    ax.text(len(names) - 0.6, 10 * 1.3, '10 CFU (WHO)', fontsize=6.5, color='#CC3311',
            ha='right', va='bottom')

    ax.set_yscale('log')
    ax.set_ylabel('Lume predicted CFU/100 mL')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=30, ha='right')

    # Add delta annotations
    for i, s in enumerate(systems):
        delta = s.get('delta_log', 0)
        y_top = max(source_plot[i], treated_plot[i]) * 1.8
        ax.text(i, y_top, f'\u0394={delta:+.2f}',
                ha='center', va='bottom', fontsize=6, color='#333333')

    ax.legend(loc='upper right', frameon=True, framealpha=0.9,
              edgecolor='#cccccc', fontsize=6.5)

    # Summary text
    n_sys = chlor.get('n_matched_systems', 0)
    cbt_pct = chlor.get('chlorinated_cbt_zero_pct', 0)
    lume_pct = chlor.get('chlorinated_lume_safe_pct', 0)
    ax.text(0.02, 0.02,
            f'{n_sys} matched systems\n'
            f'Chlorinated: {cbt_pct:.0f}% CBT=0, {lume_pct:.0f}% Lume<10',
            transform=ax.transAxes, fontsize=6, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    out = os.path.join(OUTDIR, 'fig_chlorination.pdf')
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
    fig_confusion_3level(data)
    fig_per_sensor(data)
    fig_chlorination(data)

    print('Done.')
