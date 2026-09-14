#!/usr/bin/env python3
"""
CBT revision, step 2: rebuild the sensor-CBT paired table around bucket soaks.

The submitted paper paired each CBT bag to the lowest mon2 reading within +/-10 min,
with a fixed ToF > 200 kcps air filter. That rule lets bags from neighbouring buckets
share one reading, and 200 kcps sits above every unit's air mode. This script replaces
it. Every rule below is decidable without the CBT result.

  1. Records: mWater rows whose Type of Enumeration is CBT, less the records excluded
     at the record level and with the corrections listed in judgments.json.
  2. Time: UTC = stored + (survey tz - sample tz). A blank survey tz takes the value
     recorded by the rest of its submission batch (same enumerator, submissions within
     60 min); a batch with no survey tz at all gets no shift.
  3. Reading: LED 512, bias in [2960, 3040], the single bias nearest 3000.
  4. In water: ToF below the unit's own split AND fluorescence below the unit's own
     split, each the Otsu split of that unit's readings (separate optical apertures).
  5. Immersion: a run of consecutive in-water 50065 readings (1-min cadence), broken by
     any air reading or a gap > 3 min. 50065 read nearly every record, so its runs
     define the bucket immersions.
  6. Record -> immersion: the immersion whose window, widened by TOL_MIN, contains the
     sample time. With no 50065 immersion, the other unit's own run of in-water
     readings, widened by half its reporting interval plus TOL_MIN.
  7. Per record and unit: the unit's in-water readings within REC_WIN_MIN of the sample
     time (median), else its nearest in-water reading in the immersion. No reading, no
     pair.
  8. Temperature: ln(mon2 - PED) corrected to TREF with the batch quench coefficient
     (judgments.json) on each unit's own temperature channel.
  9. Drift: minus the unit's clean-water (treated-water) baseline for that day
     (evaluate.py daily_baseline: own immersion excluded, interpolated across days).
  10. Replicates: consecutive records in one immersion up to 10 min apart form a sample.
  11. Case-by-case judgments (judgments.json) are applied, each with its recorded reason.

Inputs:  data/cbt_revision/raw/{sipm,tof,diagnostics}_{bc}.json.gz  (pumphaus proxy,
         PIN 4001, no combo filter), data/cbt-datagrid.csv (mWater export the paper used)
Outputs: data/cbt_revision/{soaks,bags,pairs,unresolved}.csv, waterfall.json
Usage:   python3 scripts/cbt_revision/pair_soaks.py
"""
import csv, gzip, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'cbt_revision' / 'raw'
OUT = ROOT / 'data' / 'cbt_revision'
DATAGRID = ROOT / 'data' / 'cbt-datagrid.csv'
JUDGMENTS = ROOT / 'data' / 'cbt_revision' / 'judgments.json'
PAPER_PAIRS = ROOT / 'data' / 'cbt_revision' / 'submitted' / 'paired_observations.csv'   # the submitted paper's pairs

UNITS = (50045, 50053, 50065)
SOAK_UNIT = 50065
TOL_MIN = 2
SOAK_GAP_MIN = 3
CBT_CEILING = 100
REC_WIN_MIN = 2          # readings within +/-2 min of the mWater sample time
PED = 170                # SiPM pedestal (SLOPE_CAL.PED fallback)
TREF = 20.0              # degC, reference temperature (validation.thelume.ai/findings)
# batch TLF quench coefficient, recorded in judgments.json (source: shared/tlf-quench.js TLF_QUENCH_BATCH_TLF)
BATCH_RHO = float(json.load(open(JUDGMENTS))['temperature']['batch_quench'])


def load(name, bc):
    with gzip.open(RAW / f'{name}_{bc}.json.gz', 'rt') as f:
        d = pd.DataFrame(json.load(f))
    d['t'] = pd.to_datetime(d.timestamp, utc=True, format='ISO8601')
    return d.sort_values('t')


def otsu_gate(values):
    x = np.log10(np.asarray(values, float))
    h, e = np.histogram(x, bins=120)
    c = (e[:-1] + e[1:]) / 2
    best, gate = -1.0, None
    for i in range(1, len(h)):
        w0, w1 = h[:i].sum(), h[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (h[:i] * c[:i]).sum() / w0
        m1 = (h[i:] * c[i:]).sum() / w1
        s = w0 * w1 * (m0 - m1) ** 2
        if s > best:
            best, gate = s, c[i]
    return float(10 ** gate)


def unit_readings(bc):
    s = load('sipm', bc)
    s = s[(s.led_power == 512) & s.sipm_bias.between(2960, 3040)].copy()
    s['d'] = (s.sipm_bias - 3000).abs()
    s = s.sort_values('d').groupby('timestamp', as_index=False).head(1).sort_values('t')
    tof = load('tof', bc)[['t', 'signal_per_spad_kcps']]
    diag = load('diagnostics', bc)
    for col in ('temperature', 'sipm_temperature'):
        if col not in diag:
            diag[col] = np.nan
    m = pd.merge_asof(s[['t', 'sipm_bias', 'mon2_val']], tof, on='t',
                      direction='nearest', tolerance=pd.Timedelta('2min'))
    m = pd.merge_asof(m, diag[['t', 'temperature', 'sipm_temperature']], on='t',
                      direction='nearest', tolerance=pd.Timedelta('5min'))
    m = m.rename(columns={'signal_per_spad_kcps': 'tof', 'mon2_val': 'mon2'})
    m = m.dropna(subset=['tof'])
    m = m[m.tof > 0]
    gate = otsu_gate(m.tof)
    # in water only when BOTH optical paths say water: ToF below its gate and fluorescence
    # below the unit's own air/water split (the two windows are separate apertures, so one
    # can be out of the water while the other is in; judgments.json "in_water_rule")
    mgate = otsu_gate(m.mon2[m.mon2 > 0])
    m['in_water'] = (m.tof < gate) & (m.mon2 < mgate)
    m.attrs['mon2_gate'] = mgate
    m['barcode'] = bc
    # the unit's own temperature: SiPM where it is reported, else the only channel it has
    has_sipm = m.sipm_temperature.notna()
    m['Tch'] = m.sipm_temperature.where(has_sipm, m.temperature)
    m['T_channel'] = np.where(has_sipm, 'sipm_temperature', 'temperature (board)')
    m['sig'] = np.log(np.clip(m.mon2 - PED, 1, None))
    return m.reset_index(drop=True), gate


def tz_minutes(s):
    mm = re.search(r'UTC([+-])(\d{2}):(\d{2})', s or '')
    if not mm:
        return None
    return (1 if mm.group(1) == '+' else -1) * (int(mm.group(2)) * 60 + int(mm.group(3)))


def load_bags():
    rows = list(csv.reader(open(DATAGRID, encoding='utf-8-sig')))
    hdr = rows[0]
    ix = {k: i for i, k in enumerate(hdr)}
    col = lambda prefix: next(i for k, i in ix.items() if k.startswith(prefix))
    c = dict(enum=col('Type of Enumeration'), sample=col('Date and time of water sample'),
             bc1=col('Lume Barcode'), bc2=col('Second Lume Barcode'), ecoli=col('Replicate 1: E Coli'),
             tzs=col('In what timezone was the sampl'), tzu=col('In what timezone was this surv'),
             who=col('Enumerator'), sub=col('Submitted On'), site=col('Water Sampling Site'),
             wtype=col('Type of Water'), past=col('Is the enumeration past detect'),
             cl=col('Chlorine residual'))
    out = []
    for r in rows[1:]:
        if 'cbt' not in r[c['enum']].lower():
            continue
        try:
            ecoli = float(r[c['ecoli']])
        except ValueError:
            continue
        out.append(dict(
            sample_local=r[c['sample']], site=r[c['site']].strip(), water_type=r[c['wtype']].strip(),
            cbt=ecoli, past_detect=r[c['past']].strip().lower().startswith('yes'),
            chlorine=r[c['cl']].strip(), enumerator=r[c['who']], submitted=pd.Timestamp(r[c['sub']]),
            tz_sample=r[c['tzs']].strip(), tz_survey=r[c['tzu']].strip(),
            barcodes=sorted({r[c['bc1']].strip(), r[c['bc2']].strip()} - {''}),
        ))
    b = pd.DataFrame(out).sort_values('submitted').reset_index(drop=True)
    # records excluded at the mWater record level (judgments.json) never enter the analysis
    J_ = json.load(open(JUDGMENTS))
    drop = np.zeros(len(b), bool)
    for o in J_.get('invalid_references', []):
        drop |= (b.sample_local.str[:19] == o['sample_local']).values & (b.site == o['site']).values
    for o in J_.get('excluded_records', []):
        drop |= (b.sample_local.str[:19] == o['sample_local']).values & b.site.str.contains(o['site_contains']).values
    b = b[~drop].reset_index(drop=True)
    # documented reference corrections (judgments.json), original value kept
    b['cbt_recorded'] = b.cbt
    for c_ in json.load(open(JUDGMENTS)).get('reference_corrections', []):
        m = (b.sample_local.str[:19] == c_['sample_local']) & (b.site == c_['site'])
        assert m.sum() == 1, f"reference correction matched {m.sum()} records: {c_}"
        b.loc[m, 'cbt'] = float(c_['corrected'])
    b['batch'] = ((b.enumerator != b.enumerator.shift()) |
                  (b.submitted.diff() > pd.Timedelta('60min'))).cumsum()
    # blank survey tz: take the batch's recorded value when the batch has exactly one
    fill = b[b.tz_survey != ''].groupby('batch').tz_survey.agg(lambda s: sorted(set(s)))
    b['tz_survey_used'] = b.tz_survey
    b['tz_imputed'] = False
    for i, r in b.iterrows():
        if r.tz_survey == '' and r.batch in fill.index and len(fill[r.batch]) == 1:
            b.at[i, 'tz_survey_used'] = fill[r.batch][0]
            b.at[i, 'tz_imputed'] = True
    shift = [(tz_minutes(u) - tz_minutes(s)) if (u and tz_minutes(u) is not None and tz_minutes(s) is not None) else 0
             for s, u in zip(b.tz_sample, b.tz_survey_used)]
    b['t_utc'] = pd.to_datetime(b.sample_local.str.replace(' ', 'T') + 'Z', utc=True) + pd.to_timedelta(shift, unit='min')
    b['shift_min'] = shift
    b['censored'] = (b.cbt >= CBT_CEILING) | b.past_detect
    # stable IDs from the sample time (a suffix separates any records sharing a minute)
    base_id = 'B' + b.sample_local.str.replace(r'\D', '', regex=True).str[:12]
    b['bag_id'] = base_id + b.groupby(base_id).cumcount().map(lambda k: '' if k == 0 else chr(97 + k))
    return b


def build_soaks(u65):
    m = u65.copy()
    brk = (~m.in_water) | (m.t.diff() > pd.Timedelta(minutes=SOAK_GAP_MIN))
    m['sid'] = brk.cumsum()
    s = m[m.in_water].groupby('sid').agg(start=('t', 'min'), end=('t', 'max'), n=('t', 'size')).reset_index(drop=True)
    s['soak_id'] = ['S%03d' % i for i in range(1, len(s) + 1)]
    return s


def apply_judgments(U, J):
    for o in J.get('force_in_water', []):
        u = U[o['barcode']]
        u.loc[(u.t >= pd.Timestamp(o['start'])) & (u.t <= pd.Timestamp(o['end'])), 'in_water'] = True
    for o in J.get('force_air', []):
        u = U[o['barcode']]
        t0 = pd.Timestamp(o['timestamp'])
        u.loc[(u.t - t0).abs() <= pd.Timedelta('90s'), 'in_water'] = False


def partner_runs(u):
    """Runs of consecutive in-water readings, each widened by half the local reporting
    interval plus TOL_MIN."""
    m = u.reset_index(drop=True)
    gaps = m.t.diff()
    brk = (~m.in_water) | (gaps > pd.Timedelta('20min'))
    m['rid'] = brk.cumsum()
    runs = []
    for _, g in m[m.in_water].groupby('rid'):
        near = m[(m.t >= g.t.iloc[0] - pd.Timedelta('1h')) & (m.t <= g.t.iloc[-1] + pd.Timedelta('1h'))]
        cadence = near.t.diff().median()
        pad = cadence / 2 + pd.Timedelta(minutes=TOL_MIN)
        runs.append(dict(start=g.t.iloc[0], end=g.t.iloc[-1], widen=pad))
    return pd.DataFrame(runs)


def fit_own_quench(u, soaks):
    """d ln(mon2 - PED)/dT on the unit's own temperature, from within-immersion variation
    (immersion fixed effects). Leaves out the first and last in-water reading of each
    immersion (entry and lift transients, identified from the ToF sequence) and any reading
    whose temperature repeats the previous one (a stale sensor value)."""
    xs, ys = [], []
    n_imm = 0
    for _, s in soaks.iterrows():
        w = u[(u.t >= s.start) & (u.t <= s.end) & u.in_water].sort_values('t')
        w = w.iloc[1:-1]
        w = w[w.Tch.diff().fillna(1) != 0]
        w = w.dropna(subset=['Tch'])
        if len(w) < 3 or w.Tch.max() - w.Tch.min() < 0.5:
            continue
        n_imm += 1
        xs.append((w.Tch - w.Tch.mean()).values)
        ys.append((w.sig - w.sig.mean()).values)
    x, y = np.concatenate(xs), np.concatenate(ys)
    b = float((x * y).sum() / (x * x).sum())
    resid = y - b * x
    se = float(np.sqrt((resid ** 2).sum() / (len(x) - n_imm - 1) / (x * x).sum()))
    r2 = float(1 - (resid ** 2).sum() / (y ** 2).sum())
    return dict(b=b, se=se, n=int(len(x)), immersions=n_imm, r2_within=r2)


def summarize(win, t_rec):
    """The unit's in-water readings within REC_WIN_MIN of the sample time; if none, the
    nearest in-water reading inside the bucket window."""
    w = win[win.in_water]
    if w.empty:
        return None
    dt = (w.t - t_rec).abs()
    near = w[dt <= pd.Timedelta(minutes=REC_WIN_MIN)]
    if near.empty:
        near = w.loc[[dt.idxmin()]]
    return dict(n_readings=len(near), mon2=float(near.mon2.median()), tof=float(near.tof.median()),
                Tch=float(near.Tch.median()) if near.Tch.notna().any() else np.nan,
                temp_diag=float(near.temperature.median()) if near.temperature.notna().any() else np.nan,
                T_channel=near.T_channel.iloc[0],
                sig=float(near.sig.median()), sig_T=float(near.sig_T.median()),
                reading_offset_min=float(((near.t - t_rec).dt.total_seconds() / 60).median()))


def main():
    U, gates = {}, {}
    for bc in UNITS:
        U[bc], gates[bc] = unit_readings(bc)
    J = json.load(open(JUDGMENTS))
    apply_judgments(U, J)
    soaks = build_soaks(U[SOAK_UNIT])
    # temperature correction (findings): each unit's own coefficient on its own temperature,
    # fitted over the analysed window where that is identifiable; otherwise the batch value
    # 50065's own within-immersion fit is not identifiable (SE about equal to the estimate),
    # so every unit takes the batch coefficient (Evan Thomas, 2026-09-14); the own fit is kept
    # in the waterfall as a sensitivity.
    own_fit_50065 = fit_own_quench(U[SOAK_UNIT], soaks)
    quench = {bc: dict(b=BATCH_RHO, source='batch TLF_QUENCH_BATCH_TLF (tlf-quench.js) on '
                       + U[bc].T_channel.iloc[0]) for bc in UNITS}
    for bc in UNITS:
        U[bc]['sig_T'] = U[bc].sig - quench[bc]['b'] * (U[bc].Tch - TREF)
    runs = {bc: partner_runs(U[bc]) for bc in UNITS if bc != SOAK_UNIT}
    excluded = {(int(o['barcode']), o['sample_local']): o['reason'] for o in J.get('exclude_pairs', [])}
    # whole-record exclusions, every unit: invalid references and operator-reported failures
    excl_records = [(o['sample_local'], o['site'], 'invalid CBT: ' + o['rule']) for o in J.get('invalid_references', [])]
    excl_records += [(o['sample_local'], o['site_contains'], J.get('excluded_records_reason', ''))
                     for o in J.get('excluded_records', [])]

    def record_excluded(b):
        for loc, site, why in excl_records:
            if b.sample_local[:19] == loc and site in str(b.site):
                return why
        return None
    bags = load_bags()
    tol = pd.Timedelta(minutes=TOL_MIN)

    def soak_for(t):
        k = soaks[(soaks.start - tol <= t) & (soaks.end + tol >= t)]
        if k.empty:
            return None
        mid = k.start + (k.end - k.start) / 2
        return k.iloc[int(np.argmin(np.abs((mid - t).dt.total_seconds())))]

    pairs, unresolved = [], []
    bag_soak = []
    for _, b in bags.iterrows():
        sk = soak_for(b.t_utc)
        bag_soak.append(sk.soak_id if sk is not None else '')
        for bc in b.barcodes:
            bc = int(bc)
            if bc not in U:
                continue
            rec = dict(bag_id=b.bag_id, barcode=bc, soak_id=sk.soak_id if sk is not None else '')
            why = record_excluded(b)
            if why:
                rec['status'] = 'excluded record: ' + why[:80]
                unresolved.append(rec); continue
            if (bc, b.sample_local) in excluded:
                rec['status'] = 'excluded by judgment: ' + excluded[(bc, b.sample_local)]
                unresolved.append(rec); continue
            if sk is None:
                r = runs.get(bc)
                hit = None if r is None or r.empty else r[(r.start - r.widen <= b.t_utc) & (r.end + r.widen >= b.t_utc)]
                if hit is None or hit.empty:
                    rec['status'] = 'no soak at bag time'
                    unresolved.append(rec); continue
                h = hit.iloc[0]
                rec['soak_id'] = f'P{bc}-{h.start.strftime("%m%d%H%M")}'
                win = U[bc][(U[bc].t >= h.start) & (U[bc].t <= h.end)]
                smry = summarize(win, b.t_utc)
                rec.update(smry, status='paired (partner soak)')
                pairs.append(rec); continue
            win = U[bc][(U[bc].t >= sk.start - tol) & (U[bc].t <= sk.end + tol)]
            smry = summarize(win, b.t_utc)
            if smry is None:
                rec['status'] = 'no in-water reading in soak' if len(win) else 'no reading in soak'
                unresolved.append(rec); continue
            rec.update(smry, status='paired')
            pairs.append(rec)
    bags['soak_id'] = bag_soak

    P = pd.DataFrame(pairs).merge(bags[['bag_id', 'site', 'water_type', 'cbt', 'censored', 't_utc']], on='bag_id')
    X = pd.DataFrame(unresolved).merge(bags[['bag_id', 'site', 'water_type', 'cbt', 't_utc', 'sample_local']], on='bag_id')
    P['b_T'] = P.barcode.map(lambda k: quench[k]['b'])

    # drift correction (findings: re-baseline, since drift cannot be projected): subtract the
    # unit's treated-water reference for the day, chosen by water type, not by CBT result.
    # Source-only days take the nearest day that has treated buckets, and are flagged.
    P['day'] = pd.to_datetime(P.t_utc, utc=True).dt.date
    treated = P[P.water_type.str.startswith('Treated')]
    per_bucket = treated.groupby(['barcode', 'day', 'soak_id']).sig_T.median().reset_index()
    ref = per_bucket.groupby(['barcode', 'day']).sig_T.agg(['median', 'size']).reset_index()
    base, base_day, age = [], [], []
    for _, r in P.iterrows():
        cand = ref[ref.barcode == r.barcode].copy()
        cand['gap'] = [abs((d - r.day).days) for d in cand.day]
        best = cand[cand.gap == cand.gap.min()]
        base.append(float(best['median'].mean()))
        base_day.append(','.join(str(d) for d in best.day))
        age.append(int(cand.gap.min()))
    P['baseline'] = base
    P['baseline_day'] = base_day
    P['baseline_age_days'] = age
    P['sig_TD'] = P.sig_T - P.baseline

    # sample groups: consecutive records in one bucket up to 10 min apart are replicates of
    # the same water (judgments.json "replicates"); every other paired bag is its own group
    grp, gid = {}, 0
    paired_bags = bags[bags.bag_id.isin(P.bag_id)].copy()
    paired_bags['soak_used'] = paired_bags.bag_id.map(P.groupby('bag_id').soak_id.first())
    for sid, g in paired_bags.sort_values('t_utc').groupby('soak_used'):
        prev, gname = None, None
        for _, r in g.iterrows():
            if prev is None or (r.t_utc - prev).total_seconds() / 60 > 10:
                gid += 1
                gname = 'G' + r.bag_id[1:]      # named after the group's first record
            grp[r.bag_id] = gname
            prev = r.t_utc
    bags['sample_group'] = bags.bag_id.map(grp)
    P['sample_group'] = P.bag_id.map(grp)

    # TLF change between consecutive records in a bucket (diagnostic)
    reps = []
    for (bc, sid), g in P.groupby(['barcode', 'soak_id']):
        g = g.sort_values('t_utc')
        tt = pd.to_datetime(g.t_utc, utc=True).tolist()
        for i in range(1, len(g)):
            gap = (tt[i] - tt[i - 1]).total_seconds() / 60
            if gap <= 10:
                reps.append(dict(barcode=bc, soak_id=sid, bag_a=g.bag_id.iloc[i - 1], bag_b=g.bag_id.iloc[i],
                                 gap_min=gap, d_sig_TD=float(g.sig_TD.iloc[i] - g.sig_TD.iloc[i - 1]),
                                 cbt_a=g.cbt.iloc[i - 1], cbt_b=g.cbt.iloc[i]))
    R = pd.DataFrame(reps)
    R.to_csv(OUT / 'replicate_candidates.csv', index=False)

    # soaks holding bags from more than one site name
    site_per_soak = bags[bags.soak_id != ''].groupby('soak_id').site.nunique()
    multi_site = site_per_soak[site_per_soak > 1].index.tolist()

    # compare with the paper's pairs
    old = pd.read_csv(PAPER_PAIRS)
    old['key'] = old.barcode.astype(str) + '|' + old.sample_date.str[:16]
    bags['key_local'] = bags.sample_local.str[:16]
    P['key'] = P.barcode.astype(str) + '|' + P.bag_id.map(bags.set_index('bag_id').key_local)
    both = old.merge(P[['key', 'mon2', 'tof']], on='key', how='left', suffixes=('_old', ''))
    moved = both[both.mon2.notna() & (both.sensor_mon2_raw != both.mon2)]
    lost = both[both.mon2.isna()]
    gained = P[~P.key.isin(old.key)]

    wf = {
        'mwater_cbt_rows': int(len(bags)),
        'bag_unit_candidates': int(sum(len([x for x in bb if int(x) in U]) for bb in bags.barcodes)),
        'bags_with_soak': int((bags.soak_id != '').sum()),
        'bags_without_soak': int((bags.soak_id == '').sum()),
        'pairs': int(len(P)),
        'pairs_by_unit': {str(k): int(v) for k, v in P.groupby('barcode').size().items()},
        'pairs_from_partner_soaks': int((P.status == 'paired (partner soak)').sum()),
        'temperature_correction': {str(k): {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in quench.items()},
        'sample_groups': int(bags.sample_group.nunique()),
        'sample_groups_with_replicates': int((bags.dropna(subset=['sample_group']).groupby('sample_group').size() > 1).sum()),
        'own_fit_50065_sensitivity': {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in own_fit_50065.items()},
        'drift_reference_age_days': {str(k): v for k, v in P.groupby('baseline_age_days').size().items()},
        'pairs_with_older_reference': int((P.baseline_age_days > 0).sum()),
        'bags_paired': int(P.bag_id.nunique()),
        'soaks_used': int(P.soak_id.nunique()),
        'unresolved_by_status': {f'{k[0]} {k[1]}': int(v) for k, v in X.groupby(['barcode', 'status']).size().items()} if len(X) else {},
        'tz_shifted_bags': int((bags.shift_min != 0).sum()),
        'tz_imputed_bags': int(bags.tz_imputed.sum()),
        'gates_kcps': {str(k): round(v, 1) for k, v in gates.items()},
        'soaks_total_50065': int(len(soaks)),
        'soaks_with_multiple_site_names': multi_site,
        'vs_paper': {'paper_pairs': int(len(old)), 'kept_same_reading': int(len(both) - len(moved) - len(lost)),
                     'kept_new_reading': int(len(moved)), 'dropped': int(len(lost)), 'added': int(len(gained))},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    soaks.to_csv(OUT / 'soaks.csv', index=False)
    bags.drop(columns=['key_local']).to_csv(OUT / 'bags.csv', index=False)
    P.drop(columns=['key']).to_csv(OUT / 'pairs.csv', index=False)
    X.to_csv(OUT / 'unresolved.csv', index=False)
    json.dump(wf, open(OUT / 'waterfall.json', 'w'), indent=2)
    print(json.dumps(wf, indent=2))
    print('\npaper pairs dropped:'); print(lost[['barcode', 'sample_date', 'cbt_ecoli_cfu', 'sensor_mon2_raw', 'sensor_tof_raw']].to_string(index=False))
    print('\npairs added (not in paper):'); print(gained[['barcode', 'bag_id', 't_utc', 'cbt', 'mon2', 'tof']].to_string(index=False))
    print('\nunresolved:'); print(X[['barcode', 'status', 'sample_local', 't_utc', 'cbt', 'water_type']].to_string(index=False))
    mv = moved.assign(dmon2=moved.mon2 - moved.sensor_mon2_raw)
    print('\nkept with a new reading: median |mon2 change| by unit'); print(mv.groupby('barcode').dmon2.agg(lambda s: s.abs().median()).to_string())


if __name__ == '__main__':
    main()
