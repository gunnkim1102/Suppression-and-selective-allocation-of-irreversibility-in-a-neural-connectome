"""
Collect the key numbers from results/*.json, print them next to the reference
values of the published run, and write results/summary.csv.

    python code/check_results.py

Reference values are rounded to four decimals; a row is marked OK when this
run agrees with the reference to that precision. Parts that have not been
computed yet are skipped.
"""
import csv
import json

import numpy as np

from compute import RESULTS_DIR

TOL = 1e-4          # reference values carry four decimals

# Reference values from the published run (seeds and settings in compute.py).
REF = {
    'Fig 1b  sigma (connectome)': 0.1657,
    'Fig 2b  suppression % degree': 20.7914,
    'Fig 2b  suppression % hierarchy': 21.8471,
    'Fig 2b  suppression % strength': 13.3471,
    'Fig 2b  suppression % strength_hierarchy': 12.1022,
    'Fig 3   F0 connectome': 0.4966,
    'Fig 3   F0 degree-null mean': 0.6699,
    'Fig 3   r(F0, sigma) over nulls': -0.0719,
    'SI S8   rotation rate 1': 0.1131,
    'SI S8   rotation rate 2': 0.1066,
    'SI S8   rotation rate 3': 0.0762,
    'SI S8   rotation rate 4': 0.0730,
    'SI S8   rotation rate 5': 0.0645,
    'SI S8   rotation rate 6': 0.0575,
    'SI S13  slope log c_i ~ log k_i': 1.2934,
    'SI S13  R2': 0.5833,
    'Fig S1a slope log sigma_i ~ log k_i': 1.4277,
    'Fig S1a R2': 0.7085,
    'Table SVI Forward real z': -0.6813,
    'Table SVI Backward real z': 2.1201,
    'Table SVI Turn/steer real z': 4.7506,
    'Table SVI Nocicept/escape real z': 3.3106,
    'Table SVI Chemosens.integ real z': 5.4519,
    'Table SVI Forward null z': -0.6257,
    'Table SVI Backward null z': 0.9685,
    'Table SVI Turn/steer null z': 5.6960,
    'Table SVI Nocicept/escape null z': 2.0012,
    'Table SVI Chemosens.integ null z': 6.2025,
    'Table SIX Forward z': -3.1599,
    'Table SIX Backward z': 3.9019,
    'Table SIX Turn/steer z': 1.2082,
    'Table SIX Nocicept/escape z': 2.6859,
    'Table SIX Chemosens.integ z': 1.5953,
    'Table SVIII rho(c_i, Weighted degree)': 0.7085,
    'Table SVIII rho(c_i, Eigenvector centrality)': 0.5596,
    'Table SVIII rho(c_i, k-core number)': 0.5208,
    'Table SVIII rho(c_i, Closeness)': 0.5175,
    'Table SVIII rho(c_i, Betweenness)': 0.4167,
    'Table SVIII rho(c_i, Clustering coefficient)': -0.1325,
    'Fig 4c  stage 1 suppression %': 13.0370,
    'Fig 4c  stage 2 suppression %': 41.9841,
    'Fig 4c  stage 3 suppression %': 56.2384,
    'Fig 4c  stage 4 suppression %': 69.6436,
    'Fig 4c  stage 5 suppression %': 57.3228,
    'Fig 4c  stage 6 suppression %': 68.0399,
    'Fig 4c  stage 7 suppression %': 50.4340,
    'Fig 4c  stage 8 suppression %': 72.7297,
    'SI S2   strength null 1x swaps, suppression %': 13.2886,
    'SI S2   strength null 5x swaps, suppression %': 13.7331,
}
# Fig 4(d): dissipation rank of RIA / AVA / AIY per stage (exact integers)
REF_RANKS = {1: (3, 31, 10), 2: (1, 33, 13), 3: (1, 38, 21), 4: (1, 26, 16),
             5: (1, 30, 20), 6: (1, 25, 31), 7: (1, 13, 34), 8: (1, 15, 38)}
# Fig S1(b): mode x circuit alignment z, rounded to two decimals
REF_S1B = [[-1.21, 3.97, 4.54, 3.50, 2.90],
           [-1.22, 1.92, 7.25, 2.32, 3.15],
           [1.67, 1.74, 0.45, 5.20, 2.99],
           [-1.18, -0.52, 3.79, 0.77, 9.94],
           [-1.73, -0.96, 5.21, -0.07, 10.69]]


def load(part):
    p = RESULTS_DIR / f'{part}.json'
    return json.loads(p.read_text()) if p.exists() else None


def collect():
    v = {}
    f1 = load('fig1b')
    if f1:
        v['Fig 1b  sigma (connectome)'] = f1['sigma'][-1]
    f2 = load('fig2')
    if f2:
        for k, d in f2['null'].items():
            v[f'Fig 2b  suppression % {k}'] = d['suppression_pct']
    f3 = load('fig3')
    if f3:
        v['Fig 3   F0 connectome'] = f3['F0_real']
        v['Fig 3   F0 degree-null mean'] = float(np.mean(f3['F0_null']))
        v['Fig 3   r(F0, sigma) over nulls'] = f3['r']
    f4 = load('fig4ab')
    if f4:
        for k, r in enumerate(f4['rates6'], 1):
            v[f'SI S8   rotation rate {k}'] = r
        v['SI S13  slope log c_i ~ log k_i'] = f4['slope_circ']
        v['SI S13  R2'] = f4['r2_circ']
        v['Fig S1a slope log sigma_i ~ log k_i'] = f4['slope_sigma']
        v['Fig S1a R2'] = f4['r2_sigma']
        for c, z in f4['z_real'].items():
            v[f'Table SVI {c} real z'] = z
        for c, d in f4['table_six'].items():
            v[f'Table SIX {c} z'] = d['z']
        for m, d in f4['table_sviii'].items():
            v[f'Table SVIII rho(c_i, {m})'] = d['rho_c']
    sn = load('snull')
    if sn:
        for c, d in sn.items():
            v[f'Table SVI {c} null z'] = d['mean']
    dev = load('dev')
    if dev:
        for d in dev:
            v[f"Fig 4c  stage {d['stage']} suppression %"] = d['suppression_pct']
    sc = load('swapconv')
    if sc:
        for f in (1, 5):
            v[f'SI S2   strength null {f}x swaps, suppression %'] = \
                sc[f'factor_{f}']['suppression_pct']
    return v, f4, dev


def main():
    values, f4, dev = collect()
    rows, n_bad = [], 0
    print(f"{'quantity':<48}{'this run':>12}{'reference':>12}   ")
    print('-' * 78)
    for name, ref in REF.items():
        if name not in values:
            continue
        val = values[name]
        ok = abs(val - ref) <= TOL
        n_bad += not ok
        rows.append((name, val, ref, ok))
        print(f"{name:<48}{val:>12.4f}{ref:>12.4f}   {'OK' if ok else 'DIFF'}")

    if dev:
        for d in dev:
            got = (d['rank_RIA'], d['rank_AVA'], d['rank_AIY'])
            ref = REF_RANKS[d['stage']]
            ok = got == ref
            n_bad += not ok
            name = f"Fig 4d  stage {d['stage']} rank RIA/AVA/AIY"
            rows.append((name, '/'.join(map(str, got)), '/'.join(map(str, ref)), ok))
            print(f"{name:<48}{rows[-1][1]:>12}{rows[-1][2]:>12}   {'OK' if ok else 'DIFF'}")

    if f4:
        dz = float(np.abs(np.round(np.array(f4['mode_circuit_z']), 2) - REF_S1B).max())
        ok = dz < 1e-9
        n_bad += not ok
        rows.append(('Fig S1b mode x circuit z, max |diff|', dz, 0.0, ok))
        print(f"{'Fig S1b mode x circuit z, max |diff|':<48}{dz:>12.4f}{0:>12.4f}"
              f"   {'OK' if ok else 'DIFF'}")

    print('-' * 78)
    print(f'{len(rows)} quantities checked, {n_bad} differ from the reference run')

    out = RESULTS_DIR / 'summary.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['quantity', 'this_run', 'reference', 'match'])
        for name, val, ref, ok in rows:
            w.writerow([name, f'{val:.6g}' if isinstance(val, float) else val, ref, ok])
    print(f'written: {out}')


if __name__ == '__main__':
    main()
