"""
Run every computation behind the figures and write the results to results/*.json.

    python code/compute.py                # all parts
    python code/compute.py fig2 dev       # selected parts only

Parts (each writes results/<part>.json and can be run independently, e.g. in
parallel processes, because each has its own seed):

    fig1b    sigma vs directed fraction lambda                        (Fig 1b)
    fig2     connectome vs four null ensembles                        (Fig 2, Fig S2)
    fig3     sigma vs trophic incoherence F0 on degree nulls          (Fig 3)
    fig4ab   per-neuron dissipation, circulation, circuit alignment   (Fig 4a,b, Fig S1,
             Tables SVIII, SIX)
    snull    circuit alignment on strength-preserving nulls           (Fig 4b, Table SVI)
    dev      eight developmental stages of Witvliet et al.            (Fig 4c,d)
    swapconv swap-budget check for the strength null (1x vs 5x)       (SI S2 text)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import connectome as cx

_HERE = Path(__file__).resolve().parent
# repository layout: <root>/code/*.py -> results go to <root>/results.
# If the scripts are simply dropped into one folder, results go next to them.
ROOT = _HERE.parent if _HERE.name == 'code' else _HERE
RESULTS_DIR = ROOT / 'results'

# The connectomes are read through the OpenWorm ConnectomeToolbox (cect), which
# ships the spreadsheets; nothing has to be downloaded by hand. Point
# LOCAL_DATA at a folder holding the Witvliet files to read them from there
# instead.
LOCAL_DATA = None

# ---- Settings used for the published results ------------------------------
N_NULL_FIG2 = 1000        # nulls per ensemble, Fig 2
N_NULL_FIG3 = 1000        # degree nulls, Fig 3
N_NULL_STRENGTH = 1000    # strength nulls, Fig 4(b)
N_NULL_DEV = 1000         # degree nulls per developmental stage, Fig 4(c)
N_PERM = 2000             # permutations per circuit-alignment z
N_BOOT = 2000             # bootstrap resamples for the Fig 4(c) CI

SWAP_FACTOR_COOK = 1      # successful swaps = factor x chemical edge count
SWAP_FACTOR_DEV = 10      # Witvliet stages (stable between 5x and 50x)

SEED_FIG2 = 42            # one generator shared by the four ensembles, in order
SEED_FIG3 = 7
SEED_PERM = 300           # circuit permutation tests; strength nulls use 301
SEED_MODES = 100          # Fig S1(b): seed = 100 + 10 * mode + circuit
SEED_DEV = 50             # stage k uses 50 + k
SEED_BOOT = 1000          # stage k uses 1000 + k
SEED_SWAPCONV = 3000      # sample r uses 3000 + r

N_MODES_FIG4 = 4          # rotation planes pooled into c_i
N_MODES_FIGS1 = 5         # rotation planes shown individually in Fig S1(b)


# ---------------------------------------------------------------------------
def progress(it, label):
    """Minimal progress line (no external dependency)."""
    it = list(it)
    n, t0 = len(it), time.time()
    for k, x in enumerate(it, 1):
        yield x
        if k % max(1, n // 20) == 0 or k == n:
            el = time.time() - t0
            print(f'\r  {label:<24} {k:>5}/{n}  {el:6.0f}s  '
                  f'(~{el / k * (n - k):5.0f}s left)', end='', flush=True)
    print()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def save(part, obj):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f'{part}.json'
    path.write_text(json.dumps(_jsonable(obj), indent=1))
    print(f'  -> {path.relative_to(ROOT)}')


def load_cook():
    names, A, G, s = cx.load_cook()
    print(f'[data] Cook 2019: N={len(names)}, chemical edges={np.count_nonzero(A)}, '
          f'self-loops={np.count_nonzero(np.diag(A))}, gap edges={np.count_nonzero(G)}, '
          f'inhibitory={int((s < 0).sum())}')
    return names, A, G, s


# ---------------------------------------------------------------------------
def run_fig1b(names, A, G, s):
    lambdas, sig = cx.sigma_lambda_curve(A, G, s)
    print(f'[Fig 1b] sigma(0)={sig[0]:.2e}, sigma(1)={sig[-1]:.6f}')
    save('fig1b', {'lambda': lambdas, 'sigma': sig})


def run_fig2(names, A, G, s):
    rng = np.random.default_rng(SEED_FIG2)
    sigma_real = cx.compute_sigma(A, G, s)
    h = cx.trophic_levels(A)
    out = {'sigma_real': sigma_real, 'trophic_h': h, 'null': {}}
    for kind in cx.NULL_KINDS:
        sig = []
        for _ in progress(range(N_NULL_FIG2), f'Fig 2 {kind}'):
            v = cx.compute_sigma(cx.make_null(A, kind, rng, h=h,
                                              swap_factor=SWAP_FACTOR_COOK), G, s)
            if v is not None:
                sig.append(v)
        sig = np.array(sig)
        supp = 100 * (sig.mean() - sigma_real) / sig.mean()
        out['null'][kind] = {'sigma': sig, 'mean': sig.mean(), 'sd': sig.std(),
                             'suppression_pct': supp}
        print(f'  {kind:<20} null {sig.mean():.6f} +/- {sig.std():.6f}  '
              f'suppression {supp:.1f}%')
    save('fig2', out)


def run_fig3(names, A, G, s):
    rng = np.random.default_rng(SEED_FIG3)
    F0, sig = [], []
    for _ in progress(range(N_NULL_FIG3), 'Fig 3 degree null'):
        An = cx.make_null(A, 'degree', rng, swap_factor=SWAP_FACTOR_COOK)
        v = cx.compute_sigma(An, G, s)
        if v is not None:
            F0.append(cx.trophic_incoherence(An))
            sig.append(v)
    F0, sig = np.array(F0), np.array(sig)
    out = {'F0_real': cx.trophic_incoherence(A), 'sigma_real': cx.compute_sigma(A, G, s),
           'F0_null': F0, 'sigma_null': sig, 'r': np.corrcoef(F0, sig)[0, 1]}
    print(f"[Fig 3] F0 real={out['F0_real']:.4f}, null={F0.mean():.4f}, r={out['r']:+.3f}")
    save('fig3', out)


def run_fig4ab(names, A, G, s):
    import networkx as nx
    from scipy.stats import spearmanr

    circuits = cx.build_circuits(names)
    sigma = cx.compute_sigma(A, G, s)
    sigma_i = cx.compute_sigma_per_neuron(A, G, s)
    k_i = A.sum(axis=1) + A.sum(axis=0) + G.sum(axis=1)    # weighted total degree

    # Fig 4(b): alignment of the pooled circulation participation c_i
    c_i, _, _, w = cx.circulation_modes(A, G, s, n_modes=N_MODES_FIG4)
    z_real = {c: cx.circuit_alignment_z(c_i, idx, SEED_PERM, N_PERM)[0]
              for c, idx in circuits.items()}

    # Two different degree-corrected residuals (do not mix them up):
    #   Fig S1(a) colour : residual of log sigma_i on log k_i
    #   SI S13, Table SIX: residual of log c_i on log k_i
    slope_s, r2_s, zres_sigma = cx.log_residual(sigma_i, k_i)
    slope_c, r2_c, zres_circ = cx.log_residual(c_i, k_i)
    table_six = {c: dict(zip(('z', 'mean'),
                             cx.circuit_alignment_z(zres_circ, idx, SEED_PERM, N_PERM)))
                 for c, idx in circuits.items()}

    # Fig S1(b): alignment of each leading rotation plane separately
    _, rates5, part, _ = cx.circulation_modes(A, G, s, n_modes=N_MODES_FIGS1)
    cnames = list(circuits)
    Z = np.array([[cx.circuit_alignment_z(part[m], circuits[c],
                                          SEED_MODES + 10 * m + ci, N_PERM)[0]
                   for ci, c in enumerate(cnames)] for m in range(N_MODES_FIGS1)])

    # Table SVIII: Spearman rho with centralities. Only 'Weighted degree' uses
    # weights; the others are computed on the unweighted undirected graph.
    U = np.abs(A) + np.abs(A.T) + np.abs(G)
    np.fill_diagonal(U, 0)
    Gb = nx.from_numpy_array((U > 0).astype(float))
    col = lambda d: np.array([v for _, v in sorted(d.items())])
    measures = {'Weighted degree': k_i,
                'Eigenvector centrality': col(nx.eigenvector_centrality_numpy(Gb, weight=None)),
                'k-core number': col(nx.core_number(nx.Graph(Gb))),
                'Closeness': col(nx.closeness_centrality(Gb)),
                'Betweenness': col(nx.betweenness_centrality(Gb)),
                'Clustering coefficient': col(nx.clustering(Gb))}
    table_sviii = {m: {'rho_c': spearmanr(c_i, x).statistic,
                       'rho_sigma': spearmanr(sigma_i, x).statistic}
                   for m, x in measures.items()}

    order = np.argsort(sigma_i)[::-1]
    print(f'[Fig 4a] sum sigma_i - sigma = {sigma_i.sum() - sigma:+.1e}, '
          f'min sigma_i = {sigma_i.min():.2e}, '
          f"top 5: {', '.join(names[i] for i in order[:5])}")
    print(f'[Fig S1a] log sigma_i ~ log k_i: slope {slope_s:.3f}, R2 {r2_s:.3f}')
    print(f'[SI S13] log c_i ~ log k_i: slope {slope_c:.3f}, R2 {r2_c:.3f}')
    print('[Fig 4b] real z: ' + ', '.join(f'{c} {z:+.2f}' for c, z in z_real.items()))

    save('fig4ab', {'names': names, 'sigma': sigma, 'sigma_i': sigma_i, 'k_i': k_i,
                    'c_i': c_i, 'rates6': cx.leading_rates(w, 6),
                    'slope_sigma': slope_s, 'r2_sigma': r2_s, 'zres_sigma': zres_sigma,
                    'slope_circ': slope_c, 'r2_circ': r2_c, 'zres_circ': zres_circ,
                    'z_real': z_real, 'table_six': table_six,
                    'mode_rates': rates5, 'mode_circuit_z': Z, 'circuit_names': cnames,
                    'table_sviii': table_sviii})


def run_snull(names, A, G, s):
    circuits = cx.build_circuits(names)
    rng = np.random.default_rng(SEED_PERM + 1)
    zs = {c: [] for c in circuits}
    for _ in progress(range(N_NULL_STRENGTH), 'Fig 4b strength null'):
        An = cx.make_null(A, 'strength', rng, swap_factor=SWAP_FACTOR_COOK)
        c_n = cx.circulation_modes(An, G, s, n_modes=N_MODES_FIG4)[0]
        for c, idx in circuits.items():
            zs[c].append(cx.circuit_alignment_z(c_n, idx, SEED_PERM, N_PERM)[0])
    out = {c: {'mean': np.mean(v), 'sd': np.std(v), 'z': v} for c, v in zs.items()}
    print('[Fig 4b] null z: ' + ', '.join(f"{c} {d['mean']:+.2f}+/-{d['sd']:.2f}"
                                          for c, d in out.items()))
    save('snull', out)


def run_dev(names, A, G, s):
    """Witvliet stages. Degree nulls use a plain swap (no weight / hierarchy
       constraint) with max_tries_factor = 10."""
    stages = []
    for stage in range(1, 9):
        nm, Ak, Gk, sk = cx.load_witvliet(stage, data_dir=LOCAL_DATA)
        sigma_real = cx.compute_sigma(Ak, Gk, sk)

        rng = np.random.default_rng(SEED_DEV + stage)
        edges = cx.to_edges(Ak)
        n_target = int(SWAP_FACTOR_DEV * len(edges))
        sig = []
        for _ in progress(range(N_NULL_DEV), f'stage {stage} null'):
            new, _ = cx.directed_double_edge_swap(edges, n_target, rng, max_tries_factor=10)
            v = cx.compute_sigma(cx.to_matrix(new, len(nm)), Gk, sk)
            if v is not None:
                sig.append(v)
        sig = np.array(sig)
        supp = (sig.mean() - sigma_real) / sig.mean()

        brng = np.random.default_rng(SEED_BOOT + stage)
        boot = []
        for _ in range(N_BOOT):
            m = brng.choice(sig, size=len(sig), replace=True).mean()
            boot.append((m - sigma_real) / m)
        lo, hi = np.percentile(boot, [2.5, 97.5])

        sigma_i = cx.compute_sigma_per_neuron(Ak, Gk, sk)
        ranked = [nm[i] for i in np.argsort(sigma_i)[::-1]]
        rank = lambda p: min(r + 1 for r, n in enumerate(ranked) if n.startswith(p))
        d = {'stage': stage, 'N': len(nm), 'n_inhibitory': int((sk < 0).sum()),
             'sigma_real': sigma_real, 'null_mean': sig.mean(),
             'suppression_pct': 100 * supp, 'ci_pct': [100 * lo, 100 * hi],
             'rank_RIA': rank('RIA'), 'rank_AVA': rank('AVA'), 'rank_AIY': rank('AIY')}
        print(f"  stage {stage}: N={d['N']}, inh={d['n_inhibitory']}, "
              f"suppression {d['suppression_pct']:.1f}% [{100*lo:.1f}, {100*hi:.1f}], "
              f"rank RIA/AVA/AIY {d['rank_RIA']}/{d['rank_AVA']}/{d['rank_AIY']}")
        stages.append(d)
    save('dev', stages)


def run_swapconv(names, A, G, s):
    """SI S2: the strength-null suppression is insensitive to a fivefold larger
       swap budget. Independent seed per sample."""
    sigma_real = cx.compute_sigma(A, G, s)
    out = {}
    for f in (1, 5):
        v = np.array([cx.compute_sigma(
            cx.make_null(A, 'strength', np.random.default_rng(SEED_SWAPCONV + r),
                         swap_factor=f), G, s)
            for r in progress(range(1000), f'swap factor {f}')])
        out[f'factor_{f}'] = {'null_mean': v.mean(), 'null_sd': v.std(ddof=1),
                              'suppression_pct': 100 * (1 - sigma_real / v.mean())}
        print(f"  factor {f}: suppression {out[f'factor_{f}']['suppression_pct']:.2f}%")
    save('swapconv', out)


PARTS = {'fig1b': run_fig1b, 'fig2': run_fig2, 'fig3': run_fig3, 'fig4ab': run_fig4ab,
         'snull': run_snull, 'dev': run_dev, 'swapconv': run_swapconv}

if __name__ == '__main__':
    todo = sys.argv[1:] or list(PARTS)
    bad = [p for p in todo if p not in PARTS]
    if bad:
        sys.exit(f'unknown part(s): {bad}. Choose from {list(PARTS)}')
    cook = load_cook()
    for p in todo:
        t0 = time.time()
        print(f'=== {p} ===')
        PARTS[p](*cook)
        print(f'    {p} done in {time.time() - t0:.0f}s')
