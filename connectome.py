"""
Core routines for
  "Suppression and selective allocation of irreversibility in a neural connectome"

Contents
  1. Data loading     Cook et al. (2019) adult hermaphrodite and the Witvliet et al.
                      (2021) developmental series, both read through the OpenWorm
                      ConnectomeToolbox (cect) and reduced to the same somatic set.
  2. Linear model     drift A = I - W * margin / max Re eig(W), unit noise B = I,
                      stationary covariance C from  A C + C A^T = 2B,
                      entropy-production capacity  sigma = Tr[M^T B^-1 M C],
                      M = A - B C^-1  (SI Sec. S1).
  3. Trophic levels   Levine / MacKay trophic levels and incoherence F0.
  4. Null models      degree, hierarchy, strength and strength+hierarchy
                      preserving double-edge swaps of the chemical layer (SI Sec. S2).
  5. Circulation      circulation modes of Omega = MC and behavioural-circuit
                      alignment by size-matched permutation.

Every random draw goes through an explicit numpy Generator, so results are
fully determined by the seeds set in compute.py.
"""
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_lyapunov, eigvals, eig

SPECTRAL_MARGIN = 0.8          # max Re eig(W) is rescaled to this value (SI S1)

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
# Somatic neuron set (SI S1): all cells in the data, minus muscles, lower-case
# (non-neuronal) cells, the 20 pharyngeal neurons and the GLR glia.
# Cook 2019: 473 cells -> 286 somatic neurons.
PHARYNGEAL = {'I1L', 'I1R', 'I2L', 'I2R', 'I3', 'I4', 'I5', 'I6', 'M1', 'M2L',
              'M2R', 'M3L', 'M3R', 'M4', 'M5', 'MCL', 'MCR', 'MI', 'NSML', 'NSMR'}
GLIA = {'GLRDL', 'GLRDR', 'GLRL', 'GLRR', 'GLRVL', 'GLRVR', 'hmc', 'HMC'}

# Dale's principle: known GABAergic neurons get sign -1, all others +1 (SI S1).
GABA = {'DD1', 'DD2', 'DD3', 'DD4', 'DD5', 'DD6',
        'VD1', 'VD2', 'VD3', 'VD4', 'VD5', 'VD6', 'VD7', 'VD8', 'VD9',
        'VD10', 'VD11', 'VD12', 'VD13',
        'RMED', 'RMEL', 'RMER', 'RMEV', 'RIS', 'AVL', 'DVB'}


def is_neuron_like(name):
    """Upper-case cell names that are not body-wall muscles."""
    if not name or name[0].islower():
        return False
    if name[:2] in ('MD', 'MV') or 'BWM' in name:
        return False
    return True


def somatic_set(raw_names):
    return sorted({n for n in raw_names if is_neuron_like(n)} - PHARYNGEAL - GLIA)


def _symmetrise_gap(G):
    """Gap junctions must enter W symmetrically.
       Cook 'gap jn symmetric' sheet is already symmetric -> unchanged.
       Witvliet lists each electrical synapse once, in one direction -> G + G^T.
       (Using 0.5 (G + G^T) there would halve every gap weight.)"""
    if np.allclose(G, G.T):
        return G
    one_sided = int(((G > 0) & (G.T == 0)).sum())
    two_sided = int(((G > 0) & (G.T > 0)).sum())
    return G + G.T if one_sided > two_sided else 0.5 * (G + G.T)


def _build(names, conns):
    """conns: iterable of (pre, post, 'chemical' | 'electrical', weight)."""
    idx = {n: i for i, n in enumerate(names)}
    N = len(names)
    A, G = np.zeros((N, N)), np.zeros((N, N))
    for pre, post, typ, w in conns:
        if pre in idx and post in idx:
            if typ == 'chemical':
                A[idx[pre], idx[post]] += w
            elif typ == 'electrical':
                G[idx[pre], idx[post]] += w
    s = np.array([-1.0 if n in GABA else 1.0 for n in names])
    return A, _symmetrise_gap(G), s


def cect_data_dir():
    """Directory holding the connectome files shipped with the OpenWorm
       ConnectomeToolbox (cect)."""
    import cect
    return Path(cect.__file__).resolve().parent / 'data'


def load_cook(sex='Hermaphrodite'):
    """Cook et al. (2019) whole-animal connectome, read through the OpenWorm
       ConnectomeToolbox (`pip install cect`), which ships the corrected
       July 2020 adjacency spreadsheet and applies its own cell-name
       conventions (e.g. VB01 -> VB1).

       Returns names, A, G, s with
         A[i, j]  chemical contact weight, presynaptic i -> postsynaptic j
         G        symmetric gap-junction weight
         s        Dale sign per neuron (+1 / -1)
    """
    from cect.readers.Cook2019DataReader import Cook2019DataReader, SEX_SPECIFIC_SHEETS

    keys = list(SEX_SPECIFIC_SHEETS)
    key = next((k for k in keys if sex[:4].lower() in k.lower()), keys[0])
    _, _, _, raw = Cook2019DataReader(sex=key).read_all_data()

    conns = []
    for c in raw:
        t = str(c.syntype)
        typ = ('chemical' if 'Chemical' in t else
               'electrical' if ('Electrical' in t or 'Gap' in t) else None)
        if typ:
            conns.append((c.pre_cell, c.post_cell, typ, c.number))
    names = somatic_set({c[0] for c in conns} | {c[1] for c in conns})
    A, G, s = _build(names, conns)
    return names, A, G, s


def load_witvliet(stage, data_dir=None):
    """One developmental stage (1 = L1 at hatching ... 8 = adult) of Witvliet
       et al. (2021), read from the edge-list spreadsheets shipped with cect
       (columns pre, post, type, synapses). Same somatic filter as for Cook.

       data_dir overrides the cect package directory if the files are kept
       locally.
    """
    import pandas as pd
    d = Path(data_dir) if data_dir else cect_data_dir()
    hits = sorted(d.glob(f'witvliet_2020_{stage}*.xlsx'))
    if not hits:
        raise FileNotFoundError(f'no Witvliet stage {stage} file in {d}')
    df = pd.read_excel(hits[0])
    names = somatic_set(set(df['pre']) | set(df['post']))
    A, G, s = _build(names, zip(df['pre'], df['post'], df['type'], df['synapses']))
    return names, A, G, s


# ---------------------------------------------------------------------------
# 2. Linear model and entropy production
# ---------------------------------------------------------------------------
def coupling_matrix(A, G, s):
    """W[i, j] = input to i from j: signed chemical input plus gap junctions."""
    return A.T * s[None, :] + G


def _drift_and_covariance(W, margin=SPECTRAL_MARGIN):
    """Return (drift, C) or None if the rescaled system is not stable."""
    N = W.shape[0]
    ab = np.max(eigvals(W).real)
    if ab <= 0:
        return None
    Ad = np.eye(N) - W * (margin / ab)
    if np.min(eigvals(Ad).real) <= 1e-9:
        return None
    C = solve_continuous_lyapunov(-Ad, -2 * np.eye(N))      # B = I
    return Ad, C


def _sigma_from_W(W, margin=SPECTRAL_MARGIN):
    out = _drift_and_covariance(W, margin)
    if out is None:
        return None
    Ad, C = out
    M = Ad - np.linalg.inv(C)
    return float(np.real(np.trace(M.T @ M @ C)))


def compute_sigma(A, G, s, margin=SPECTRAL_MARGIN):
    """sigma = Tr[M^T B^-1 M C],  M = A - B C^-1,  A C + C A^T = 2B,  B = I."""
    return _sigma_from_W(coupling_matrix(A, G, s), margin)


def compute_sigma_per_neuron(A, G, s, margin=SPECTRAL_MARGIN):
    """Per-neuron dissipation  sigma_i = [B^-1/2 M C M^T B^-1/2]_ii  (B = I).

       From sigma = < |B^-1/2 M x|^2 >, sigma_i is the share carried by the
       i-th component of the noise-scaled current. It is the squared row norm
       of B^-1/2 M C^1/2, hence sigma_i >= 0, and sum_i sigma_i = sigma.
    """
    out = _drift_and_covariance(coupling_matrix(A, G, s), margin)
    if out is None:
        return None
    Ad, C = out
    M = Ad - np.linalg.inv(C)
    return np.real(np.einsum('ij,jk,ik->i', M, C, M))


def sigma_lambda_curve(A, G, s, n_points=21, margin=SPECTRAL_MARGIN):
    """Fig 1(b): W(lambda) = W_sym + lambda * W_asym. lambda = 0 is detailed
       balance (sigma = 0); lambda = 1 is the connectome."""
    W = coupling_matrix(A, G, s)
    W_sym, W_asym = 0.5 * (W + W.T), 0.5 * (W - W.T)
    lambdas = np.linspace(0, 1, n_points)
    return lambdas, [_sigma_from_W(W_sym + lam * W_asym, margin) for lam in lambdas]


# ---------------------------------------------------------------------------
# 3. Trophic levels
# ---------------------------------------------------------------------------
def trophic_levels(A):
    """Trophic levels h from the chemical layer, shifted so min(h) = 0."""
    k_out, k_in = A.sum(axis=1), A.sum(axis=0)
    Lam = np.diag(k_in + k_out) - A - A.T
    h = np.linalg.pinv(Lam) @ (k_in - k_out)
    return h - h.min()


def trophic_incoherence(A):
    """F0 = sum_ij A_ij (h_j - h_i - 1)^2 / sum_ij A_ij   (SI Eq. S2)."""
    h = trophic_levels(A)
    ii, jj = np.nonzero(A)
    d2 = (h[jj] - h[ii] - 1.0) ** 2
    return float(np.sum(A[ii, jj] * d2) / np.sum(A[ii, jj]))


def edge_category(h, i, j, tol=1e-9):
    d = h[j] - h[i]
    return 'fwd' if d > tol else ('bwd' if d < -tol else 'lat')


# ---------------------------------------------------------------------------
# 4. Null models (double-edge swaps of the chemical layer)
# ---------------------------------------------------------------------------
def to_edges(A):
    ii, jj = np.nonzero(A)
    return [(int(i), int(j), float(A[i, j])) for i, j in zip(ii, jj)]


def to_matrix(edges, N):
    A = np.zeros((N, N))
    for i, j, w in edges:
        A[i, j] = w
    return A


def directed_double_edge_swap(edges, n_swaps, rng, same_weight_only=False,
                              h=None, max_tries_factor=50):
    """(i1 -> j1, i2 -> j2)  ->  (i1 -> j2, i2 -> j1).

       Preserves every in- and out-degree. Four distinct nodes are required,
       so the 37 self-loops are never moved.

       same_weight_only  swap only edges with equal weight, so in/out-strength
                         is preserved exactly. A weight class is drawn
                         uniformly first, then a pair inside it (weight 1
                         dominates the edge list; without this, heavy edges
                         would almost never move).
       h                 hierarchy constraint: accept only if both new edges
                         keep the forward / backward / lateral category of
                         the edges they replace.
    """
    edges = list(edges)
    m = len(edges)
    edge_set = {(i, j) for i, j, _ in edges}

    groups = None
    if same_weight_only:
        by_w = {}
        for k, (_, _, w) in enumerate(edges):
            by_w.setdefault(w, []).append(k)
        groups = [np.array(g) for g in by_w.values() if len(g) >= 2]

    n_ok, n_try = 0, 0
    max_tries = n_swaps * max_tries_factor
    while n_ok < n_swaps and n_try < max_tries:
        n_try += 1
        if groups is not None:
            if not groups:
                break
            g = groups[rng.integers(len(groups))]
            pick = rng.choice(len(g), size=2, replace=False)
            k1, k2 = int(g[pick[0]]), int(g[pick[1]])
        else:
            k1, k2 = rng.integers(0, m, size=2)
            if k1 == k2:
                continue
        i1, j1, w1 = edges[k1]
        i2, j2, w2 = edges[k2]
        if len({i1, j1, i2, j2}) < 4:
            continue
        if same_weight_only and abs(w1 - w2) > 1e-12:
            continue
        if h is not None and (edge_category(h, i1, j2) != edge_category(h, i1, j1) or
                              edge_category(h, i2, j1) != edge_category(h, i2, j2)):
            continue
        if (i1, j2) in edge_set or (i2, j1) in edge_set:
            continue
        edge_set.discard((i1, j1)); edge_set.discard((i2, j2))
        edge_set.add((i1, j2)); edge_set.add((i2, j1))
        edges[k1] = (i1, j2, w1); edges[k2] = (i2, j1, w2)
        n_ok += 1
    return edges, n_ok


NULL_KINDS = ('degree', 'hierarchy', 'strength', 'strength_hierarchy')


def make_null(A, kind, rng, h=None, swap_factor=1):
    """One null network. Gap junctions and Dale signs are held fixed; only the
       chemical layer is rewired, with a target of swap_factor x (edge count)
       successful swaps."""
    edges = to_edges(A)
    same_w = kind in ('strength', 'strength_hierarchy')
    use_h = kind in ('hierarchy', 'strength_hierarchy')
    tries = 50 * (5 if same_w else 1) * (5 if use_h else 1)
    new_edges, _ = directed_double_edge_swap(
        edges, int(swap_factor * len(edges)), rng, same_weight_only=same_w,
        h=(h if use_h else None), max_tries_factor=tries)
    return to_matrix(new_edges, A.shape[0])


# ---------------------------------------------------------------------------
# 5. Circulation modes and circuit alignment
# ---------------------------------------------------------------------------
def _paired_modes(w, n):
    """Omega is real antisymmetric, so its eigenvalues come in pairs
       +/- i|lambda|. Each pair is ONE rotation plane: keep one index per pair,
       ordered by rotation rate |Im lambda|."""
    order = np.argsort(-np.abs(w.imag))
    seen, keep = set(), []
    for k in order:
        if k in seen:
            continue
        rest = [j for j in order if j not in seen and j != k]
        partner = min(rest, key=lambda j: abs(w[j] - np.conj(w[k]))) if rest else None
        seen |= ({k, partner} if partner is not None else {k})
        keep.append(k)
        if len(keep) >= n:
            break
    return keep


def circulation_modes(A, G, s, n_modes=4, margin=SPECTRAL_MARGIN):
    """Leading rotation planes of Omega = MC.

       Returns
         c      rate-weighted circulation participation per neuron (sums to 1)
         rates  rotation rates of the n_modes leading planes
         part   (n_modes, N) per-mode participation, |Re v| + |Im v|, normalised
         w      all eigenvalues of Omega
    """
    N = A.shape[0]
    W = coupling_matrix(A, G, s)
    ab = np.max(eigvals(W).real)
    Ad = np.eye(N) - W * (margin / ab)
    C = solve_continuous_lyapunov(-Ad, -2 * np.eye(N))
    MC = (Ad - np.linalg.inv(C)) @ C
    Omega = 0.5 * (MC - MC.T)
    w, V = eig(Omega)

    rates, part = [], []
    for k in _paired_modes(w, n_modes):
        p = np.abs(V[:, k].real) + np.abs(V[:, k].imag)
        if p.sum() > 0:
            p = p / p.sum()
        rates.append(float(abs(w[k].imag)))
        part.append(p)
    part = np.array(part)
    c = (np.array(rates)[:, None] * part).sum(axis=0)
    if c.sum() > 0:
        c = c / c.sum()
    return c, rates, part, w


def leading_rates(w, n=6):
    return [float(abs(w[k].imag)) for k in _paired_modes(w, n)]


# Behavioural circuits (sizes 22 / 27 / 16 / 10 / 8).
CIRCUITS = {
    'Forward':         {'lr': ['AVB', 'PVC'], 'numbered': [('DB', 1, 7), ('VB', 1, 11)]},
    'Backward':        {'lr': ['AVA', 'AVD', 'AVE'], 'numbered': [('DA', 1, 9), ('VA', 1, 12)]},
    'Turn/steer':      {'lr': ['RIA', 'RIB', 'RIV', 'SMDD', 'SMDV', 'RMDD', 'RMDV', 'RMD'],
                        'numbered': []},
    'Nocicept/escape': {'lr': ['PVD', 'PLM', 'FLP', 'ALM', 'ASH'], 'numbered': []},
    'Chemosens.integ': {'lr': ['AIY', 'AIZ', 'AIA', 'AIB'], 'numbered': []},
}
CIRCUIT_SIZE = {'Forward': 22, 'Backward': 27, 'Turn/steer': 16,
                'Nocicept/escape': 10, 'Chemosens.integ': 8}


def build_circuits(names):
    """Circuit name -> list of neuron indices."""
    pos = {n: i for i, n in enumerate(names)}
    out = {}
    for cname, spec in CIRCUITS.items():
        members = [p + side for p in spec['lr'] for side in ('L', 'R')]
        members += [f'{pre}{k}' for pre, a, b in spec['numbered'] for k in range(a, b + 1)]
        idx = [pos[m] for m in members if m in pos]
        if len(idx) != CIRCUIT_SIZE[cname]:
            print(f"  [warn] {cname}: {len(idx)} members found, "
                  f"expected {CIRCUIT_SIZE[cname]}")
        out[cname] = idx
    return out


def circuit_alignment_z(values, idx, rng_seed, n_perm=2000):
    """Size-matched permutation test (SI S1): z of the circuit mean against
       n_perm random neuron sets of the same size."""
    rng = np.random.default_rng(rng_seed)
    N, n = len(values), len(idx)
    real = values[idx].mean()
    rand = np.array([values[rng.choice(N, size=n, replace=False)].mean()
                     for _ in range(n_perm)])
    return float((real - rand.mean()) / rand.std()), float(real)


def log_residual(y, x):
    """Regress log y on log x; return slope, R^2, and the standardised
       residual per neuron (NaN where x or y is not positive)."""
    ok = (x > 0) & (y > 0)
    lx, ly = np.log(x[ok]), np.log(y[ok])
    slope, icpt = np.polyfit(lx, ly, 1)
    r = ly - (slope * lx + icpt)
    z = np.full(len(x), np.nan)
    z[ok] = (r - r.mean()) / r.std()
    r2 = 1 - np.sum(r ** 2) / np.sum((ly - ly.mean()) ** 2)
    return float(slope), float(r2), z
