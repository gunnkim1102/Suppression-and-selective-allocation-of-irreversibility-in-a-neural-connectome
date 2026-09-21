"""
Draw the figures from results/*.json (run compute.py first).

    python code/make_figures.py

Writes figures/fig1b, fig2, fig3, fig4, figS1, figS2, figS3 as PDF and PNG.
Fig 1(a) is a schematic and is not generated here.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, ScalarFormatter

from compute import ROOT, RESULTS_DIR, load_cook

FIG_DIR = ROOT / 'figures'
FORMATS = ('pdf', 'png')
plt.rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'Times', 'Liberation Serif',
                                    'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 9,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

CIRCUIT_LABEL = {'Nocicept/escape': 'Nocicept', 'Chemosens.integ': 'Chemosens'}
# Fig 4(a) colour classes (not the same as the five circuits of Fig 4(b))
COMMAND = ('AVA', 'AVB', 'AVD', 'AVE', 'PVC')
INTEGRATION = ('AIY', 'AIZ', 'AIB', 'RIA')
CATEGORY_COLOR = {'command': '#d62728', 'integration': '#ff7f0e', 'other': '#7f7f7f'}


def load(part):
    return json.loads((RESULTS_DIR / f'{part}.json').read_text())


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in FORMATS:
        fig.savefig(FIG_DIR / f'{name}.{ext}', dpi=300)
    plt.close(fig)
    print(f'  figures/{name}.{{{",".join(FORMATS)}}}')


def tag(ax, label):
    ax.text(-0.02, 1.02, label, transform=ax.transAxes, ha='right', va='bottom',
            fontweight='bold', fontsize=10)


# ---------------------------------------------------------------------------
def fig1b():
    d = load('fig1b')
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.plot(d['lambda'], d['sigma'], '-o', color='#7b3294', lw=1.2, ms=3)
    ax.plot(0, 0, 'o', color='red', ms=8, fillstyle='none', mew=1.0)
    ax.annotate('symmetrized:\n$\\sigma\\approx0$', xy=(0.001, 0.001),
                xytext=(0.10, 0.030), arrowprops=dict(arrowstyle='->', lw=1.0))
    ax.set_xlabel('directed fraction  $\\lambda$')
    ax.set_ylabel('$\\sigma$')
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.009, 0.176)
    save(fig, 'fig1b')


def fig2():
    d = load('fig2')
    null, s0 = d['null'], d['sigma_real']
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.2, 4.2))

    kinds = ('degree', 'hierarchy', 'strength')
    colors = ('#8c8c8c', '#7fb3d9', '#8fbf6a')
    vals = np.concatenate([null[k]['sigma'] for k in kinds] + [[s0]])
    pad = 0.05 * (vals.max() - vals.min())
    bins = np.linspace(vals.min() - pad, vals.max() + pad, 40)
    for k, c in zip(kinds, colors):
        a.hist(null[k]['sigma'], bins=bins, density=True, histtype='stepfilled',
               color=c, alpha=0.55, edgecolor=c, lw=0.8, label=f'{k} null')
    a.axvline(s0, color='red', lw=1.5, label='connectome')
    a.set_xlabel('entropy-production capacity $\\sigma$')
    a.set_ylabel('density'); a.set_yticks([])
    a.set_xticks(np.arange(0.17, 0.21 + 1e-9, 0.01))
    a.set_xlim(bins[0], bins[-1])
    a.set_ylim(0, a.get_ylim()[1] * 1.08)
    # keep the legend clear of the connectome line at the left edge
    a.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(0.07, 1.0))
    tag(a, '(a)')

    kinds4 = ('degree', 'hierarchy', 'strength', 'strength_hierarchy')
    labels = ('degree', 'hierarchy', 'strength', 'strength\n+hierarchy')
    supp = [null[k]['suppression_pct'] for k in kinds4]
    y = np.arange(4)[::-1]
    b.barh(y, supp, color='#c0453a', height=0.6)
    for yi, v in zip(y, supp):
        b.text(v + 0.3, yi, f'{v:.1f}%', va='center')
    b.set_yticks(y); b.set_yticklabels(labels)
    b.set_xlim(0, max(supp) * 1.25)
    b.set_xlabel('fractional suppression (%)')
    tag(b, '(b)')
    save(fig, 'fig2')


def fig3():
    d = load('fig3')
    x, y = np.array(d['F0_null']), np.array(d['sigma_null'])
    x0, y0 = d['F0_real'], d['sigma_real']
    slope, icpt = np.polyfit(x, y, 1)
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.scatter(x, y, s=8, color='#1f77b4', alpha=0.7, edgecolor='k', lw=0.2,
               label='degree-preserving nulls')
    xr = np.linspace(min(x0, x.min()), x.max(), 100)
    ax.plot(xr, slope * xr + icpt, '--', color='0.35', lw=1.0, label='null regression')
    ax.scatter([x0], [y0], marker='*', s=140, color='red', edgecolor='k', lw=0.6,
               zorder=5, label='connectome')
    y_pred = slope * x0 + icpt
    ax.annotate('', xy=(x0, y_pred), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2, shrinkA=6, shrinkB=0))
    ax.text(x0 + 0.02 * (x.max() - x0), 0.5 * (y0 + y_pred),
            'below\nhierarchy\nprediction', color='red', fontsize=8, va='center')
    ax.set_xlabel('trophic incoherence  $F_0$')
    ax.set_ylabel('entropy-production capacity  $\\sigma$')
    ax.legend(frameon=False, fontsize=7, loc='lower right')
    save(fig, 'fig3')


def fig4():
    f4, sn, dev = load('fig4ab'), load('snull'), load('dev')
    names, sigma_i = f4['names'], np.array(f4['sigma_i'])
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 4.8))

    # (a) top 12 per-neuron dissipation
    a = axs[0, 0]
    top = np.argsort(sigma_i)[::-1][:12]
    cat = lambda n: ('command' if n.startswith(COMMAND) else
                     'integration' if n.startswith(INTEGRATION) else 'other')
    yy = np.arange(12)
    a.barh(yy, sigma_i[top][::-1], height=0.6,
           color=[CATEGORY_COLOR[cat(names[i])] for i in top][::-1])
    a.set_yticks(yy); a.set_yticklabels([names[i] for i in top][::-1])
    a.tick_params(axis='y', length=0)
    a.set_xlabel('$\\sigma_i$')
    a.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in CATEGORY_COLOR.items()],
             loc='lower right', frameon=False, fontsize=7)
    tag(a, '(a)')

    # (b) circuit alignment: connectome vs strength-preserving null
    b = axs[0, 1]
    z = f4['z_real']
    order = sorted(z, key=z.get)
    yy = np.arange(len(order))
    real = [z[c] for c in order]
    nm = [sn[c]['mean'] for c in order]
    sd = [sn[c]['sd'] for c in order]
    b.barh(yy - 0.18, real, height=0.36, color='#c0392b', label='real')
    b.barh(yy + 0.18, nm, height=0.36, color='#93a1a1', xerr=sd,
           error_kw=dict(lw=0.8, capsize=2), label='strength null')
    span = max(real + nm) - min(real + nm)
    for yi, r, n in zip(yy, real, nm):
        if r - n > 0.5:
            b.text(max(r, n) + 0.04 * span, yi, f'+{r - n:.1f}', va='center',
                   color='#c0392b', fontsize=8)
    b.axvline(0, color='k', lw=0.8)
    b.set_yticks(yy); b.set_yticklabels([CIRCUIT_LABEL.get(c, c) for c in order])
    b.invert_yaxis()
    b.set_xlabel('circulation alignment $z$')
    b.legend(loc='upper right', frameon=False, fontsize=7)
    tag(b, '(b)')

    # (c) developmental suppression with 95 % bootstrap CI
    c = axs[1, 0]
    st = [d['stage'] for d in dev]
    sp = np.array([d['suppression_pct'] for d in dev])
    ci = np.array([d['ci_pct'] for d in dev])
    c.errorbar(st, sp, yerr=[sp - ci[:, 0], ci[:, 1] - sp], fmt='o-', capsize=3,
               color='#c0392b')
    c.set_xticks(st)
    c.set_xlabel('developmental stage'); c.set_ylabel('suppression (%)')
    tag(c, '(c)')

    # (d) dissipation rank of RIA / AVA / AIY
    d_ = axs[1, 1]
    for key, lab, col, mk in [('rank_RIA', 'RIA', '#ff7f0e', 's'),
                              ('rank_AVA', 'AVA', '#1f77b4', 'o'),
                              ('rank_AIY', 'AIY', '#9467bd', '^')]:
        d_.plot(st, [d[key] for d in dev], marker=mk, color=col, label=lab, ms=4)
    d_.set_yscale('log'); d_.invert_yaxis()
    d_.set_yticks([1, 2, 5, 10, 20, 50])
    d_.yaxis.set_major_formatter(ScalarFormatter())
    d_.set_xticks(st)
    d_.set_xlabel('developmental stage'); d_.set_ylabel('dissipation rank')
    d_.legend(frameon=False, fontsize=7, ncol=3, loc='upper right',
              bbox_to_anchor=(1.0, 0.86))      # empty band between rank 1 and 10
    tag(d_, '(d)')
    save(fig, 'fig4')


def figS1():
    f4 = load('fig4ab')
    k, sig = np.array(f4['k_i']), np.array(f4['sigma_i'])
    zr = np.array([np.nan if v is None else v for v in f4['zres_sigma']])
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.7),
                               gridspec_kw={'width_ratios': [1, 1.12]})
    ok = (k > 0) & (sig > 0)
    sc = a.scatter(k[ok], sig[ok], c=zr[ok], cmap='coolwarm', vmin=-3, vmax=3, s=7)
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel('total degree $k_i$'); a.set_ylabel('$\\sigma_i$')
    fig.colorbar(sc, ax=a, label='degree-corrected residual $z$')
    tag(a, '(a)')

    Z = np.array(f4['mode_circuit_z'])
    im = b.imshow(Z, cmap='RdBu_r', vmin=-10, vmax=10, aspect='auto')
    b.set_xticks(range(Z.shape[1]))
    b.set_xticklabels([CIRCUIT_LABEL.get(c, c) for c in f4['circuit_names']],
                      rotation=30, ha='right')
    b.set_yticks(range(Z.shape[0]))
    b.set_yticklabels([f'mode {i + 1}' for i in range(Z.shape[0])])
    b.tick_params(length=0)
    for r in range(Z.shape[0]):
        for q in range(Z.shape[1]):
            txt = f'{Z[r, q]:.0f}'
            if txt == '-0':
                txt = '0'
            b.text(q, r, txt, ha='center', va='center', fontsize=8,
                   color='white' if abs(Z[r, q]) > 6 else 'k')
    fig.colorbar(im, ax=b, label='alignment $z$')
    tag(b, '(b)')
    save(fig, 'figS1')


def figS2():
    h = np.array(load('fig2')['trophic_h'])
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    ax.hist(h, bins=30, color='#6f9fc8', edgecolor='#20405c', lw=0.5, alpha=0.85)
    mu, sd = h.mean(), h.std(ddof=1)
    ax.axvline(mu, color='red', lw=1.2, label=f'mean = {mu:.2f}')
    ax.axvspan(mu - sd, mu + sd, color='red', alpha=0.10, label=f's.d. = {sd:.2f}')
    ax.set_xlabel('trophic level  $h_i$'); ax.set_ylabel('number of neurons')
    ax.legend(frameon=False, fontsize=7)
    save(fig, 'figS2')


def figS3():
    _, A, _, _ = load_cook()
    B = A != 0
    fig, axs = plt.subplots(2, 1, figsize=(3.2, 4.1))
    for ax, lab, v_out, v_in, xlabel in [
            (axs[0], '(a)', B.sum(1), B.sum(0), 'chemical degree  $k_i$'),
            (axs[1], '(b)', A.sum(1), A.sum(0), 'chemical strength  $s_i$  (contacts)')]:
        v_out, v_in = v_out.astype(float), v_in.astype(float)
        hi = max(v_out.max(), v_in.max())
        # degree: bins of width 3; strength: 26 equal bins
        bins = np.arange(0, hi + 3, 3) if lab == '(a)' else np.linspace(0, hi, 27)
        for v, col, ls, name in [(v_out, '#d9793f', '-', 'out'), (v_in, '#4a8fc0', '--', 'in')]:
            ax.hist(v, bins=bins, histtype='step', color=col, ls=ls, lw=1.1,
                    label=f'{name}  $\\mu$={v.mean():.2f}, $\\sigma^2$={v.var(ddof=1):.0f}')
        ax.set_xlabel(xlabel); ax.set_ylabel('number of neurons'); ax.set_xlim(left=0)
        ax.legend(frameon=False, fontsize=7)
        tag(ax, lab)
    save(fig, 'figS3')


if __name__ == '__main__':
    for f in (fig1b, fig2, fig3, fig4, figS1, figS2, figS3):
        f()
