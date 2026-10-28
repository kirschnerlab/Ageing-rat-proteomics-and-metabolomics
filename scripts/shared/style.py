"""Shared figure style: purple age ramp, orange = increase / injury pole, blue = decrease / differentiated pole,
gray = CI includes zero."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.edgecolor': '#3a3a3a', 'xtick.color': '#3a3a3a', 'ytick.color': '#3a3a3a',
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': True, 'mathtext.default': 'regular',
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'})
NADP = 'NAD$^+$'
AGEC = {9: '#b9a7e3', 18: '#8a6fd1', 21: '#5b3ea8', 25: '#2e1a66'}
UP, DOWN, NS = '#eb6834', '#2a78d6', '#9a9a9a'
INK, MUTED, PALE = '#1f1f1f', '#6b6b6b', '#c4c7cb'
MM = 1 / 25.4


def title(ax, letter, text, dx=-0.16, size=7.5):
    ax.set_title(text, loc='left', pad=5, fontsize=size, color=INK)
    ax.text(dx, 1.02, letter, transform=ax.transAxes, fontsize=9.5, fontweight='bold', va='bottom', ha='left')


def age_scatter(ax, x, y, age, s=14):
    for a in (9, 18, 21, 25):
        k = age == a
        ax.scatter(x[k], y[k], s=s, c=AGEC[a], edgecolor='white', lw=0.4, zorder=3, label=f'{a} months')


def snapshot(ax, png, letter=None, text=None, whiteout=(), dx=-0.04):
    """Place a recycled image panel unchanged; whiteout boxes (fractions x0, y0, x1, y1 from top-left) hide old letters."""
    im = np.array(Image.open(png).convert('RGB'))
    h, w = im.shape[:2]
    for x0, y0, x1, y1 in whiteout:
        im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 255
    ax.imshow(im, interpolation='lanczos')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if letter:
        title(ax, letter, text or '', dx=dx)


def welch_ci(a, b):
    """Difference of means b - a with a Welch 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    from scipy import stats
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    d = b.mean() - a.mean(); se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(0.975, df)
    return d, d - t * se, d + t * se, 2 * stats.t.sf(abs(d / se), df)
