"""Real nilearn figures inside a cardkit scene (needs `pip install nilearn`; no data download).

Step 1  a glass brain whose activation follows a BOLD time course that draws itself underneath
Step 2  a connectome: nodes first, then edges, next to a bar chart of node strength drawn with cardkit

The nilearn plots are rendered once to RGBA arrays and placed with Img; the animation (cross-fades, curves, bars, text)
is cardkit.  The same scene exports to GIF and to an animated SVG.

    python brain.py           -> out/brain.gif, out/brain_animated.svg, out/brain_poster.png
    python brain.py check     -> text-overlap check
"""
import os
import sys
import warnings
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.datasets import load_mni152_template
import cardkit as ck
from cardkit import Txt, Path, Poly, Circ, Img, ramp, fade

warnings.filterwarnings("ignore")
INK, MUTED, DIM = "#1d2433", "#8a93a3", "#e3e7ee"
TEAL, VIOLET, ORANGE = "#0f9d8a", "#6c5ce7", "#e17d38"


# ---------------------------------------------------------------- nilearn -> RGBA arrays (rendered once, before animating)
def render(plot, size=(6.0, 2.4), **kw):
    """Draw a nilearn plot on a fresh figure and return it as an RGBA array (white background)."""
    fig = plt.figure(figsize=size, dpi=200, facecolor="white")
    plot(figure=fig, annotate=False, **kw)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba()).astype(float) / 255
    plt.close(fig)
    return arr


def crop(*arrs):
    """Crop images to the bounding box of their combined content (the same box for all, so they stay aligned)."""
    ink = np.min([a[..., :3].min(-1) for a in arrs], axis=0) < 0.98
    cols, rows = np.flatnonzero(ink.any(0)), np.flatnonzero(ink.any(1))
    return [a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] for a in arrs]


def blobs(peaks, sigma=7.0):
    """Synthetic statistical map in MNI space: Gaussian blobs at the given (x, y, z, height) peaks."""
    tpl = load_mni152_template(resolution=2)
    ijk = np.indices(tpl.shape).reshape(3, -1).T
    xyz = nib.affines.apply_affine(tpl.affine, ijk)
    data = np.zeros(len(xyz))
    for x, y, z, h in peaks:
        data += h * np.exp(-np.sum((xyz - (x, y, z)) ** 2, axis=1) / (2 * sigma ** 2))
    return nib.Nifti1Image(data.reshape(tpl.shape).astype(np.float32), tpl.affine)


def overlay(base, full):
    """RGBA layer that turns ``base`` into ``full`` when composited over it: what changed, with exact colours.
    Animate its opacity to fade the activation in and out on an unchanged brain outline."""
    diff = np.abs(full[..., :3] - base[..., :3]).max(-1)
    a = np.clip(diff * 4, 0, 1)
    rgb = base[..., :3] + (full[..., :3] - base[..., :3]) / np.maximum(a, 1e-6)[..., None]
    return np.dstack([np.clip(rgb, 0, 1), a])


PEAKS = [(-38, -22, 56, 1.0), (40, -20, 54, 0.8), (-4, -88, 2, 0.9), (2, -12, 48, 0.6)]   # motor L/R, visual, SMA
def glass_brain(figure, annotate, stat=None):
    """empty glass brain; the activation is added as an overlay so both renders share exactly the same layout"""
    display = plotting.plot_glass_brain(None, display_mode="lzr", figure=figure, annotate=annotate)
    if stat is not None:
        display.add_overlay(stat, cmap="magma_r", vmin=0, vmax=1.0, threshold=0.08)
    return display


GLASS, ACTIVE = crop(render(glass_brain), render(glass_brain, stat=blobs(PEAKS)))
ACTIVATION = overlay(GLASS, ACTIVE)

NODES = {"PCC": (0, -52, 26), "mPFC": (0, 52, -6), "L-AG": (-46, -66, 30), "R-AG": (48, -62, 30),
         "L-M1": (-38, -22, 56), "R-M1": (40, -20, 54), "V1": (2, -88, 4), "SMA": (2, -10, 52)}
rng = np.random.default_rng(3)
A = rng.uniform(0.2, 1.0, (8, 8)); A = (A + A.T) / 2; np.fill_diagonal(A, 0)
A[:4, :4] += 0.6; A[4:, 4:] += 0.4; np.fill_diagonal(A, 0)                 # two modules: default mode, sensorimotor/visual
COLORS = [TEAL] * 4 + [VIOLET] * 3 + [ORANGE]
conn = dict(node_coords=list(NODES.values()), node_color=COLORS, node_size=40, display_mode="lzr", colorbar=False,
            edge_cmap="Greys", edge_vmin=0, edge_vmax=2.0, edge_kwargs={"linewidth": 1.6})
CONN_NODES = render(plotting.plot_connectome, adjacency_matrix=np.zeros_like(A), **conn)
CONN_FULL = render(plotting.plot_connectome, adjacency_matrix=A, edge_threshold="70%", **conn)
CONN_NODES, CONN_FULL = crop(CONN_NODES, CONN_FULL)


# ---------------------------------------------------------------- scene
END = 10.0
scene = ck.Scene(duration=END, size=(9, 9), background="#ffffff", holds={4.6: 0.8, 9.6: 0.8}, title="cardkit + nilearn")

STEPS = [(0.0, "Task activation"), (5.0, "Functional connectivity")]
for i, (t0, title) in enumerate(STEPS):
    t1 = STEPS[i + 1][0] if i + 1 < len(STEPS) else END
    scene.add(Txt(f"{i + 1}", (0.5, 8.0), 22, TEAL, weight="bold", op=fade(t0, t1)))
    scene.add(Txt(title, (1.05, 8.0), 22, INK, op=fade(t0, t1)))


def place(arr, x0, x1, y_mid):
    """extent for an image of width x1 - x0 centred vertically on y_mid, keeping its aspect ratio"""
    h = (x1 - x0) * arr.shape[0] / arr.shape[1]
    return (x0, x1, y_mid - h / 2, y_mid + h / 2)


# step 1: the BOLD signal (bottom) drives the activation shown in the glass brain (top)
with scene.group("activation", fade(0.0, 5.0)):
    tt = np.linspace(0, 1, 160)
    bold = ck.seg(tt, 0.1, 0.35) * (1 - ck.seg(tt, 0.65, 0.9))                                # one block response, 0..1
    signal = ck.fn(lambda t: float(np.interp(ck.lin(t, 0.8, 4.2), tt, bold)))                    # current activation
    ext = place(GLASS, 0.5, 8.5, 5.25)
    scene.add(Img(GLASS, ext, op=ramp(0.1, 0.6), interp="antialiased", z=1))
    scene.add(Img(ACTIVATION, ext, op=signal, interp="antialiased", z=1.1))      # activation follows the signal

    X0, X1, Y0, H = 1.0, 8.0, 1.0, 1.6
    curve = np.c_[X0 + tt * (X1 - X0), Y0 + H * bold]
    scene.add(Path([(X0, Y0), (X1, Y0)], DIM, 1.2, op=ramp(0.4, 0.8)))
    scene.add(Path(curve, TEAL, 3, draw=ramp(0.8, 4.2, eased=False), z=2))
    scene.add(Circ(ck.along(curve, 0.8, 4.2, eased=False), 0.1, fill=TEAL, stroke="#ffffff", lw=1.5, op=ramp(0.7, 0.9), z=3))
    scene.add(Txt("BOLD signal", (X0, Y0 + H + 0.25), 11, MUTED, op=ramp(0.5, 1.0)))
    scene.add(Txt("time", (X1, Y0 - 0.35), 10.5, MUTED, "right", op=ramp(0.5, 1.0)))

# step 2: connectome (nilearn) + node strength (cardkit bars)
with scene.group("connectivity", fade(5.0, END)):
    ext = place(CONN_NODES, 0.5, 8.5, 5.4)
    scene.add(Img(CONN_NODES, ext, op=ramp(5.1, 5.6), interp="antialiased", z=1))
    scene.add(Img(CONN_FULL, ext, op=ramp(5.9, 6.8), interp="antialiased", z=1.1))
    strength = A.sum(1) / A.sum(1).max()
    order = np.argsort(-strength)
    for r, j in enumerate(order):
        x0, t0 = 1.0 + 0.92 * r, 6.6 + 0.12 * r
        flat, full = ck.rect(x0, 1.1, x0 + 0.62, 1.11), ck.rect(x0, 1.1, x0 + 0.62, 1.1 + 1.7 * strength[j])
        scene.add(Poly(ck.tween(flat, full, t0, t0 + 0.6), fill=COLORS[j]))
        scene.add(Txt(list(NODES)[j], (x0 + 0.31, 0.75), 9.5, MUTED, "center", op=ramp(t0, t0 + 0.4)))
    scene.add(Path([(0.85, 1.1), (8.3, 1.1)], DIM, 1.2, op=ramp(6.4, 6.8)))
    scene.add(Txt("node strength", (1.0, 3.15), 11, MUTED, op=ramp(6.5, 7.0)))

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out, exist_ok=True)
    if sys.argv[1:] == ["check"]:
        for tau in (2.5, 4.5, 8.0, 9.5):
            scene.check_text(scene.to_display(tau))
        sys.exit()
    scene.storyboard(os.path.join(out, "brain_storyboard.png"), n=8)
    scene.save_all(os.path.join(out, "brain"), "gif", "svg", poster=9.5)
