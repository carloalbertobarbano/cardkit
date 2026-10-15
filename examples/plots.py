"""Animated scientific plots with cardkit.plot.Axes: a regression whose fitted curve morphs between models while the
residuals follow it, and two learning curves that grow together along x, with a moving tip and a shaded regime.

    python plots.py           -> out/plots.gif, out/plots_animated.svg, out/plots_poster.png
    python plots.py check     -> text-overlap check
"""
import os
import sys
import numpy as np
import cardkit as ck
from cardkit import Txt, Circ, ramp

INK, MUTED = "#1d2433", "#8a93a3"
TEAL, VIOLET = "#0f9d8a", "#6c5ce7"

scene = ck.Scene(duration=8, size=(12, 5), background="#ffffff", title="cardkit plots")
scene.add(Txt("Plots that move", (0.5, 4.3), 20, INK, weight="bold"))

# ---------------------------------------------------------------- left: a fit that morphs between three models
x = np.linspace(0, 1, 120)
models = [1.6 * (x - 0.5), np.sin(2 * np.pi * x) * 0.9, np.cos(3 * np.pi * x) * 0.8]
xp = np.sort(np.random.default_rng(5).uniform(0.03, 0.97, 18))
noise = np.random.default_rng(6).normal(0, 0.35, xp.size)
which = ck.keyframes({0.5: 0.0, 2.0: 0.0, 2.8: 1.0, 4.6: 1.0, 5.4: 2.0})       # which model, as a continuous index


def fit(t):
    m = which(t)
    i = min(int(m), 1)
    return models[i] * (1 - (m - i)) + models[i + 1] * (m - i)


at_points = lambda t: np.interp(xp, x, fit(t))
ax = scene.axes((0.9, 1.0, 4.6, 2.7), xlim=(-0.02, 1.02), ylim=(-2, 2), xlabel="feature", ylabel="target",
                op=ramp(0.0, 0.4))
ax.stems(xp, at_points, lambda t: at_points(t) + noise, VIOLET, 0.7, op=0.45)
ax.scatter(xp, lambda t: at_points(t) + noise, r=0.05, fill=VIOLET, z=3)
ax.line(x, fit, TEAL, 2.4, z=2)
ax.text(0.03, 0.9, r"$y = f(x) + \epsilon$", 10, INK, coords="axes")

# ---------------------------------------------------------------- right: learning curves revealed along x
n = np.logspace(np.log10(5), 3, 90)
ln = np.log10(n)
ours, base = 0.02 + 0.33 * (1 - np.exp(-n / 30)), 0.37 * (1 - np.exp(-n / 160))
ax2 = scene.axes((7.1, 1.0, 4.4, 2.7), xlim=(np.log10(5) - 0.02, 3.02), ylim=(0, 0.45), xlabel="training examples",
                 ylabel="score", xticks=np.log10([10, 100, 1000]), xticklabels=["10", "100", "1000"], op=ramp(0.0, 0.4))
ax2.vspan(np.log10(5) - 0.02, 2, TEAL, op=0.07, z=0)
ax2.vline(2, TEAL, 0.8, dash=(0.05, 0.05), op=0.5)
grow = ramp(0.8, 6.5)
for y, col, lw in [(base, MUTED, 2.0), (ours, TEAL, 2.6)]:
    ax2.line(ln, y, col, lw, reveal=grow)                 # both curves advance together along x
    scene.add(Circ(ax2.tip(ln, y, grow), 0.06, fill=col, z=5, op=ck.fn(lambda t: 1.0 if 0.8 < t < 6.5 else 0.0)))
ax2.text(np.log10(5) + 0.04, 0.41, "small-data regime", 8.5, TEAL)
for k, (col, lab) in enumerate([(TEAL, "ours"), (MUTED, "baseline")]):
    scene.add(ck.Path([(9.2 + 1.0 * k, 4.0), (9.45 + 1.0 * k, 4.0)], col, 2.2), Txt(lab, (9.52 + 1.0 * k, 3.95), 9, MUTED))

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out, exist_ok=True)
    if sys.argv[1:] == ["check"]:
        for t in (1.0, 4.0, 7.5):
            scene.check_text(t)
        sys.exit()
    scene.save_all(os.path.join(out, "plots"), "gif", "svg", poster=7.5)
