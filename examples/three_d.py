"""3-D with cardkit.three: a connectivity matrix as a lit slab that sends shaded tubes to rows of label cubes
(as in the BrainPFN card), then a 3-D bar chart that grows from the back to the front.

Everything is plain Poly faces from a fixed oblique camera, so the animated SVG stays fully vector.

    python three_d.py         -> out/three_d.gif, out/three_d_animated.svg, out/three_d_poster.png
    python three_d.py check   -> text-overlap check
"""
import os
import sys
import numpy as np
import cardkit as ck
from cardkit import Txt, Circ, ramp, fade
from cardkit.three import Camera, box, slab, tube

INK, MUTED = "#1d2433", "#8a93a3"
TEAL, VIOLET, ORANGE = "#0f9d8a", "#6c5ce7", "#e17d38"
HUES = [TEAL, VIOLET, ORANGE]

END = 11.0
scene = ck.Scene(duration=END, size=(9, 9), background="#ffffff", holds={5.15: 0.8, 10.6: 0.8}, title="cardkit 3-D")

STEPS = [(0.0, "Matrices with depth", "a lit slab, tubes that grow, cubes that recolour"),
         (5.5, "3-D bar charts", "boxes whose height is a function of time")]
for i, (t0, title, sub) in enumerate(STEPS):
    t1 = STEPS[i + 1][0] if i + 1 < len(STEPS) else END
    scene.add(Txt(f"{i + 1}", (0.5, 8.0), 22, TEAL, weight="bold", op=fade(t0, t1)))
    scene.add(Txt(title, (1.05, 8.0), 22, INK, op=fade(t0, t1)))
    scene.add(Txt(sub, (1.05, 7.45), 13, MUTED, op=fade(t0 + 0.3, t1)))


def corr_color(v):
    return ck.mix("#ffffff", TEAL if v >= 0 else VIOLET, float(min(1.0, abs(v) * 1.15)))


# ---------------------------------------------------------------- step 1: slab -> tubes -> label cubes
rng = np.random.default_rng(31)
lat = rng.normal(size=(3, 400)); W = rng.uniform(-0.2, 1, (8, 3)) * (rng.random((8, 3)) < 0.6)
FC = np.corrcoef(W @ lat + 0.9 * rng.normal(size=(8, 400)))
LABELS = rng.uniform(0.15, 1.0, (3, 12, 6))                    # 12 label patterns per row, cycled over time

with scene.group("slab", fade(0.0, 5.5)):
    cam = Camera(origin=(0.45, 3.0))
    scene.add(slab(cam, (0, 0, 0), FC, 0.32, corr_color, z=3, reveal=lambda k: ramp(0.3 + 0.012 * k, 0.6 + 0.012 * k)))
    scene.add(Txt("matrix  $x$", (1.9, 2.55), 12.5, MUTED, "center", op=ramp(0.9, 1.4)))
    start = np.array([2.56, 1.28, 0.06])                        # right edge of the slab, half-way in depth
    for r, row_y in enumerate([5.55, 4.05, 2.55]):
        D = (row_y - cam.origin[1]) / cam.dy                    # depth that puts this row at screen height row_y
        X = 6.1 - cam.origin[0] - cam.dx * D - 0.17
        end = np.array([X - 0.62, D, 0.17])
        path = ck.bezier(start, start + (0.25 * (end - start)[0], 0.3 * (end - start)[1], 0.45),
                         start + (0.7 * (end - start)[0], 0.8 * (end - start)[1], 0.5), end, 48)
        t_arrow = 1.2 + 0.6 * r
        scene.add(tube(cam, path, 0.04, HUES[r], t0=t_arrow, dur=1.0, z=6 + 0.5 * r))
        mid = cam.project(path[36])[0]
        scene.add(Txt(f"$\\varphi_{r + 1}$", (mid[0] - 0.22, mid[1] + 0.36), 17, HUES[r], "center", z=20,
                      op=ramp(t_arrow + 0.5, t_arrow + 1.0)))
        pattern = ck.every(3.4, 0.6, 12)                        # a new label pattern every 0.6 s
        for k in range(6):
            t_cube = t_arrow + 1.0 + 0.09 * k
            col = ck.fn(lambda t, r=r, k=k: ck.mix("#ffffff", HUES[r], 0.22 + 0.78 * LABELS[r, pattern(t), k]))
            corner = (X + 0.46 * k - 0.17, D - 0.17, 0.0)
            scene.add(box(cam, corner, (0.34, 0.34, 0.31), col, op=ramp(t_cube, t_cube + 0.25), z=4 + cam.depth(corner) * 0.01))
    scene.add(Txt("labels  $y$", (5.85, 6.6), 12.5, MUTED, op=ramp(3.0, 3.5)))
    for q in range(3):
        scene.add(Circ((7.3 + 0.2 * q, 1.9), 0.04, fill=MUTED, op=ramp(3.6, 4.0)))
    scene.add(Txt("each tube $\\varphi_i$ maps the matrix to its own row of labels", (4.5, 1.1), 12.5, MUTED, "center",
                  op=ramp(4.0, 4.5)))

# ---------------------------------------------------------------- step 2: 3-D bar chart
with scene.group("bars", fade(5.5, END)):
    cam2 = Camera(origin=(1.0, 1.55))
    n, pitch, w = 5, 0.95, 0.62
    gx, gd = np.meshgrid(np.arange(n), np.arange(n))
    H = 0.4 + 2.6 * np.exp(-((gx - 2.6) ** 2 + (gd - 1.6) ** 2) / 3.5)          # a smooth bump of heights
    scene.add(box(cam2, (-0.2, -0.2, -0.12), (n * pitch + 0.1, n * pitch + 0.1, 0.12), "#e8ebf0", z=1,
                  op=ramp(5.7, 6.1)))
    for i in range(n):                         # i: depth row (0 = front), j: column
        for j in range(n):
            corner = (j * pitch, i * pitch, 0.0)
            t0 = 6.0 + 0.12 * (n - 1 - i) + 0.05 * j                            # back rows grow first
            height = ramp(t0, t0 + 1.0) * (H[i, j] - 0.02) + 0.02
            color = ck.mix("#c9f0ea", TEAL, (H[i, j] - 0.4) / 2.6)
            scene.add(box(cam2, corner, (w, w, height), color, z=10 + cam2.depth(np.add(corner, (w / 2, w / 2, 0)))))
    scene.add(Txt("height = a function of t", (8.5, 1.0), 11, MUTED, "right", op=ramp(7.0, 7.5)))

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out, exist_ok=True)
    if sys.argv[1:] == ["check"]:
        for tau in (3.0, 5.0, 8.5, 10.5):
            scene.check_text(scene.to_display(tau))
        sys.exit()
    scene.storyboard(os.path.join(out, "three_d_storyboard.png"))
    scene.save_all(os.path.join(out, "three_d"), "gif", "svg", poster=4.9)
