"""A tour of cardkit: an intro, two steps that fade in and out as groups, reading pauses, motion along a path,
colour keyframes, a counter, mathtext, an image and a progress indicator.

    python tour.py            -> out/tour.gif, out/tour_animated.svg, out/tour_poster.png, out/tour_storyboard.png
    python tour.py check      -> text-overlap check at a few instants
"""
import os
import sys
import numpy as np
import matplotlib
import cardkit as ck
from cardkit import Txt, Path, Poly, Circ, Img, ramp, fade

INK, MUTED, DIM = "#1d2433", "#8a93a3", "#e3e7ee"
TEAL, VIOLET, ORANGE = "#0f9d8a", "#6c5ce7", "#e17d38"

STEPS = [(0.0, "Anything can move"), (4.0, "Data and maths")]          # (start, title) in scene time
END = 8.0

scene = ck.Scene(duration=END, size=(9, 9), background="#ffffff",
                 intro=1.5,                       # scene time starts at -1.5: the intro is authored at negative times
                 holds={3.6: 0.8, 7.6: 0.8},      # pause 0.8 s just before each step fades out (reading time)
                 title="cardkit tour")

# ---------------------------------------------------------------- intro (negative scene time), base group
hero = 1 - ramp(-0.5, -0.1)
scene.add(Txt("cardkit", (4.5, 4.7), 40, INK, "center", weight="bold", op=hero))
scene.add(Txt("a tour in two steps", (4.5, 4.0), 15, MUTED, "center", op=hero))

# ---------------------------------------------------------------- step titles + progress (base group)
for i, (t0, title) in enumerate(STEPS):
    t1 = STEPS[i + 1][0] if i + 1 < len(STEPS) else END
    scene.add(Txt(f"{i + 1}", (0.5, 8.0), 22, TEAL, weight="bold", op=fade(t0, t1)))
    scene.add(Txt(title, (1.05, 8.0), 22, INK, op=fade(t0, t1)))

step = ck.every(0.0, 4.0)                                            # 0 during step 1, 1 during step 2
for i, x in enumerate([4.2, 4.8]):
    scene.add(Circ((x, 0.55), r=ck.fn(lambda t, i=i: 0.12 if step(t) == i else 0.08),
                   fill=ck.fn(lambda t, i=i: TEAL if step(t) >= i else DIM), op=ramp(-0.3, 0.1)))

# ---------------------------------------------------------------- step 1: motion
with scene.group("s1", fade(0.0, 4.0)):
    track = ck.bezier((1.0, 2.0), (2.5, 7.2), (6.0, -1.0), (8.0, 5.5), 80)
    scene.add(Path(track, DIM, 2.0, dash=(0.08, 0.08)))
    scene.add(Path(track, TEAL, 3.0, draw=ramp(0.6, 2.8, eased=True), z=1))
    dot = ck.along(track, 0.6, 2.8)
    scene.add(Circ(dot, 0.16, fill=TEAL, stroke="#ffffff", lw=2, z=2, op=ramp(0.4, 0.6)))
    scene.add(Txt("follow me", dot + np.array([-0.25, 0.2]), 12, MUTED, "right", op=ramp(0.4, 0.6), z=2))   # text moves too

    # a square that changes colour and grows
    col = ck.keyframes({1.0: TEAL, 2.0: VIOLET, 3.0: ORANGE})
    side = ck.keyframes({1.0: 0.5, 2.0: 0.9, 3.0: 0.6})
    scene.add(Poly(ck.fn(lambda t: ck.rect(6.6, 1.2, 6.6 + side(t), 1.2 + side(t))), fill=col, op=ramp(0.8, 1.2)))
    scene.add(Txt("keyframed colour", (6.6, 0.95), 10.5, MUTED, op=ramp(0.8, 1.2)))

# ---------------------------------------------------------------- step 2: data, maths, images
with scene.group("s2", fade(4.0, END)):
    # an image: any RGB(A) array, here a colormapped field
    yy, xx = np.mgrid[-1:1:120j, -1:1:120j]
    field = matplotlib.colormaps["magma"]((np.sin(5 * xx) * np.cos(4 * yy) + 1) / 2)
    scene.add(Img(field, (0.6, 3.6, 2.4, 5.4), op=ramp(4.3, 4.9)))
    scene.add(Txt("Img: any array", (2.1, 2.05), 11, MUTED, "center", op=ramp(4.5, 5.0)))

    # mathtext is converted to outlines like any text
    scene.add(Txt("$\\hat{y} = \\sum_i \\, w_i \\, \\varphi_i(x)$", (6.2, 6.4), 20, INK, "center", op=ramp(4.6, 5.2)))

    # a counter: the string itself is a function of time
    count = lambda t: f"{int(10 ** (6 * ck.seg(t, 5.2, 7.0))):,}"
    scene.add(Txt(count, (6.2, 4.3), 30, TEAL, "center", weight="bold", op=ramp(5.0, 5.3)))
    scene.add(Txt("samples seen", (6.2, 3.75), 12, MUTED, "center", op=ramp(5.1, 5.5)))

    # bars that grow
    for k, v in enumerate([0.55, 0.8, 0.35, 0.95]):
        x0 = 4.6 + 0.8 * k
        flat, full = ck.rect(x0, 1.4, x0 + 0.55, 1.41), ck.rect(x0, 1.4, x0 + 0.55, 1.4 + 1.6 * v)
        scene.add(Poly(ck.tween(flat, full, 5.4 + 0.15 * k, 6.2 + 0.15 * k), fill=[TEAL, VIOLET, ORANGE, TEAL][k]))
    scene.add(Path([(4.4, 1.4), (7.9, 1.4)], MUTED, 1.2, op=ramp(5.2, 5.6)))

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out, exist_ok=True)
    if sys.argv[1:] == ["check"]:
        for tau in (-1.0, 2.0, 3.5, 6.5, 7.5):
            scene.check_text(scene.to_display(tau))
        sys.exit()
    print(scene)
    scene.storyboard(os.path.join(out, "tour_storyboard.png"))
    scene.save_all(os.path.join(out, "tour"), "gif", "svg", poster=7.5)
