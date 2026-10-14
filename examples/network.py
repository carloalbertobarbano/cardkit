"""A neural network explainer: forward-pass activation waves through an MLP, then training, where every edge's width and
colour follow its weight while the loss curve draws itself.  Shows animated colours, line widths and radii on ~80 edges,
a softmax read-out, a counter and a loss plot.

    python network.py         -> out/network.gif, out/network_animated.svg, out/network_poster.png
    python network.py check   -> text-overlap check
"""
import os
import sys
import numpy as np
import cardkit as ck
from cardkit import Txt, Path, Poly, Circ, Img, ramp, fade

INK, MUTED, EDGE = "#1d2433", "#8a93a3", "#c9ced8"
TEAL, VIOLET, ORANGE = "#0f9d8a", "#6c5ce7", "#e17d38"

END = 11.0
FWD = (0.0, 5.5)                       # step 1: forward pass
TRAIN = (5.5, END)                     # step 2: training
scene = ck.Scene(duration=END, size=(9, 9), background="#ffffff", holds={5.15: 0.8, 10.6: 0.8}, title="cardkit network")

STEPS = [(FWD, "Forward pass", "activations flow from the input to the prediction"),
         (TRAIN, "Training", "each edge follows its weight as it is learned")]
for i, ((t0, t1), title, sub) in enumerate(STEPS):
    scene.add(Txt(f"{i + 1}", (0.5, 8.0), 22, TEAL, weight="bold", op=fade(t0, t1)))
    scene.add(Txt(title, (1.05, 8.0), 22, INK, op=fade(t0, t1)))
    scene.add(Txt(sub, (1.05, 7.45), 13, MUTED, op=fade(t0 + 0.3, t1)))

# ---------------------------------------------------------------- the network (base group: visible in both steps)
SIZES, XS, YC, DY = [4, 6, 6, 3], [2.2, 3.7, 5.2, 6.7], 4.75, 0.66
NODES = [[(x, YC + (k - (n - 1) / 2) * DY) for k in range(n)] for n, x in zip(SIZES, XS)]
rng = np.random.default_rng(7)
W0 = [rng.normal(0, 0.35, (a, b)) for a, b in zip(SIZES, SIZES[1:])]              # weights at initialisation
W1 = [w + rng.normal(0, 0.9, w.shape) * (rng.random(w.shape) < 0.55) for w in W0]  # ...after training
learn = ramp(TRAIN[0] + 0.5, TRAIN[1] - 1.2)                                      # 0 -> 1 during training
alive = ramp(0.1, 0.6) * (1 - ramp(END - 0.35, END))                              # the network fades in / out with the loop


def wave(t, layer):
    """forward-pass pulse at a (possibly fractional) layer index: peaks as the pulse passes, three passes in step 1"""
    if not FWD[0] + 0.8 <= t <= FWD[1] - 0.4:
        return 0.0
    phase = ((t - FWD[0] - 0.8) * 1.0) % 1.5
    return float(max(0.0, 1 - abs(phase - 0.32 * layer) / 0.28))


for li in range(3):
    for a, pa in enumerate(NODES[li]):
        for b, pb in enumerate(NODES[li + 1]):
            def weight(t, li=li, a=a, b=b):
                k = learn(t)
                return W0[li][a, b] * (1 - k) + W1[li][a, b] * k

            def color(t, li=li, weight=weight):
                if t < TRAIN[0]:
                    return ck.mix(EDGE, TEAL, wave(t, li + 0.5))
                return ck.mix(EDGE, TEAL if weight(t) >= 0 else ORANGE, min(1.0, abs(weight(t)) * 0.8 * ramp(*TRAIN)(t + 0.5)))

            def width(t, li=li, weight=weight):
                if t < TRAIN[0]:
                    return 0.8 + 1.6 * wave(t, li + 0.5)
                return 0.5 + 2.2 * min(1.0, abs(weight(t)))

            scene.add(Path([pa, pb], ck.fn(color), ck.fn(width), op=alive * 0.9, z=1))

for li, layer in enumerate(NODES):
    for k, (x, y) in enumerate(layer):
        glow = ck.fn(lambda t, li=li: wave(t, li))
        scene.add(Circ((x, y), 0.17 + 0.06 * glow, fill=ck.fn(lambda t, g=glow: ck.mix("#ffffff", TEAL, 0.25 + 0.75 * g(t))),
                       stroke=ck.shade(TEAL, 0.7), lw=1.2, op=alive, z=3))
        scene.add(Circ((x - 0.05, y + 0.06), 0.045, fill="#ffffff", op=alive * 0.8, z=4))

# ---------------------------------------------------------------- step 1: input image, softmax read-out
with scene.group("forward", fade(*FWD)):
    yy, xx = np.mgrid[0:8, 0:8]
    digit = ((np.abs(np.hypot(xx - 3.5, yy - 3.5) - 2.4) < 0.9) & (xx < 6)).astype(float)       # a pixelated "C"
    rgba = np.dstack([np.ones((8, 8, 3)) * (1 - 0.85 * digit[..., None]) * [0.9, 0.92, 0.95] + digit[..., None] * np.array([0.06, 0.6, 0.54]),
                      np.ones((8, 8))])
    scene.add(Img(rgba, (0.3, 1.5, YC - 0.6, YC + 0.6), interp="nearest", op=ramp(0.3, 0.8)))
    scene.add(Txt("input", (0.9, YC - 0.95), 11, MUTED, "center", op=ramp(0.4, 0.9)))
    for (x, y) in NODES[0]:
        scene.add(Path([(1.6, YC), (x - 0.2, y)], EDGE, 1.0, op=ramp(0.6, 1.0), z=0.5))

    probs = [(0.62, "cat"), (0.27, "dog"), (0.11, "bird")]
    for k, ((x, y), (p, name)) in enumerate(zip(NODES[3][::-1], probs)):
        t0 = 1.6 + 0.15 * k
        flat, full = ck.rect(x + 0.35, y - 0.11, x + 0.36, y + 0.11), ck.rect(x + 0.35, y - 0.11, x + 0.35 + 1.5 * p, y + 0.11)
        scene.add(Poly(ck.tween(flat, full, t0, t0 + 0.7), fill=TEAL if k == 0 else EDGE))
        scene.add(Txt(f"{name}  {p:.2f}", (x + 0.35, y + 0.2), 10, INK if k == 0 else MUTED, op=ramp(t0 + 0.3, t0 + 0.7)))
    scene.add(Txt("hidden layers", (4.45, YC - 2.4), 11, MUTED, "center", op=ramp(0.8, 1.3)))

# ---------------------------------------------------------------- step 2: loss curve, epoch counter
with scene.group("training", fade(*TRAIN)):
    X0, X1, Y0, Y1 = 1.0, 5.6, 0.8, 2.3
    ep = np.linspace(0, 1, 120)
    loss = 0.12 + 0.88 * np.exp(-4.5 * ep) + 0.03 * np.sin(40 * ep) * np.exp(-2 * ep)
    curve = np.c_[X0 + ep * (X1 - X0), Y0 + (Y1 - Y0) * loss]
    t_draw = (TRAIN[0] + 0.5, TRAIN[1] - 1.2)
    scene.add(Path([(X0, Y1 + 0.1), (X0, Y0), (X1 + 0.1, Y0)], MUTED, 1.1, op=ramp(TRAIN[0] + 0.2, TRAIN[0] + 0.6)))
    scene.add(Path(curve, ORANGE, 2.6, draw=ramp(*t_draw, eased=False), z=2))
    scene.add(Circ(ck.along(curve, *t_draw, eased=False), 0.08, fill=ORANGE, z=3, op=ramp(TRAIN[0] + 0.4, TRAIN[0] + 0.6)))
    scene.add(Txt("loss", (X0, Y1 + 0.25), 11, MUTED, op=ramp(TRAIN[0] + 0.3, TRAIN[0] + 0.7)))
    scene.add(Txt("epochs", (X1 + 0.1, Y0 - 0.32), 10.5, MUTED, "right", op=ramp(TRAIN[0] + 0.3, TRAIN[0] + 0.7)))
    epoch = lambda t: f"epoch {int(round(50 * ck.lin(t, *t_draw)))}"
    scene.add(Txt(epoch, (6.4, 1.75), 18, INK, weight="bold", op=ramp(TRAIN[0] + 0.4, TRAIN[0] + 0.8)))
    acc = lambda t: f"accuracy {int(round(100 * (0.34 + 0.6 * (1 - np.exp(-4.5 * ck.lin(t, *t_draw))))))}%"
    scene.add(Txt(acc, (6.4, 1.2), 12, TEAL, op=ramp(TRAIN[0] + 0.5, TRAIN[0] + 0.9)))
    for k, (c, name) in enumerate([(TEAL, "positive weight"), (ORANGE, "negative weight")]):
        scene.add(Path([(6.4 + 1.25 * k, 6.75), (6.7 + 1.25 * k, 6.75)], c, 2.6, op=ramp(TRAIN[0] + 0.6, TRAIN[0] + 1.0)))
    scene.add(Txt("+ / − weight", (7.65, 6.38), 10, MUTED, "center", op=ramp(TRAIN[0] + 0.6, TRAIN[0] + 1.0)))

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out, exist_ok=True)
    if sys.argv[1:] == ["check"]:
        for tau in (2.5, 5.0, 8.0, 10.5):
            scene.check_text(scene.to_display(tau))
        sys.exit()
    scene.storyboard(os.path.join(out, "network_storyboard.png"))
    scene.save_all(os.path.join(out, "network"), "gif", "svg", poster=10.4)
