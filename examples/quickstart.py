"""The README quick start: a 5-second card with a title, a line that draws itself and a bar chart that grows."""
import os
import numpy as np
import cardkit as ck
from cardkit import Txt, Path, Poly, Circ, ramp

INK, ACCENT, MUTED = "#1d2433", "#0f9d8a", "#8a93a3"

scene = ck.Scene(duration=5, size=(9, 9), background="#ffffff")

# title: fades in between t = 0.2 s and t = 0.8 s
scene.add(Txt("Hello, cardkit", (0.6, 7.9), size=26, color=INK, weight="bold", op=ramp(0.2, 0.8)))
scene.add(Txt("one scene $\\rightarrow$ GIF, MP4 and animated SVG", (0.6, 7.3), size=14, color=MUTED, op=ramp(0.5, 1.1)))

# a sine wave that draws itself from left to right
x = np.linspace(0.8, 8.2, 120)
scene.add(Path(np.c_[x, 5.2 + 0.7 * np.sin(2.2 * x)], color=ACCENT, lw=3, draw=ramp(1.0, 2.6)))

# five bars that grow one after the other: the shape morphs from a flat to a full rectangle
for k, h in enumerate([1.2, 2.0, 1.6, 2.6, 3.1]):
    x0, t0 = 1.0 + 1.5 * k, 2.4 + 0.2 * k
    flat, full = ck.rect(x0, 0.9, x0 + 1.0, 0.91), ck.rect(x0, 0.9, x0 + 1.0, 0.9 + h)
    scene.add(Poly(ck.tween(flat, full, t0, t0 + 0.8), fill=ACCENT, op=0.35 + 0.13 * k))

# a dot that pulses forever
scene.add(Circ((8.2, 7.75), r=ck.pulse(1.2, 0.10, 0.18), fill=ACCENT))

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    scene.save_all(os.path.join(out, "quickstart"), "gif", "svg", poster=4.5)
