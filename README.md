# cardkit

**Write an animation once in Python. Get a GIF, a WebP, an MP4 and an animated SVG, and they all look the same.**

cardkit is a small scene graph built on matplotlib. You describe a card as a list of shapes, lines, text and images.
Any property of any element can be a function of time. cardkit then renders the scene in two ways:

* **raster**: matplotlib draws each frame, which becomes a GIF, an animated WebP, an MP4 or a PNG poster;
* **vector**: a native SVG whose animation uses SMIL `<animate>` tags, built by sampling the same functions.
  It loops forever, needs no JavaScript and no fonts, and plays inside a plain `<img>` tag (GitHub READMEs, websites, slides).

All text is converted to outlines, mathtext included (`"$\hat{y} = \sum_i w_i \varphi_i(x)$"`), so the SVG matches the
frames pixel for pixel on any machine.

<p align="center">
  <img src="https://raw.githubusercontent.com/carloalbertobarbano/cardkit/main/docs/tour_animated.svg" width="48%" alt="cardkit tour (animated SVG)">
  <img src="https://raw.githubusercontent.com/carloalbertobarbano/cardkit/main/docs/tour.gif" width="48%" alt="cardkit tour (GIF)">
</p>
<p align="center"><sub>The same scene (<a href="https://github.com/carloalbertobarbano/cardkit/blob/main/examples/tour.py">examples/tour.py</a>): animated SVG on the left (101 kB), GIF on the right.</sub></p>

It was built for scientific graphical abstracts, website cards and social-media videos, where you want real plots,
real numbers and typeset maths, plus an SVG that stays sharp.

---

## Install

```bash
pip install cardkit-anim             # numpy, matplotlib and pillow are the only dependencies
pip install "cardkit-anim[video]"    # + imageio-ffmpeg, for MP4 export if ffmpeg is not installed
```

The package on PyPI is `cardkit-anim`; the module you import is `cardkit` (`import cardkit as ck`).

## Quick start

```python
import numpy as np
import cardkit as ck
from cardkit import Txt, Path, Poly, Circ, ramp

INK, ACCENT, MUTED = "#1d2433", "#0f9d8a", "#8a93a3"

scene = ck.Scene(duration=5, size=(9, 9), background="#ffffff")

# a title that fades in between t = 0.2 s and t = 0.8 s
scene.add(Txt("Hello, cardkit", (0.6, 7.9), size=26, color=INK, weight="bold", op=ramp(0.2, 0.8)))

# a sine wave that draws itself from left to right
x = np.linspace(0.8, 8.2, 120)
scene.add(Path(np.c_[x, 5.2 + 0.7 * np.sin(2.2 * x)], color=ACCENT, lw=3, draw=ramp(1.0, 2.6)))

# bars that grow one after the other: each shape morphs from a flat to a full rectangle
for k, h in enumerate([1.2, 2.0, 1.6, 2.6, 3.1]):
    x0, t0 = 1.0 + 1.5 * k, 2.4 + 0.2 * k
    flat, full = ck.rect(x0, 0.9, x0 + 1.0, 0.91), ck.rect(x0, 0.9, x0 + 1.0, 0.9 + h)
    scene.add(Poly(ck.tween(flat, full, t0, t0 + 0.8), fill=ACCENT, op=0.35 + 0.13 * k))

# a dot that pulses forever
scene.add(Circ((8.2, 7.75), r=ck.pulse(1.2, 0.10, 0.18), fill=ACCENT))

scene.save_all("out/quickstart", "gif", "svg", poster=4.5)
# -> out/quickstart.gif, out/quickstart_animated.svg, out/quickstart_poster.png
```

<p align="center"><img src="https://raw.githubusercontent.com/carloalbertobarbano/cardkit/main/docs/quickstart_animated.svg" width="300" alt="quick start result"></p>

To preview while you work, run `scene.frame(2.0)`, which returns a PIL image (it shows inline in Jupyter), or
`scene.storyboard("board.png")` for a contact sheet of key frames. By default it shows one finished frame per group:
the last instant the group is fully visible, when everything in it has appeared and nothing has started to fade. You
can also pick the frames yourself, in scene time: `scene.storyboard("board.png", keys=[("hook", -1.3), ("results", 23.6)])`.

<p align="center"><img src="https://raw.githubusercontent.com/carloalbertobarbano/cardkit/main/docs/tour_storyboard.png" width="640" alt="storyboard"></p>

## Gallery

<p align="center">
  <img src="https://raw.githubusercontent.com/carloalbertobarbano/cardkit/main/docs/network_animated.svg" width="32%" alt="neural network example">
  <img src="https://raw.githubusercontent.com/carloalbertobarbano/cardkit/main/docs/three_d_animated.svg" width="32%" alt="3-D example">
  <img src="https://raw.githubusercontent.com/carloalbertobarbano/cardkit/main/docs/brain_animated.svg" width="32%" alt="nilearn example">
</p>

| example | shows |
|---|---|
| [`network.py`](https://github.com/carloalbertobarbano/cardkit/blob/main/examples/network.py) | an MLP: forward-pass activation waves, then training, with about 80 edges whose width and colour follow their weights; softmax bars, a loss curve, counters |
| [`three_d.py`](https://github.com/carloalbertobarbano/cardkit/blob/main/examples/three_d.py) | `cardkit.three`: a connectivity matrix as a lit 3-D slab sending shaded tubes to recolouring label cubes, then a growing 3-D bar chart |
| [`brain.py`](https://github.com/carloalbertobarbano/cardkit/blob/main/examples/brain.py) | nilearn glass brains whose activation follows a BOLD curve as it draws, then a connectome with a node-strength bar chart |

Every image above is the animated SVG itself, with no GIF involved. Run `python docs/make_docs.py` to rebuild all the demos.

---

## The model in five ideas

### 1. Coordinates are slide units

A scene is `size=(width, height)` units, with **y pointing up** and `(0, 0)` at the bottom-left corner. The default card
is 9 × 9. Font sizes and line widths are in **points**, and 1 unit = 43.2 pt, so a 9-unit card is 388.8 pt
(5.4 in) wide. This keeps type the same size relative to the card at any output resolution. `width_px` (default
120 px per unit, so 1080 px for 9 units) only sets the resolution of the raster outputs.

For other aspect ratios, change `size` (and optionally `origin`, the slide coordinates of the bottom-left corner):

```python
ck.Scene(25, size=(9, 11.25), origin=(0, -0.85))   # 4:5 portrait: the 9x9 card stays at 0..9, extra room below and above
```

### 2. Anything can be a function of time

Every attribute takes either a constant or a function `t -> value`: positions, point lists, colours, radii, opacity,
line widths, the fraction of a line that is drawn, even the text string.

```python
Circ(c=lambda t: (1 + t, 4.5), r=0.2, fill="#e17d38")             # moves right at 1 unit / s
Txt(lambda t: f"{int(100 * t)} %", (4.5, 4.5), 30, ha="center")   # a counter
```

<p align="center"><img src="https://raw.githubusercontent.com/carloalbertobarbano/cardkit/main/docs/network_animated.svg" width="360" alt="network example"></p>
<p align="center"><sub><a href="https://github.com/carloalbertobarbano/cardkit/blob/main/examples/network.py">network.py</a>: about 80 edges whose colour and width are functions of time (activation waves, then the weights as they are learned), plus counters whose text changes.</sub></p>

The helpers in [Timing helpers](#timing-helpers) cover the common cases without lambdas.

### 3. Five primitives

| primitive | what | main arguments |
|---|---|---|
| `Poly(pts, fill=None, stroke=None, lw=0)` | closed polygon | `pts`: (N, 2) vertices |
| `Path(pts, color, lw=1.5, draw=1.0, dash=None, cap="round")` | open polyline | `draw`: fraction drawn, 0..1; `dash`: (on, off) in units |
| `Circ(c, r, fill=None, stroke=None, lw=0)` | circle | centre and radius in units |
| `Txt(s, xy, size=12, color="#000", ha="left", weight=, style=, family=)` | text as outlines | `xy` on the baseline; `ha`: left / center / right |
| `Img(arr, extent, interp="bilinear")` | raster image | `arr`: (H, W, 3 or 4) array; `extent`: (x0, x1, y0, y1) |

They all also take `op` (opacity, 0..1), `z` (drawing order: higher is on top) and `group` (see below). If you prefer
longer names, `ck.Polygon`, `ck.Line`, `ck.Circle`, `ck.Text` and `ck.Image` are the same classes.

### 4. Groups fade as a whole

Put the elements of one step of your story in a **group** with an opacity **envelope**. The group fades as a single
layer, so overlapping translucent shapes never show seams, and the SVG only samples a group while it is visible, which
keeps files small.

```python
with scene.group("step1", ck.fade(0.0, 6.4)):     # visible from 0 to 6.4 s, fading in and out
    scene.add(Txt("Step one", (0.5, 8), 22))
    scene.add(Path(curve, "#0f9d8a", 3, draw=ramp(0.5, 2.0)))

with scene.group("step2", (6.4, 12.0)):           # a (t0, t1) tuple is shorthand for fade(t0, t1)
    ...
```

Elements added outside any group are in the base group, which is always visible.

### 5. Author in scene time, then add an intro and pauses

You time every element in **scene time**. The scene can be played back differently from how you wrote it:

```python
scene = ck.Scene(
    duration=25,                    # the story runs from scene time 0 to 25
    intro=3.0,                      # ...plus an intro authored at negative times, -3 .. 0
    holds={6.05: 0.9, 12.0: 0.9},   # freeze at these scene times for 0.9 s (reading pauses)
)
scene.duration                      # 29.8: the actual loop length (display time)
scene.to_display(6.4)               # scene time -> display time
```

<p align="center"><img src="https://raw.githubusercontent.com/carloalbertobarbano/cardkit/main/docs/tour_animated.svg" width="360" alt="tour example"></p>
<p align="center"><sub><a href="https://github.com/carloalbertobarbano/cardkit/blob/main/examples/tour.py">tour.py</a>: an intro authored at negative times, two groups, and a 0.8 s reading pause before each fade-out.</sub></p>

Adding an intro or a pause never forces you to re-time the rest. `frame`, `check_text` and `save(..., at=)` take display time;
`frame_at(tau)`, `poster=` and the storyboard `keys` take scene time.

---

## Timing helpers

They all return `Fn` objects: functions of time that support arithmetic, such as `ramp(1, 2) * 0.5` or
`1 - ramp(3, 3.5)`, or `fade(0, 6) * pulse(2, 0.6, 1)`.

| helper | gives |
|---|---|
| `ramp(t0, t1)` | 0 → 1 between t0 and t1, smoothstep-eased (`eased=False` for linear) |
| `fade(t0, t1=None, d=0.35)` | fades in after t0 and out before t1, over `d` s (or `(d_in, d_out)`) |
| `tween(a, b, t0, t1)` | interpolates numbers, arrays (point lists, so shapes morph) or colours |
| `keyframes({t: value, ...})` | piecewise interpolation through several keys (numbers, arrays, colours) |
| `along(pts, t0, t1)` | a position travelling along a polyline: `Circ(along(path, 1, 3), 0.1)` |
| `pulse(period, lo, hi)` | smooth periodic oscillation |
| `every(t0, period, n=None)` | integer step counter (0, 1, 2, ...), handy to cycle through examples |
| `fn(callable)` | wraps your own `t -> value` function to get the arithmetic |
| `seg(t, a, b)`, `ease(x)`, `lin(t, a, b)` | the raw scalar functions |

> **The lambda-in-a-loop trap.** `for k in range(5): Circ(..., op=lambda t: seg(t, k, k + 1))` gives every circle
> the *last* `k`. Use the helpers (`op=ramp(k, k + 1)`), which capture values correctly, or bind with `lambda t, k=k: ...`.

## Shapes and colours

`rect(x0, y0, x1, y1)`, `rounded_rect(x0, y0, x1, y1, r)`, `bezier(p0, c1, c2, p1, n)` (2-D or 3-D),
`arc(c, r, a0, a1)`, `ngon(c, r, k)` and `arrow_head(tip, direction)` return point arrays.
`mix(c1, c2, k)`, `shade(c, k)` and `lighten(c, k)` blend colours, and `colorize(c, fn)` applies one of them to a
colour that is itself a function of time. `text_width(s, size)` measures a label, for example to size a chip around it.

## Exporting

```python
scene.save("card.svg")                    # animated SVG (SMIL)
scene.save("card_still.svg", at=12.0)     # static SVG snapshot at display time 12 s
scene.save("poster.png", at=23.6)         # one frame
scene.save("card.gif")                    # renders every frame, then encodes
scene.save_all("out/card", "gif", "webp", "mp4", "svg", poster=23.6)   # renders the frames once for all formats
```

| format | notes |
|---|---|
| **SVG** | `save_svg(path, at=None, dt=0.1, simplify=True)`. Attributes are sampled every `dt` s; runs of identical samples are dropped (about half the size, same animation). Plays in `<img>`. |
| **GIF** | Pillow only, no ffmpeg. One shared 256-colour palette without dithering (crisp flat colours), `gif_fps=15`, `gif_width=720`. |
| **WebP** | Full resolution, `webp_quality=90`. |
| **MP4** | H.264 / yuv420p / faststart, `crf=16`: plays on LinkedIn, X, Slack and Keynote. Needs ffmpeg (`$FFMPEG`, `PATH`, or `pip install imageio-ffmpeg`). |
| **PNG** | `frame(t)` / `frame_at(tau)` return PIL images. |

Frames are rendered in parallel with all cores but one (`workers=1` renders serially). Use `fps=` (default 30) and
`width_px=` to change the frame rate and resolution.

## Quality checks

```python
scene.check_text(scene.to_display(6.0))
# [check t=7.80] 14 texts, 1 issues
#     OVERLAP 'FC matrix' <-> 'subjects per dataset'
```

`check_text(t, margin=0.25)` lists visible texts that overlap each other or come closer than `margin` units to the edge.
It is cheap: run it at a few key instants after every layout change.

## 3-D

`cardkit.three` builds 3-D-looking objects from plain `Poly` faces, seen through a fixed oblique camera with Lambert
lighting. Because the camera never moves, hidden faces are removed and faces are ordered once, when the scene is built.
The objects can still grow, move and change colour over time, and the SVG stays fully vector.

```python
from cardkit.three import Camera, box, slab, tube

cam = Camera(origin=(1.0, 2.0))                     # world axes: X right, D into the screen, Z up
scene.add(box(cam, (0, 0, 0), (0.6, 0.6, ramp(1, 2) * 2.5), "#0f9d8a"))       # a bar that grows to 2.5 units
scene.add(slab(cam, (0, 0, 0), matrix, 0.3, colors=lambda v: ck.mix("#fff", "#0f9d8a", v)))   # a matrix lying flat
scene.add(tube(cam, ck.bezier(p0, c1, c2, p1), 0.04, "#6c5ce7", t0=1.0, dur=1.0))            # a shaded arrow that grows
```

<p align="center"><img src="https://raw.githubusercontent.com/carloalbertobarbano/cardkit/main/docs/three_d_animated.svg" width="360" alt="three_d example"></p>
<p align="center"><sub><a href="https://github.com/carloalbertobarbano/cardkit/blob/main/examples/three_d.py">three_d.py</a>: a matrix slab, tubes that grow and cubes that recolour, then boxes that grow from the back to the front.</sub></p>

`Camera.project(P)` maps (N, 3) world points to the slide, `Camera.lit(color, normal)` shades a colour, and
`Camera.depth(P)` gives a value to use as `z` when separate objects overlap (closer objects get higher `z`).

## Using real figures

`Img` takes any array, so you can render a matplotlib or nilearn figure to an RGBA array once and place it in the card:

```python
fig = plt.figure(figsize=(6, 2.4), dpi=200)
plotting.plot_glass_brain(stat_img, display_mode="lzr", figure=fig, annotate=False)
fig.canvas.draw(); arr = np.asarray(fig.canvas.buffer_rgba()) / 255; plt.close(fig)
scene.add(Img(arr, (0.5, 8.5, 3.6, 6.8), op=ramp(0.3, 0.9)))
```

To animate *inside* a rendered figure, render it in two states and fade between them.
[`brain.py`](https://github.com/carloalbertobarbano/cardkit/blob/main/examples/brain.py) renders an empty glass brain and the same brain with the activation, then turns the
difference into an exact overlay whose opacity follows a signal. The outline never moves; only the activation fades.
Render both states from the same figure layout (for example with `display.add_overlay`), so that they line up.

<p align="center"><img src="https://raw.githubusercontent.com/carloalbertobarbano/cardkit/main/docs/brain_animated.svg" width="360" alt="brain example"></p>
<p align="center"><sub><a href="https://github.com/carloalbertobarbano/cardkit/blob/main/examples/brain.py">brain.py</a>: nilearn glass brains whose activation follows the BOLD curve, then a connectome next to node-strength bars drawn with cardkit.</sub></p>

Inside the SVG, images are embedded as palette PNGs (128 colours) to keep files small, so smooth gradients can show
some banding there. Mathtext follows matplotlib's `rcParams` (for example `plt.rcParams["mathtext.fontset"] = "cm"`),
and so does the default font family.

## Limitations (SVG only; the raster backend has none of these)

* `Txt.size` and `Img.arr` cannot be animated. Position, colour, opacity and the string itself can.
* A `Path` with `dash` cannot also animate `draw`: the SVG shows the dashes and cardkit warns you.
* Shapes whose number of vertices changes over time switch between shapes discretely instead of morphing.
* SVG time is sampled every `dt` (0.1 s by default): very fast motion may need a smaller `dt`.

---

## Reference

```
Scene(duration, size=(9, 9), background="#ffffff", *, origin=(0, 0), width_px=None, fps=30,
      intro=0.0, holds=None, hold=0.9, title="")
  .add(*prims)                      add primitives or lists of primitives; returns them
  .group(name, envelope=None)       register a group; also a context manager
  .duration                         loop length in display seconds
  .to_display(tau)                  scene time -> display time
  .frame(t, width_px=None, isolate=True)   PIL image at display time t
  .frame_at(tau)                    PIL image at scene time tau
  .storyboard(path=None, keys=None, cols=5, width=432, labels=False, check=False, save_frames=None)
  .storyboard_keys()                default storyboard keys: [(name, scene_time)], one finished frame per group
  .check_text(t, margin=0.25, min_alpha=0.2)
  .save(path, at=None, ...)         by extension: .svg .png .jpg .gif .webp .mp4
  .save_svg(path, at=None, dt=0.1, simplify=True, width=None)
  .save_all(stem, *formats, poster=None, fps=None, width_px=None, workers=None,
            gif_fps=15, gif_width=720, webp_quality=90, crf=16)
  .render_frames(folder, fps=None, width_px=None, workers=None)
Timeline(duration, intro=0.0, holds=None, hold=0.9)    .warp(T)  .to_display(tau)  .display_duration
```

## Examples and tests

* [`examples/quickstart.py`](https://github.com/carloalbertobarbano/cardkit/blob/main/examples/quickstart.py): the quick start above.
* [`examples/tour.py`](https://github.com/carloalbertobarbano/cardkit/blob/main/examples/tour.py): intro, groups, holds, motion along a path, keyframed colour, a counter,
  mathtext, an image, a progress indicator, the storyboard and the text check.
* [`examples/network.py`](https://github.com/carloalbertobarbano/cardkit/blob/main/examples/network.py), [`examples/three_d.py`](https://github.com/carloalbertobarbano/cardkit/blob/main/examples/three_d.py) and
  [`examples/brain.py`](https://github.com/carloalbertobarbano/cardkit/blob/main/examples/brain.py): the gallery (the last one needs `pip install nilearn`).
* `python docs/make_docs.py` rebuilds every demo image in `docs/`.
* `python -m unittest discover tests`. The tests include a SMIL evaluator that checks the compressed SVG animates exactly
  like the uncompressed one.

## License

MIT, see [LICENSE](https://github.com/carloalbertobarbano/cardkit/blob/main/LICENSE).
