"""cardkit: one animated scene -> GIF / WebP / MP4 frames *and* a looping, script-free animated SVG that look the same.

    import cardkit as ck
    from cardkit import Txt, Path, Poly, Circ, Img, ramp, fade

    scene = ck.Scene(duration=6, size=(9, 9), background="#ffffff")
    scene.add(Txt("Hello", (0.5, 8), 24, "#111111", weight="bold", op=ramp(0.2, 0.8)))
    scene.save_all("out/hello", "gif", "svg")

Every primitive attribute is a constant or a function of time.  Text is converted to outlines (mathtext works), so the
raster and SVG outputs are identical and need no fonts.  Coordinates are slide units with y up; 1 unit = 43.2 pt.
See README.md for a tour.
"""
from .primitives import Prim, Poly, Path, Circ, Txt, Img, BASE, Polygon, Line, Circle, Text, Image
from .timing import Fn, fn, ease, seg, lin, ramp, fade, pulse, keyframes, tween, along, every, Timeline
from .shapes import rect, rounded_rect, bezier, arc, ngon, arrow_head, mix, shade, lighten, colorize
from .text import U, text_path, text_width
from .text import text_geom as _text_geom, text_bbox as _text_bbox
from .scene import Scene, Group
from .context import Context
from . import render as _render, svg as _svg, three

__version__ = "0.2.0"

__all__ = ["Scene", "Group", "Poly", "Path", "Circ", "Txt", "Img", "Fn", "fn", "ease", "seg", "lin", "ramp", "fade",
           "pulse", "keyframes", "tween", "along", "every", "Timeline", "rect", "rounded_rect", "bezier", "arc", "ngon",
           "arrow_head", "mix", "shade", "lighten", "colorize", "text_width", "U"]


# ---------------------------------------------------------------------------------------------------------------------
# Module-level API of the original single-file cardkit (kept so that older scripts run unchanged):
#     K.DUR = 25.0; K.WARP = warp; K.GROUP_ENV["f1"] = env; K.GROUP_WIN["f1"] = (t0, t1)
#     K.draw_all(ax, prims, t, only=None, env1=False); K.export_svg(prims, path, bg, title)
# These globals describe a single 9 x 9 card.  New code should use Scene, which keeps this state per scene.
FS = 388.8          # SVG viewBox size of the 9 x 9 card
DUR = 24.0          # loop duration (display time)
GROUP_ENV = {}      # group -> env(scene_t)
GROUP_WIN = {}      # group -> (t0, t1) display-time sampling window for the SVG exporter
WARP = lambda t: t  # display time -> scene time


def _legacy_ctx():
    return Context(DUR, WARP, GROUP_ENV, GROUP_WIN, (0.0, 9.0, 0.0, 9.0), U, svg_size=(900, 900), top=FS)


def _v(x, t):
    return x(WARP(t)) if callable(x) else x


def f2(v):
    return f"{v:.2f}"


hexc = _svg.hexc


def text_geom(p, t):
    return _text_geom(p, t, _v)


def text_bbox(p, t):
    return _text_bbox(p, t, _v)


def draw_all(ax, prims, t, only=None, env1=False):
    """Draw ``prims`` at display time ``t`` on a matplotlib axes spanning the card (legacy API)."""
    _render.draw(ax, prims, t, _legacy_ctx(), only, env1)


def export_svg(prims, path, bg, title="", dt=0.1, simplify=False):
    """Animated SVG of a 9 x 9 card (legacy API).  Returns the size in bytes."""
    return _svg.export(prims, path, _legacy_ctx(), bg, title, dt, simplify)
