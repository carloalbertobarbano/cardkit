"""Raster backend: primitives -> matplotlib artists -> RGBA frames."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Polygon as MPoly, PathPatch, Circle as MCircle
from matplotlib.path import Path as MPath
from PIL import Image

from .text import text_geom


def draw(ax, prims, t, ctx, only=None, env1=False):
    """Add every primitive visible at display time ``t`` to ``ax``.
    only: restrict to these groups.  env1: ignore group envelopes (when a group is composited as its own layer)."""
    v = ctx.v
    for k, p in enumerate(sorted(prims, key=lambda q: q.z)):
        if only is not None and p.group not in only:
            continue
        env = 1.0 if env1 else ctx.env(p.group, t)
        a = env * v(p.op, t)
        if a < 0.004:
            continue
        zo = p.z + k * 1e-6
        if p.kind == "poly":
            pts = np.asarray(v(p.pts, t)); fc = v(p.fill, t)
            ax.add_patch(MPoly(pts, closed=True, fc=fc if fc else "none", ec=v(p.stroke, t) or "none", lw=v(p.lw, t), alpha=a,
                               zorder=zo, joinstyle="round"))
        elif p.kind == "path":
            pts = np.asarray(v(p.pts, t)); d = float(v(p.draw, t)); lw = v(p.lw, t)
            if d <= 0:
                continue
            if d < 1:
                seg_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]; L = d * seg_[-1]
                kk = min(np.searchsorted(seg_, L, side="right") - 1, len(pts) - 2)
                q = pts[kk] + (pts[kk + 1] - pts[kk]) * (L - seg_[kk]) / max(seg_[kk + 1] - seg_[kk], 1e-9)
                pts = np.vstack([pts[:kk + 1], q])
            ls = "-" if p.dash is None else (0, (p.dash[0] * ctx.unit / lw, p.dash[1] * ctx.unit / lw))
            ax.plot(pts[:, 0], pts[:, 1], color=v(p.color, t), lw=lw, alpha=a, zorder=zo, solid_capstyle=p.cap,
                    dash_capstyle=p.cap, ls=ls, solid_joinstyle="round")
        elif p.kind == "circ":
            fc = v(p.fill, t); c = v(p.c, t)
            ax.add_patch(MCircle(c, v(p.r, t), fc=fc if fc else "none", ec=v(p.stroke, t) or "none", lw=v(p.lw, t), alpha=a,
                                 zorder=zo))
        elif p.kind == "txt":
            g = text_geom(p, t, v)
            if g is None:
                continue
            ax.add_patch(PathPatch(MPath(g[0], g[1]), fc=v(p.color, t), ec="none", lw=0, alpha=a, zorder=zo))
        elif p.kind == "img":
            x0, x1, y0, y1 = v(p.extent, t)
            ax.imshow(p.arr, extent=(x0, x1, y0, y1), alpha=a, zorder=zo, interpolation=p.interp, aspect="auto")


def canvas(ctx, width_px):
    """Transparent figure + axes covering ``ctx.view`` exactly, ``width_px`` pixels wide."""
    x0, x1, y0, y1 = ctx.view
    w_in, h_in = (x1 - x0) * ctx.unit / 72, (y1 - y0) * ctx.unit / 72
    fig = Figure(figsize=(w_in, h_in), dpi=width_px / w_in)
    FigureCanvasAgg(fig)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.axis("off"); ax.patch.set_alpha(0)
    return fig, ax


def grab(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy(), "RGBA")


def background_image(bg, size):
    """RGBA PIL image of ``size`` from a colour, a PIL image or an array."""
    if isinstance(bg, Image.Image):
        im = bg.convert("RGBA")
    elif isinstance(bg, np.ndarray):
        a = bg if bg.dtype == np.uint8 else (np.clip(bg, 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(a).convert("RGBA")
    else:
        return Image.new("RGBA", size, bg)
    return im if im.size == size else im.resize(size, Image.LANCZOS)


def frame(prims, t, ctx, bg, width_px, isolate=True):
    """Render display time ``t`` to an RGB PIL image.

    isolate=True composites every group that is mid-fade as its own layer, faded as a whole (like the SVG group opacity),
    so overlapping translucent shapes inside a group do not show seams while it fades."""
    if not isolate:
        fig, ax = canvas(ctx, width_px)
        draw(ax, prims, t, ctx)
        im = grab(fig)
        return Image.alpha_composite(background_image(bg, im.size), im).convert("RGB")
    groups = list(dict.fromkeys(p.group for p in prims))
    envs = {g: ctx.env(g, t) for g in groups}
    full = [g for g in groups if envs[g] >= 0.995]
    part = [g for g in groups if 0.004 < envs[g] < 0.995]
    fig, ax = canvas(ctx, width_px)
    draw(ax, prims, t, ctx, only=full, env1=True)
    im = grab(fig)
    out = Image.alpha_composite(background_image(bg, im.size), im)
    for g in part:
        f2, a2 = canvas(ctx, width_px)
        draw(a2, prims, t, ctx, only=[g], env1=True)
        arr = np.asarray(grab(f2)).copy()
        arr[..., 3] = (arr[..., 3] * envs[g]).astype(np.uint8)
        out = Image.alpha_composite(out, Image.fromarray(arr, "RGBA"))
    return out.convert("RGB")
