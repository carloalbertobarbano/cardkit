"""Text as outlines: one geometry for the raster and the SVG backends, no fonts needed by the viewer."""
import numpy as np
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

U = 43.2            # points per slide unit (a 9-unit card is 388.8 pt = 5.4 in wide)

_cache = {}


def _const(x, t):
    return x(t) if callable(x) else x


def text_path(s, size, weight="normal", style="normal", family=None):
    """Outline of ``s`` at the origin, in slide units: (vertices (N, 2), path codes (N,))."""
    k = (s, size, weight, style, family)
    if k not in _cache:
        tp = TextPath((0, 0), s, size=size, prop=FontProperties(family=family, weight=weight, style=style))
        _cache[k] = (np.asarray(tp.vertices) / U, np.asarray(tp.codes))
    return _cache[k]


def text_geom(p, t, v=_const):
    """Placed outline of a text primitive at time ``t``, or None if empty.  ``v(attr, t)`` evaluates attributes."""
    s = v(p.s, t)
    if not s:
        return None
    V_, C_ = text_path(s, p.size, p.weight, p.style, getattr(p, "family", None))
    if len(V_) == 0:
        return None
    xmin, xmax = V_[:, 0].min(), V_[:, 0].max()
    sh = {"left": 0.0, "center": -(xmin + xmax) / 2, "right": -xmax}[p.ha]
    x, y = v(p.xy, t)
    return V_ + np.array([x + sh, y]), C_


def text_bbox(p, t, v=_const):
    """(x0, x1, y0, y1) of a text primitive's ink at time ``t``, or None."""
    g = text_geom(p, t, v)
    if g is None:
        return None
    V_ = g[0]
    return V_[:, 0].min(), V_[:, 0].max(), V_[:, 1].min(), V_[:, 1].max()


def text_width(s, size=12, weight="normal", style="normal", family=None):
    """Ink width of a string in slide units (handy to size chips / boxes around labels)."""
    V_, _ = text_path(s, size, weight, style, family)
    return float(V_[:, 0].max() - V_[:, 0].min()) if len(V_) else 0.0
