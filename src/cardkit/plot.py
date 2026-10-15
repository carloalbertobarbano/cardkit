"""Scientific plots inside a scene: minimal L-shaped axes in slide coordinates that emit ordinary primitives.

    ax = scene.axes((1, 3, 4, 2.5), xlim=(0, 1), ylim=(-2, 2), xlabel="feature", ylabel="behaviour")
    ax.line(x, y, color="#0f9d8a", lw=2, draw=ramp(1, 3))          # a curve that draws itself
    ax.scatter(xp, lambda t: yp + np.sin(t), r=0.035, fill="#6c5ce7")   # points that move

Any data argument may be a function of time returning an array; everything is mapped from data to slide units, so the
plot animates in the raster and SVG outputs like any other primitive.
"""
import numpy as np

from .primitives import Poly, Path, Circ, Txt, BASE
from .text import U, text_width
from .timing import Fn


def _memo(f):
    """cache the last evaluation of a time function (many primitives read the same array at the same t)"""
    if not callable(f):
        return f
    last = [None, None]

    def g(t):
        if last[0] != t:
            last[0], last[1] = t, np.asarray(f(t), float)
        return last[1]
    return g


def _at(v, t):
    return v(t) if callable(v) else v


def _cap(size):
    """approximate cap height of a font size, in slide units"""
    return 0.72 * size / U


def _asc(size):
    """font ascent (matplotlib's text box above the baseline), in slide units"""
    return 0.9 * size / U


def _desc(size):
    """font descent (below the baseline), in slide units"""
    return 0.22 * size / U


def _revealed(P, u):
    """(N, 2) points with everything past fraction ``u`` of the samples collapsed onto the interpolated tip
    (a constant point count, so SVG can interpolate it smoothly)"""
    P = np.asarray(P, float)
    f = float(np.clip(u, 0, 1)) * (len(P) - 1)
    k = int(np.floor(f))
    if k >= len(P) - 1:
        return P
    out = P.copy()
    out[k + 1:] = P[k] + (P[k + 1] - P[k]) * (f - k)
    return out


def arrow(p0, p1, color="#8a93a3", lw=1.1, head=0.11, width=0.085, op=1.0, group=BASE, z=0):
    """Straight arrow from p0 to p1 with a filled triangular head (like matplotlib's "-|>").  Returns [Path, Poly]."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-u[1], u[0]])
    base = p1 - u * head
    return [Path([p0, base + u * 0.01], color, lw, op=op, group=group, z=z, cap="butt"),
            Poly([p1, base + n * width / 2, base - n * width / 2], fill=color, op=op, group=group, z=z)]


class Axes:
    """L-shaped axes occupying ``rect`` = (x, y, width, height) in slide units, showing data range ``xlim`` x ``ylim``.

    The frame (spines, ticks, tick labels, axis labels) is built immediately (``ax.frame``).  Plotting methods return the
    primitives they create; if the axes were made with ``scene.axes(...)`` they are also added to the scene (inside the
    current group, if any).  ``op``, ``group`` and ``z`` apply to the frame; data elements default to the same group and
    ``z + 1``, and accept their own ``op`` / ``group`` / ``z``."""

    def __init__(self, rect, xlim=(0.0, 1.0), ylim=(0.0, 1.0), *, xlabel=None, ylabel=None, xticks=None, xticklabels=None,
                 yticks=None, yticklabels=None, color="#8a93a3", lw=0.9, label_size=8.5, tick_size=7.8, labelpad=4 / U,
                 tick_len=0.058, tickpad=3.5 / U, xlabelpad=None, ylabelpad=None, op=1.0, group=BASE, z=0, scene=None):
        self.x0, self.y0, self.w, self.h = map(float, rect)
        self.xlim, self.ylim = tuple(map(float, xlim)), tuple(map(float, ylim))
        self.color, self.op, self.group, self.z, self.scene = color, op, group, z, scene
        if scene is not None and scene._stack and group == BASE:
            self.group = scene._stack[-1]
        kw = dict(op=op, group=self.group, z=z)
        x0, y0, x1, y1 = self.x0, self.y0, self.x0 + self.w, self.y0 + self.h
        F = [Path([(x0, y1), (x0, y0), (x1, y0)], color, lw, cap="butt", **kw)]
        below = 0.0
        if xticks is not None:
            labels = xticklabels if xticklabels is not None else [f"{v:g}" for v in xticks]
            for v, lab in zip(xticks, labels):
                x = float(self.X(v))
                F.append(Path([(x, y0), (x, y0 - tick_len)], color, lw * 0.9, cap="butt", **kw))
                F.append(Txt(lab, (x, y0 - tick_len - tickpad - _asc(tick_size)), tick_size, color, "center", **kw))
            below = tick_len + tickpad + _asc(tick_size) + _desc(tick_size)
        left = 0.0
        if yticks is not None:
            labels = yticklabels if yticklabels is not None else [f"{v:g}" for v in yticks]
            for v, lab in zip(yticks, labels):
                y = float(self.Y(v))
                F.append(Path([(x0, y), (x0 - tick_len, y)], color, lw * 0.9, cap="butt", **kw))
                F.append(Txt(lab, (x0 - tick_len - tickpad, y - _cap(tick_size) / 2), tick_size, color, "right", **kw))
            left = tick_len + tickpad + max(text_width(lab, tick_size) for lab in labels)
        if xlabel:
            pad = labelpad if xlabelpad is None else xlabelpad
            F.append(Txt(xlabel, ((x0 + x1) / 2, y0 - below - pad - _asc(label_size)), label_size, color, "center", **kw))
        if ylabel:
            pad = labelpad if ylabelpad is None else ylabelpad
            F.append(Txt(ylabel, (x0 - left - pad - _desc(label_size), (y0 + y1) / 2), label_size, color, "center", rotation=90,
                         **kw))
        self.frame = self._emit(F)

    # ------------------------------------------------------------ coordinates
    def X(self, x):
        """data x -> slide x"""
        return self.x0 + (np.asarray(x, float) - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w

    def Y(self, y):
        """data y -> slide y"""
        return self.y0 + (np.asarray(y, float) - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h

    def pts(self, x, y):
        """data points -> (N, 2) slide points; if x or y is a function of time, so is the result"""
        x, y = _memo(x), _memo(y)
        if callable(x) or callable(y):
            return lambda t: np.c_[self.X(_at(x, t)), self.Y(_at(y, t))]
        return np.c_[self.X(x), self.Y(y)]

    # ------------------------------------------------------------ plotting
    def _kw(self, kw):
        out = dict(group=self.group, z=self.z + 1)
        out.update(kw)
        return out

    def _emit(self, prims):
        if self.scene is not None:
            self.scene.add(prims)
        return prims

    def line(self, x, y, color="#0f9d8a", lw=1.5, reveal=None, **kw):
        """polyline through (x, y); x / y may be functions of time.
        ``reveal=ramp(t0, t1)`` grows it along the data order (e.g. along x, so several curves advance together), while
        ``draw=`` grows it by arc length."""
        P = self.pts(x, y)
        if reveal is not None and not callable(P):
            # static curve: revealing a fraction u of the samples == drawing the arc length up to sample u*(N-1),
            # so map it onto ``draw`` (the SVG then animates one dash offset instead of the whole point list)
            L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            L = L / max(L[-1], 1e-12)
            idx = np.arange(len(P))
            kw["draw"] = lambda t: float(np.interp(float(np.clip(reveal(t), 0, 1)) * (len(P) - 1), idx, L))
        elif reveal is not None:
            base = P
            P = lambda t: _revealed(_at(base, t), reveal(t))
            op = kw.pop("op", 1.0)
            kw["op"] = lambda t: _at(op, t) * (1.0 if reveal(t) > 0 else 0.0)
        return self._emit(Path(P, color, lw, **self._kw(kw)))

    def tip(self, x, y, reveal):
        """position of the growing end of ``line(x, y, reveal=reveal)``, as a function of time (put a dot there)"""
        P = self.pts(x, y)

        def f(t):
            Q = np.asarray(_at(P, t), float)
            u = float(np.clip(reveal(t), 0, 1)) * (len(Q) - 1)
            k = min(int(u), len(Q) - 2)
            return Q[k] + (Q[k + 1] - Q[k]) * (u - k)
        return Fn(f)

    def scatter(self, x, y, r=0.035, fill="#0f9d8a", **kw):
        """one circle per point (radius ``r`` in slide units); x / y may be functions of time"""
        P = self.pts(x, y)
        n = len(P(0.0)) if callable(P) else len(P)
        out = []
        for i in range(n):
            c = (lambda t, i=i: P(t)[i]) if callable(P) else P[i]
            out.append(Circ(c, r, fill=fill, **self._kw(kw)))
        return self._emit(out)

    def stems(self, x, y0, y1, color="#6c5ce7", lw=0.6, **kw):
        """vertical segments from y0 to y1 at each x (residuals, error bars); arrays or functions of time"""
        x, y0, y1 = np.asarray(x, float), _memo(y0), _memo(y1)
        out = []
        for i in range(len(x)):
            def pts(t, i=i):
                a, b = (np.broadcast_to(np.asarray(_at(v, t), float), x.shape)[i] for v in (y0, y1))
                return np.array([(self.X(x[i]), self.Y(a)), (self.X(x[i]), self.Y(b))])
            out.append(Path(pts if callable(y0) or callable(y1) else pts(0.0), color, lw, cap="butt", **self._kw(kw)))
        return self._emit(out)

    def fill_between(self, x, y0, y1, fill="#0f9d8a", **kw):
        """filled band between y0 and y1 (either may be a scalar, an array or a function of time)"""
        x, y0, y1 = np.asarray(x, float), _memo(y0), _memo(y1)

        def pts(t):
            lo = np.broadcast_to(_at(y0, t), x.shape); hi = np.broadcast_to(_at(y1, t), x.shape)
            return np.r_[np.c_[self.X(x), self.Y(hi)], np.c_[self.X(x[::-1]), self.Y(lo[::-1])]]
        return self._emit(Poly(pts if callable(y0) or callable(y1) else pts(0.0), fill=fill, **self._kw(kw)))

    def bar(self, x, height, width=0.8, bottom=0.0, fill="#0f9d8a", **kw):
        """vertical bars centred on x (data units); heights may be a function of time returning an array"""
        x, height = np.asarray(x, float), _memo(height)
        out = []
        for i in range(len(x)):
            def pts(t, i=i):
                h = np.atleast_1d(_at(height, t))[i]
                xa, xb = self.X(x[i] - width / 2), self.X(x[i] + width / 2)
                ya, yb = self.Y(bottom), self.Y(bottom + h)
                return np.array([(xa, ya), (xb, ya), (xb, yb), (xa, yb)])
            out.append(Poly(pts if callable(height) else pts(0.0), fill=fill, **self._kw(kw)))
        return self._emit(out)

    def vspan(self, x0, x1, fill="#0f9d8a", **kw):
        """shaded vertical band from x0 to x1 over the full height"""
        xa, xb = float(self.X(x0)), float(self.X(x1))
        return self._emit(Poly([(xa, self.y0), (xb, self.y0), (xb, self.y0 + self.h), (xa, self.y0 + self.h)], fill=fill,
                               **self._kw(kw)))

    def vline(self, x, color="#0f9d8a", lw=0.8, dash=None, **kw):
        """vertical line at data x over the full height"""
        xa = float(self.X(x))
        return self._emit(Path([(xa, self.y0), (xa, self.y0 + self.h)], color, lw, dash=dash, cap="butt", **self._kw(kw)))

    def hline(self, y, color="#0f9d8a", lw=0.8, dash=None, **kw):
        """horizontal line at data y over the full width"""
        ya = float(self.Y(y))
        return self._emit(Path([(self.x0, ya), (self.x0 + self.w, ya)], color, lw, dash=dash, cap="butt", **self._kw(kw)))

    def text(self, x, y, s, size=8.5, color="#1d2433", ha="left", coords="data", **kw):
        """text anchored at (x, y) on its baseline; ``coords="axes"`` places it in 0..1 axes fractions"""
        if coords == "axes":
            p = (self.x0 + x * self.w, self.y0 + y * self.h)
        else:
            p = (float(self.X(x)), float(self.Y(y)))
        return self._emit(Txt(s, p, size, color, ha, **self._kw(kw)))
