"""Time helpers.

Attributes of primitives are functions of scene time.  These helpers build such functions without writing lambdas
(and without the classic late-binding bug of lambdas created in a loop).  They return :class:`Fn` objects, which
support arithmetic:  ``op = ramp(1.3, 1.9) * 0.95``,  ``op = fade(0, 6) * (1 - ramp(4, 4.5))``.
"""
import numpy as np
from matplotlib.colors import to_rgb, to_hex


def ease(x):
    """Smoothstep on [0, 1] (clipped)."""
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def seg(t, a, b):
    """Eased 0 -> 1 as t goes from a to b (0 before, 1 after)."""
    return ease((t - a) / (b - a))


def lin(t, a, b):
    """Linear 0 -> 1 as t goes from a to b (clipped)."""
    return float(np.clip((t - a) / (b - a), 0, 1))


class Fn:
    """A function of time with arithmetic: ``Fn * 0.5``, ``1 - Fn``, ``Fn * Fn`` ... all return new Fn."""
    __slots__ = ("f",)

    def __init__(self, f):
        self.f = f.f if isinstance(f, Fn) else f

    def __call__(self, t):
        return self.f(t)

    @staticmethod
    def _g(o):
        return o if callable(o) else (lambda t: o)

    def __mul__(self, o): g = Fn._g(o); return Fn(lambda t: self.f(t) * g(t))
    def __add__(self, o): g = Fn._g(o); return Fn(lambda t: self.f(t) + g(t))
    def __sub__(self, o): g = Fn._g(o); return Fn(lambda t: self.f(t) - g(t))
    def __rsub__(self, o): g = Fn._g(o); return Fn(lambda t: g(t) - self.f(t))
    def __truediv__(self, o): g = Fn._g(o); return Fn(lambda t: self.f(t) / g(t))
    def __neg__(self): return Fn(lambda t: -self.f(t))
    __rmul__, __radd__ = __mul__, __add__

    def shift(self, dt):
        """The same function, delayed by ``dt`` seconds."""
        return Fn(lambda t: self.f(t - dt))


def fn(f):
    """Wrap any ``t -> value`` callable as an :class:`Fn` (to get the arithmetic)."""
    return Fn(f)


def ramp(t0, t1, eased=True):
    """0 -> 1 between t0 and t1 (smoothstep by default).  The everyday fade-in / grow / draw-on."""
    return Fn((lambda t: float(seg(t, t0, t1))) if eased else (lambda t: lin(t, t0, t1)))


def fade(t0, t1=None, d=0.35):
    """Visible from t0 to t1: fades in over ``d`` after t0 and out over ``d`` before t1 (``d`` may be (d_in, d_out)).
    With ``t1=None`` it only fades in."""
    din, dout = (d, d) if np.isscalar(d) else d
    if t1 is None:
        return Fn(lambda t: float(seg(t, t0, t0 + din)))
    return Fn(lambda t: float(seg(t, t0, t0 + din) * (1 - seg(t, t1 - dout, t1))))


def pulse(period, lo=0.0, hi=1.0, phase=0.0):
    """Smooth periodic oscillation between lo and hi (sin^2), ``period`` in seconds."""
    return Fn(lambda t: lo + (hi - lo) * float(np.sin(np.pi * (t - phase) / period) ** 2))


def _lerp(a, b, k):
    if isinstance(a, str):
        return to_hex(np.array(to_rgb(a)) * (1 - k) + np.array(to_rgb(b)) * k)
    if np.isscalar(a):
        return a + (b - a) * k
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * k


def keyframes(keys, eased=True):
    """Piecewise interpolation through ``{t: value}`` (or a list of (t, value)).  Values may be numbers, arrays
    (e.g. points) or colours.  Constant before the first and after the last key.

        x = keyframes({0: 1.0, 2: 4.0, 5: 4.0, 6: 1.0})          # move right, hold, come back
        col = keyframes({3: "#2a9d8f", 4: "#e76f51"})            # colour change
    """
    items = sorted(keys.items() if isinstance(keys, dict) else keys, key=lambda kv: kv[0])
    ts = [k for k, _ in items]
    vs = [v for _, v in items]

    def f(t):
        if t <= ts[0]:
            return vs[0]
        if t >= ts[-1]:
            return vs[-1]
        i = int(np.searchsorted(ts, t, side="right")) - 1
        k = (t - ts[i]) / (ts[i + 1] - ts[i])
        return _lerp(vs[i], vs[i + 1], float(ease(k)) if eased else k)
    return Fn(f)


def tween(a, b, t0, t1, eased=True):
    """``a`` until t0, ``b`` after t1, interpolated in between (numbers, arrays or colours)."""
    return keyframes([(t0, a), (t1, b)], eased)


def along(pts, t0, t1, eased=True):
    """Position travelling along the polyline ``pts`` (constant speed, eased start/stop) between t0 and t1.
    ``Circ(along(path, 1, 3), 0.1)`` moves a dot along ``path``."""
    pts = np.asarray(pts, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s /= s[-1]

    def f(t):
        u = float(seg(t, t0, t1)) if eased else lin(t, t0, t1)
        return np.array([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])])
    return Fn(f)


def every(t0, period, n=None):
    """Integer step counter: 0 before t0, then +1 every ``period`` s (wrapping modulo ``n`` if given).
    Useful to cycle through examples:  ``color=fn(lambda t: COLORS[every(10, 0.9, 3)(t)])``."""
    def f(t):
        k = int(max(0.0, t - t0) / period + 1e-9)
        return k % n if n else k
    return Fn(f)


class Timeline:
    """Maps display time (what the viewer sees, 0 .. ``display_duration``) to scene time (what you author in).

    * ``intro``: seconds of scene time *before* 0.  Author an intro at negative times without re-timing the rest.
    * ``holds``: ``{scene_time: seconds}`` pauses; playback freezes at that scene time for that long (reading time),
      without touching any element timing.  A list of scene times uses ``hold`` seconds for each.
    """

    def __init__(self, duration, intro=0.0, holds=None, hold=0.9):
        if holds is None:
            holds = {}
        elif not isinstance(holds, dict):
            holds = {h: hold for h in holds}
        self.duration, self.intro = float(duration), float(intro)
        self.holds = sorted((float(h), float(d)) for h, d in holds.items())
        self.display_duration = self.intro + self.duration + sum(d for _, d in self.holds)

    def warp(self, T):
        """display time -> scene time"""
        T -= self.intro
        acc = 0.0
        for h, d in self.holds:
            if T <= h + acc:
                return T - acc
            if T <= h + acc + d:
                return h
            acc += d
        return T - acc

    def to_display(self, tau):
        """scene time -> display time (the first display instant showing that scene time)"""
        return tau + self.intro + sum(d for h, d in self.holds if h < tau)

    def __repr__(self):
        return f"Timeline(duration={self.duration}, intro={self.intro}, holds={dict(self.holds)})"
