"""Rendering context shared by the raster and SVG backends: clock, group envelopes, coordinate frame."""
from .text import U


def _identity(t):
    return t


class Context:
    """
    duration   display duration of the loop (s)
    warp       display time -> scene time
    envelopes  {group: env(scene_t)} group opacity
    windows    {group: (t0, t1)} display-time span the SVG exporter samples a group in
    view       (x0, x1, y0, y1) visible region in slide units
    svg_size   (width, height) of the SVG element in px
    """

    def __init__(self, duration, warp=None, envelopes=None, windows=None, view=(0.0, 9.0, 0.0, 9.0), unit=U,
                 svg_size=None, top=None):
        self.duration = duration
        self.warp = warp or _identity
        self.envelopes = {} if envelopes is None else envelopes
        self.windows = {} if windows is None else windows
        self.view, self.unit = tuple(view), unit
        x0, x1, y0, y1 = self.view
        self.vb_w, self.vb_h = (x1 - x0) * unit, (y1 - y0) * unit          # SVG viewBox size (pt)
        self.top = y1 * unit if top is None else top                       # SVG y of the slide-space y = 0 line
        self.left = x0 * unit
        self.svg_size = svg_size or (round(100 * (x1 - x0)), round(100 * (y1 - y0)))

    def v(self, x, t):
        """value of an attribute at display time t"""
        return x(self.warp(t)) if callable(x) else x

    def env(self, group, t):
        """group envelope at display time t"""
        f = self.envelopes.get(group)
        return 1.0 if f is None else f(self.warp(t))
