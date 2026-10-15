"""Scene primitives.

Every attribute may be a constant or a function of (scene) time ``t -> value``.
Coordinates are in slide units with y pointing up; line widths and font sizes are in points.
"""

BASE = "ui"          # name of the default group: always visible, no envelope


class Prim:
    kind = "prim"
    win = None       # optional (t0, t1) display-time window the SVG exporter samples this primitive in

    def __init__(self, group=BASE, z=0, op=1.0):
        self.group, self.z, self.op = group, z, op

    def __repr__(self):
        return f"<{type(self).__name__} group={self.group!r} z={self.z}>"


class Poly(Prim):
    """Closed polygon.  ``pts``: (N, 2) vertices.  ``fill`` / ``stroke``: colours or None, ``lw``: stroke width (pt)."""
    kind = "poly"

    def __init__(self, pts, fill=None, stroke=None, lw=0.0, op=1.0, group=BASE, z=0):
        super().__init__(group, z, op)
        self.pts, self.fill, self.stroke, self.lw = pts, fill, stroke, lw


class Path(Prim):
    """Open polyline stroke.  ``draw``: fraction drawn (0..1), animate it to "draw on" the line.
    ``dash``: (on, off) lengths in slide units (static, cannot be combined with an animated ``draw``)."""
    kind = "path"

    def __init__(self, pts, color, lw=1.5, op=1.0, draw=1.0, dash=None, group=BASE, z=0, cap="round"):
        super().__init__(group, z, op)
        self.pts, self.color, self.lw, self.draw, self.dash, self.cap = pts, color, lw, draw, dash, cap


class Circ(Prim):
    """Circle with centre ``c`` and radius ``r`` (slide units)."""
    kind = "circ"

    def __init__(self, c, r, fill=None, stroke=None, lw=0.0, op=1.0, group=BASE, z=0):
        super().__init__(group, z, op)
        self.c, self.r, self.fill, self.stroke, self.lw = c, r, fill, stroke, lw


class Txt(Prim):
    """Text, converted to outlines (mathtext such as ``"$\\varphi_1$"`` works).
    ``xy``: anchor on the baseline, ``ha``: "left" | "center" | "right", ``size`` in points (constant).
    ``s`` may be a function of time (e.g. a counter); each distinct string becomes its own outline in the SVG.
    ``rotation``: degrees counter-clockwise around the anchor (constant), e.g. 90 for a y-axis label."""
    kind = "txt"

    def __init__(self, s, xy, size=12, color="#000", ha="left", weight="normal", style="normal", op=1.0, group=BASE, z=0,
                 family=None, rotation=0.0):
        super().__init__(group, z, op)
        self.s, self.xy, self.size, self.color, self.ha, self.weight, self.style = s, xy, size, color, ha, weight, style
        self.family, self.rotation = family, rotation


class Img(Prim):
    """Raster image.  ``arr``: (H, W, 3|4) array (float 0..1 or uint8).  ``extent``: (x0, x1, y0, y1), may be a function of t."""
    kind = "img"

    def __init__(self, arr, extent, op=1.0, group=BASE, z=0, interp="bilinear"):
        super().__init__(group, z, op)
        self.arr, self.extent, self.interp = arr, extent, interp


# readable aliases
Polygon, Line, Circle, Text, Image = Poly, Path, Circ, Txt, Img
