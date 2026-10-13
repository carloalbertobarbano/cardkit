"""Point-list builders for common shapes (feed them to Poly / Path) and colour helpers."""
import numpy as np
from matplotlib.colors import to_rgb, to_hex


# ---------------------------------------------------------------- shapes (all return (N, 2) arrays in slide units)
def rect(x0, y0, x1, y1):
    """Axis-aligned rectangle, corners (x0, y0) and (x1, y1)."""
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)


def rounded_rect(x0, y0, x1, y1, r, n=6):
    """Rectangle with corner radius ``r`` (``n`` points per corner)."""
    pts = []
    for cx, cy, a0 in [(x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270)]:
        for a in np.linspace(a0, a0 + 90, n):
            pts.append((cx + r * np.cos(np.radians(a)), cy + r * np.sin(np.radians(a))))
    return np.array(pts)


def bezier(p0, c1, c2, p1, n=40):
    """Cubic Bezier from p0 to p1 with control points c1, c2 (works in 2-D or 3-D)."""
    s = np.linspace(0, 1, n)[:, None]
    p0, c1, c2, p1 = (np.array(v, float) for v in (p0, c1, c2, p1))
    return (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * c1 + 3 * (1 - s) * s ** 2 * c2 + s ** 3 * p1


def arc(c, r, a0, a1, n=48):
    """Circular arc around ``c`` from angle a0 to a1 (degrees, counter-clockwise)."""
    a = np.radians(np.linspace(a0, a1, n))
    return np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]


def ngon(c, r, k, rot=90.0):
    """Regular polygon with ``k`` vertices (rot = angle of the first vertex, degrees)."""
    return arc(c, r, rot, rot + 360.0 * (k - 1) / k, k)


def arrow_head(tip, direction, length=0.18, width=0.14):
    """Triangle for an arrow head at ``tip`` pointing along ``direction``."""
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    n = np.array([-u[1], u[0]])
    tip = np.asarray(tip, float)
    b = tip - u * length
    return np.array([tip, b + n * width / 2, b - n * width / 2])


# ---------------------------------------------------------------- colours
def mix(c1, c2, k):
    """Blend two colours, k=0 -> c1, k=1 -> c2."""
    return to_hex(np.array(to_rgb(c1)) * (1 - k) + np.array(to_rgb(c2)) * k)


def shade(c, k):
    """Scale a colour's brightness (k < 1 darker, k > 1 lighter, clipped)."""
    return to_hex(np.clip(np.array(to_rgb(c)) * k, 0, 1))


def lighten(c, k):
    """Blend towards white by k (0..1)."""
    return mix(c, "#ffffff", k)


def colorize(c, fn):
    """Apply ``fn`` to a colour that may itself be a function of time (keeps it a function)."""
    return (lambda t: fn(c(t))) if callable(c) else fn(c)
