"""Lightweight 3-D: an oblique camera, Lambert-lit faces, boxes, matrix slabs and tubes, all built from Poly.

World axes: X to the right, D into the screen (depth), Z up.  The camera is fixed, so back faces are culled and faces
are painter-sorted once, at build time; the shapes can still grow, move and change colour over time.

    cam = Camera(origin=(1.0, 2.0))
    scene.add(box(cam, (0, 0, 0), (1, 1, lambda t: 2 * ramp(0, 1)(t)), "#0f9d8a"))   # a bar growing upwards
"""
import numpy as np

from .primitives import Poly
from .shapes import mix, shade, colorize
from .timing import seg


class Camera:
    """Oblique ("cabinet") projection: a point (X, D, Z) lands at origin + (X + dx * D, dy * D + Z).
    ``light``: direction towards the light (world axes).  ``ambient``: brightness of faces facing away from it."""

    def __init__(self, origin=(0.0, 0.0), dx=0.4, dy=0.3, light=(-0.45, -0.55, 0.75), ambient=0.56, specular=0.5):
        self.origin, self.dx, self.dy = np.asarray(origin, float), dx, dy
        v = np.array([dx, -1.0, dy]); self.view = v / np.linalg.norm(v)          # towards the viewer
        l_ = np.asarray(light, float); self.light = l_ / np.linalg.norm(l_)
        self.ambient, self.specular = ambient, specular

    def project(self, P):
        """(N, 3) world points -> (N, 2) slide coordinates"""
        P = np.atleast_2d(np.asarray(P, float))
        return np.c_[self.origin[0] + P[:, 0] + self.dx * P[:, 1], self.origin[1] + self.dy * P[:, 1] + P[:, 2]]

    def depth(self, P):
        """larger = closer to the viewer (use it to order separate objects: z=cam.depth(center))"""
        return float(np.mean(np.atleast_2d(np.asarray(P, float)), axis=0) @ self.view)

    def visible(self, normal):
        return float(np.asarray(normal, float) @ self.view) > 0

    def lit(self, color, normal):
        """colour of a face with this normal: diffuse shading plus a soft specular highlight"""
        n = np.asarray(normal, float); n = n / np.linalg.norm(n)
        l_ = float(max(0.0, n @ self.light))
        base = shade(color, self.ambient + (1 - self.ambient) * l_)
        sp = max(0.0, (l_ - 0.8) / 0.2) * self.specular
        return mix(base, "#ffffff", sp) if sp > 0 else base

    def face(self, verts, normal, color, **kw):
        """one lit Poly; ``verts`` may be a function of time returning (N, 3) world points"""
        fill = colorize(color, lambda c: self.lit(c, normal))
        pts = (lambda t: self.project(verts(t))) if callable(verts) else self.project(verts)
        return Poly(pts, fill=fill, **kw)


def _val(x, t):
    return x(t) if callable(x) else x


def box(cam, corner, size, color, op=1.0, group="ui", z=0.0, stroke=None, lw=0.0):
    """Axis-aligned box with its lower-front-left corner at ``corner`` = (X, D, Z) and ``size`` = (sx, sd, sz).
    Any size component may be a function of time (e.g. a bar that grows).  Returns the visible faces as Poly."""
    def verts(axis, sign):
        def f(t):
            x0, d0, z0 = corner
            sx, sd, sz = (_val(s, t) for s in size)
            lo, hi = np.array([x0, d0, z0], float), np.array([x0 + sx, d0 + sd, z0 + sz], float)
            q = [lo.copy() for _ in range(4)]
            other = [a for a in range(3) if a != axis]
            for k, (u, v) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
                q[k][axis] = hi[axis] if sign > 0 else lo[axis]
                q[k][other[0]] = (lo, hi)[u][other[0]]
                q[k][other[1]] = (lo, hi)[v][other[1]]
            return np.array(q)
        return f if any(callable(s) for s in size) else f(0.0)
    faces = []
    for axis in range(3):
        for sign in (-1, 1):
            n = np.zeros(3); n[axis] = sign
            if cam.visible(n):
                faces.append((n, verts(axis, sign)))
    faces.sort(key=lambda f: f[0] @ cam.view)                                    # far faces first
    sk = dict(stroke=stroke, lw=lw) if stroke else {}
    return [cam.face(v, n, color, op=op, group=group, z=z + 1e-4 * k, **sk) for k, (n, v) in enumerate(faces)]


def slab(cam, corner, values, cell, colors, thickness=0.15, gap_color="#ffffff", reveal=None, side="#9aa3b5",
         group="ui", z=0.0):
    """A matrix lying flat as a thick slab (the "3-D FC matrix").  ``values``: (n, m) array; ``colors(v)`` maps a value to
    a colour.  Rows go into the screen (+D), columns to the right (+X).  ``reveal(k)``: opacity function for cell k
    (row-major), e.g. ``lambda k: ramp(1 + 0.01 * k, 1.3 + 0.01 * k)``; the slab sides use ``reveal(0)``."""
    values = np.asarray(values)
    n, m = values.shape
    X0, D0, Z0 = corner
    op = (lambda k: 1.0) if reveal is None else reveal
    P = []
    top = Z0 + thickness
    front = [(X0, D0, Z0), (X0 + m * cell, D0, Z0), (X0 + m * cell, D0, top), (X0, D0, top)]
    right = [(X0 + m * cell, D0, Z0), (X0 + m * cell, D0 + n * cell, Z0), (X0 + m * cell, D0 + n * cell, top), (X0 + m * cell, D0, top)]
    P.append(cam.face(front, (0, -1, 0), side, op=op(0), group=group, z=z - 0.01))
    P.append(cam.face(right, (1, 0, 0), side, op=op(0), group=group, z=z - 0.01))
    for i in range(n):
        for j in range(m):
            d = D0 + (n - 1 - i) * cell                                               # row 0 at the back
            q = [(X0 + j * cell, d, top), (X0 + (j + 1) * cell, d, top), (X0 + (j + 1) * cell, d + cell, top), (X0 + j * cell, d + cell, top)]
            P.append(Poly(cam.project(q), fill=colors(values[i, j]), stroke=gap_color, lw=0.5, op=op(i * m + j), group=group, z=z))
    return P


def tube(cam, path, r, color, t0=None, dur=1.0, head=True, sides=9, samples=26, group="ui", z=0.0, op=1.0):
    """Shaded 3-D tube along ``path`` ((N, 3) world points) with a cone head.  If ``t0`` is given the tube grows
    along the path between t0 and t0 + dur.  ``color`` may be a function of time.  Returns a list of Poly."""
    Pc = np.asarray(path, float)
    seglen = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pc, axis=0), axis=1))]
    Lc = 5.6 * r if head else 0.0
    sc = np.linspace(0, seglen[-1] - Lc, samples)
    sh = np.c_[[np.interp(sc, seglen, Pc[:, k]) for k in range(3)]].T
    Tn = np.gradient(sh, axis=0); Tn /= np.linalg.norm(Tn, axis=1, keepdims=True)
    Sv = np.cross(Tn, [0, 0, 1.0]); bad = np.linalg.norm(Sv, axis=1) < 1e-3; Sv[bad] = [1.0, 0, 0]
    Sv /= np.linalg.norm(Sv, axis=1, keepdims=True); Nv = np.cross(Sv, Tn)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False); ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    ring_n = [ca * Sv[i] + sa * Nv[i] for i in range(samples)]
    ring = [sh[i] + r * ring_n[i] for i in range(samples)]
    quads = []                                                                  # (world verts, normal, reveal fraction)
    for i in range(samples - 1):
        for j in range(sides):
            j2 = (j + 1) % sides
            nrm = ring_n[i][j] + ring_n[i][j2] + ring_n[i + 1][j] + ring_n[i + 1][j2]
            quads.append(([ring[i][j], ring[i][j2], ring[i + 1][j2], ring[i + 1][j]], nrm / np.linalg.norm(nrm), i / samples * 0.9))
    if head:
        Te, base = Tn[-1], sh[-1]; tip = base + Te * Lc; R = 2.9 * r
        ring_c = [base + R * (ca[k, 0] * Sv[-1] + sa[k, 0] * Nv[-1]) for k in range(sides)]
        for j in range(sides):
            j2 = (j + 1) % sides
            quads.append(([ring[-1][j], ring[-1][j2], ring_c[j2], ring_c[j]], -Te, 0.9))
            nj = (ring_c[j] - base) / R + Te * (R / Lc); nj2 = (ring_c[j2] - base) / R + Te * (R / Lc)
            nrm = nj + nj2
            quads.append(([ring_c[j], ring_c[j2], tip], nrm / np.linalg.norm(nrm), 0.93))
    vis = sorted((q for q in quads if cam.visible(q[1])), key=lambda q: np.mean(q[0], axis=0) @ cam.view)
    P = []
    for k, (vs, nrm, fr) in enumerate(vis):
        if t0 is None:
            o = op
        else:
            a0 = t0 + fr * dur
            o = (lambda t, a0=a0: float(seg(t, a0, a0 + 0.12)) * _val(op, t))
        fill = colorize(color, lambda c, nrm=nrm: cam.lit(c, nrm))
        P.append(Poly(cam.project(vs), fill=fill, stroke=fill, lw=0.5, op=o, group=group, z=z + k * 1e-4))
    return P
