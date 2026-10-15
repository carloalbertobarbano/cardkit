"""Vector backend: native SVG with SMIL <animate> elements, built by sampling the same attribute functions.

The result loops forever, needs no script or font, and plays inside a plain <img> tag.
"""
import io
import base64
import warnings
from xml.sax.saxutils import escape

import numpy as np
from matplotlib.colors import to_hex
from PIL import Image

from .text import text_geom, text_path, text_layout, FONT_SCALE


def f2(v): return f"{v:.2f}"
def hexc(c): return to_hex(c) if not isinstance(c, str) or not c.startswith("#") else c
def _num(v): return f"{v:.4f}".rstrip("0").rstrip(".")


def png_b64(arr, scale=1.0, colors=128):
    """RGBA array -> base64 PNG (palette-quantised to ``colors``; ``colors=None`` keeps full colour)."""
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = (a * 255).astype(np.uint8)
    im = Image.fromarray(a).convert("RGBA") if a.ndim == 2 or a.shape[-1] != 4 else Image.fromarray(a, "RGBA")
    if scale != 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    b = io.BytesIO()
    (im.quantize(colors=colors, method=Image.Quantize.FASTOCTREE) if colors else im).save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode()


class _Exporter:
    def __init__(self, ctx, dt=0.1, simplify=False, at=None, glyphs=False):
        self.ctx, self.dt, self.simplify, self.at, self.glyphs = ctx, dt, simplify, at, glyphs
        self.glyph_ids, self.defs = {}, []
        U, top, left = ctx.unit, ctx.top, ctx.left
        self.X = lambda x: x * U - left if left else x * U          # keeps legacy output byte-identical when left == 0
        self.Y = lambda y: top - y * U

    # ------------------------------------------------------------ sampling
    def times(self, win):
        if self.at is not None:
            return np.array([float(self.at)])
        t0, t1 = win
        n = max(2, int(round((t1 - t0) / self.dt)) + 1)
        return np.linspace(t0, t1, n)

    def anim(self, attr, ts, vals, discrete=False):
        """SMIL animate over the whole loop; constant before/after the sampling window."""
        D = self.ctx.duration
        if self.simplify:                                         # drop samples inside runs of identical values
            keep = [i for i in range(len(vals)) if i == 0 or i == len(vals) - 1 or not (vals[i - 1] == vals[i] == vals[i + 1])]
            ts, vals = [ts[i] for i in keep], [vals[i] for i in keep]
        kt = [0.0] + [t / D for t in ts] + [1.0]; vv = [vals[0]] + list(vals) + [vals[-1]]
        if kt[1] <= 1e-9: kt, vv = kt[1:], vv[1:]; kt[0] = 0.0
        if kt[-2] >= 1 - 1e-9: kt, vv = kt[:-1], vv[:-1]; kt[-1] = 1.0
        return (f'<animate attributeName="{attr}" dur="{D}s" repeatCount="indefinite" keyTimes="{";".join(f"{k:.4f}" for k in kt)}" values="{";".join(vv)}"'
                + (' calcMode="discrete"' if discrete else "") + "/>")

    def attr(self, val, name, ts, fmt, discrete=False):
        """-> (static value string, <animate> string or '')"""
        vals = [fmt(self.ctx.v(val, t)) for t in ts]
        if all(v == vals[0] for v in vals):
            return vals[0], ""
        return vals[0], self.anim(name, ts, vals, discrete)

    def geom_attr(self, val, name, ts, fmt):
        """points / d: SMIL only interpolates equal vertex counts, otherwise switch values discretely"""
        if callable(val):
            counts = {len(np.asarray(self.ctx.v(val, t))) for t in ts}
            if len(counts) > 1:
                return self.attr(val, name, ts, fmt, True)
        return self.attr(val, name, ts, fmt)

    def pts_str(self, pts): return " ".join(f"{self.X(x):.2f},{self.Y(y):.2f}" for x, y in np.asarray(pts))
    def d_str(self, pts): return "M" + " L".join(f"{self.X(x):.2f},{self.Y(y):.2f}" for x, y in np.asarray(pts))

    def path_d(self, V_, C_, P=None):
        out, i = [], 0
        P = P or (lambda v: f"{self.X(v[0]):.2f},{self.Y(v[1]):.2f}")
        while i < len(C_):
            c = C_[i]
            if c == 1: out.append("M" + P(V_[i])); i += 1
            elif c == 2: out.append("L" + P(V_[i])); i += 1
            elif c == 3: out.append("Q" + P(V_[i]) + " " + P(V_[i + 1])); i += 2
            elif c == 4: out.append("C" + P(V_[i]) + " " + P(V_[i + 1]) + " " + P(V_[i + 2])); i += 3
            elif c == 79: out.append("Z"); i += 1
            else: i += 1
        return "".join(out)

    # ------------------------------------------------------------ elements
    def element(self, p, ts):
        v, U = self.ctx.v, self.ctx.unit
        op0, opa = self.attr(p.op, "opacity", ts, lambda x: f2(float(x)))
        opacity_attr = f' opacity="{op0}"'
        lw = lambda name: self.attr(p.lw, name, ts, lambda x: f"{x}")
        if p.kind == "poly":
            pts0, pa = self.geom_attr(p.pts, "points", ts, self.pts_str)
            fill0, fa = self.attr(p.fill if p.fill else "none", "fill", ts, lambda c: hexc(c) if c != "none" else "none", True)
            if p.stroke:
                st0, sa_ = self.attr(p.stroke, "stroke", ts, hexc, True); w0, wa = lw("stroke-width")
                stroke_attrs = f' stroke="{st0}" stroke-width="{w0}" stroke-linejoin="round"'
            else:
                sa_, wa, stroke_attrs = "", "", ""
            return f'<polygon points="{pts0}" fill="{fill0}"{stroke_attrs}{opacity_attr}>{pa}{fa}{sa_}{wa}{opa}</polygon>'
        if p.kind == "path":
            d0, da = self.geom_attr(p.pts, "d", ts, self.d_str)
            col0, ca = self.attr(p.color, "stroke", ts, hexc, True)
            dr0, dra = self.attr(p.draw, "stroke-dashoffset", ts, lambda x: f2(1.04 * (1 - float(x))))
            w0, wa = lw("stroke-width")
            if p.dash is not None:
                if dra:
                    warnings.warn("cardkit: Path with both `dash` and an animated `draw`: the SVG shows the dashes, not the draw-on")
                dashattr, extra = f' stroke-dasharray="{p.dash[0] * U:.2f} {p.dash[1] * U:.2f}"', ""
            elif dra:
                dashattr, extra = ' pathLength="1" stroke-dasharray="1 2"' + f' stroke-dashoffset="{dr0}"', dra
            else:
                dashattr, extra = "", ""
            return (f'<path d="{d0}" fill="none" stroke="{col0}" stroke-width="{w0}" stroke-linecap="{p.cap}" stroke-linejoin="round"'
                    f'{dashattr}{opacity_attr}>{da}{ca}{extra}{wa}{opa}</path>')
        if p.kind == "circ":
            c0, cx_a = self.attr(p.c, "cx", ts, lambda c: f2(self.X(c[0]))); _, cy_a = self.attr(p.c, "cy", ts, lambda c: f2(self.Y(c[1])))
            cy0 = f2(self.Y(v(p.c, ts[0])[1])); r0, ra = self.attr(p.r, "r", ts, lambda x: f2(x * U))
            fill0, fa = self.attr(p.fill if p.fill else "none", "fill", ts, lambda c: hexc(c) if c != "none" else "none", True)
            if p.stroke:
                st0, sa_ = self.attr(p.stroke, "stroke", ts, hexc, True); w0, wa = lw("stroke-width")
                st = f' stroke="{st0}" stroke-width="{w0}"'
            else:
                st, sa_, wa = "", "", ""
            return f'<circle cx="{c0}" cy="{cy0}" r="{r0}" fill="{fill0}"{st}{opacity_attr}>{cx_a}{cy_a}{ra}{fa}{opa}{sa_}{wa}</circle>'
        if p.kind == "txt":
            return self.text(p, ts)
        if p.kind == "img":
            x0, xa = self.attr(p.extent, "x", ts, lambda e: f2(self.X(e[0]))); y0, ya = self.attr(p.extent, "y", ts, lambda e: f2(self.Y(e[3])))
            w0, wa = self.attr(p.extent, "width", ts, lambda e: f2((e[1] - e[0]) * U)); h0, ha = self.attr(p.extent, "height", ts, lambda e: f2((e[3] - e[2]) * U))
            return (f'<image x="{x0}" y="{y0}" width="{w0}" height="{h0}" preserveAspectRatio="none"{opacity_attr} '
                    f'xlink:href="data:image/png;base64,{png_b64(p.arr)}">{xa}{ya}{wa}{ha}{opa}</image>')
        raise ValueError(f"unknown primitive kind {p.kind!r}")

    def text(self, p, ts):
        """one outlined <path> per distinct string; opacity switches between them; a moving anchor becomes a translate"""
        v, U = self.ctx.v, self.ctx.unit
        strs = [v(p.s, t) for t in ts]
        out = []
        for s_ in dict.fromkeys(strs):
            if not s_:
                continue
            idx = [i for i, q in enumerate(strs) if q == s_]
            pp = _TextAt(s_, p)
            if self.glyphs:
                body = self.glyph_text(pp, ts[idx[0]])
                if body is None:
                    continue
            else:
                gg = text_geom(pp, ts[idx[0]], v)
                if gg is None:
                    continue
                d = self.path_d(*gg)
            ops = [f2(float(v(p.op, t)) if q == s_ else 0.0) for t, q in zip(ts, strs)]
            col0, ca = self.attr(p.color, "fill", ts, hexc, True)
            anim_op = "" if len(set(ops)) == 1 else self.anim("opacity", ts, ops)
            move = ""
            if callable(p.xy):
                x_ref, y_ref = v(p.xy, ts[idx[0]])
                offs = [f"{(v(p.xy, t)[0] - x_ref) * U:.2f},{-(v(p.xy, t)[1] - y_ref) * U:.2f}" for t in ts]
                if len(set(offs)) > 1:
                    move = self.anim("transform", ts, offs).replace("<animate ", '<animateTransform type="translate" ', 1)
            if self.glyphs:
                out.append(f'<g fill="{col0}" opacity="{ops[0]}">{anim_op}{ca}{move}{body}</g>')
            else:
                out.append(f'<path d="{d}" fill="{col0}" opacity="{ops[0]}">{anim_op}{ca}{move}</path>')
        return "".join(out)

    def glyph_text(self, p, t):
        """a text frozen to one string as <use> references to glyphs defined once in <defs> (same geometry as TextPath)"""
        v, U = self.ctx.v, self.ctx.unit
        V_, _ = text_path(p.s, p.size, p.weight, p.style, p.family)
        if len(V_) == 0:
            return None
        xmin, xmax = V_[:, 0].min(), V_[:, 0].max()
        sh = {"left": 0.0, "center": -(xmin + xmax) / 2, "right": -xmax}[p.ha] * U          # in points
        info, gmap, rects = text_layout(p.s, p.weight, p.style, p.family)
        k = p.size / FONT_SCALE                                                              # points per font unit
        uses = []
        for g, gx, gy, gs in info:
            if g not in self.glyph_ids:
                gid = f"g{len(self.glyph_ids)}"
                self.glyph_ids[g] = gid
                self.defs.append(f'<path id="{gid}" d="{self.path_d(*gmap[g], P=lambda q: f"{q[0]:.1f},{q[1]:.1f}")}"/>')
            uses.append(f'<use xlink:href="#{self.glyph_ids[g]}" transform="translate({f2(sh + gx * k)},{f2(-gy * k)}) '
                        f'scale({gs * k:.4f},{-gs * k:.4f})"/>')
        for rv, rc in rects:
            uses.append(f'<path d="{self.path_d(rv, rc, P=lambda q: f"{sh + q[0] * k:.2f},{-q[1] * k:.2f}")}"/>')
        x, y = v(p.xy, t)
        rot = f" rotate({-p.rotation:g})" if p.rotation else ""
        return f'<g transform="translate({f2(self.X(x))},{f2(self.Y(y))}){rot}">{"".join(uses)}</g>'

    # ------------------------------------------------------------ document
    def document(self, prims, bg, title=""):
        ctx = self.ctx
        groups = {}
        for p in prims:
            groups.setdefault(p.group, []).append(p)
        body = []
        for g, ps in groups.items():
            gts = self.times(ctx.windows.get(g, (0.0, ctx.duration)))
            envs = [ctx.env(g, t) for t in gts]
            el = []
            for p in sorted(ps, key=lambda q: q.z):
                ts = self.times(p.win) if getattr(p, "win", None) else gts
                el.append(self.element(p, ts))
            if g not in ctx.envelopes:
                genv, go = "", "1"
            else:
                genv = "" if self.at is not None else self.anim("opacity", gts, [f2(e) for e in envs])
                go = f2(envs[0])
            body.append(f'<g id="{escape(str(g))}" opacity="{go}">{"".join(el)}{genv}</g>')
        W, H = _num(ctx.vb_w), _num(ctx.vb_h)
        if isinstance(bg, str):
            back = f'<rect width="{W}" height="{H}" fill="{bg}"/>'
        else:
            back = (f'<image x="0" y="0" width="{W}" height="{H}" preserveAspectRatio="none" '
                    f'xlink:href="data:image/png;base64,{png_b64(np.asarray(bg), colors=None)}"/>')
        sw, sh = ctx.svg_size
        defs = f'<defs>{"".join(self.defs)}</defs>' if self.defs else ""
        return (f'<?xml version="1.0" encoding="utf-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{sw}" height="{sh}">'
                f'<title>{escape(title)}</title>{defs}{back}' + "".join(body) + "</svg>")


class _TextAt:
    """a text primitive frozen to one string"""
    def __init__(self, s, p):
        self.s, self.xy, self.size, self.ha, self.weight, self.style = s, p.xy, p.size, p.ha, p.weight, p.style
        self.family, self.rotation = getattr(p, "family", None), getattr(p, "rotation", 0.0)


def export(prims, path, ctx, bg, title="", dt=0.1, simplify=False, at=None, glyphs=False):
    """Write an SVG of ``prims``.  at=None: animated (SMIL, sampled every ``dt`` s); at=t: static snapshot at display time t.
    simplify=True drops redundant samples (much smaller files, same animation); glyphs=True defines every glyph once and
    reuses it (much smaller text, identical geometry).  Returns the size in bytes."""
    svg = _Exporter(ctx, dt, simplify, at, glyphs).document(prims, bg, title)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return len(svg)
