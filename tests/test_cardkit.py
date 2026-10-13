"""Run with `python -m unittest discover tests` (or pytest) from the cardkit folder."""
import os
import re
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))
import cardkit as ck                                         # noqa: E402
from cardkit import Txt, Path, Poly, Circ, Img, ramp, fade   # noqa: E402

SVG = "{http://www.w3.org/2000/svg}"


def smil_value(anim, t):
    """value of a SMIL <animate> at time t (numbers interpolated, discrete or non-numeric values stepped)"""
    D = float(anim.get("dur")[:-1])
    kt = [float(k) for k in anim.get("keyTimes").split(";")]
    vals = anim.get("values").split(";")
    u = (t % D) / D
    i = max(0, min(len(kt) - 1, int(np.searchsorted(kt, u, side="right")) - 1))
    if anim.get("calcMode") == "discrete":
        return vals[i]
    if i == len(kt) - 1:
        return np.array(re.split(r"[ ,]+", vals[i].strip()), float)
    a = np.array(re.split(r"[ ,]+", vals[i].strip()), float)
    b = np.array(re.split(r"[ ,]+", vals[i + 1].strip()), float)
    k = 0.0 if kt[i + 1] == kt[i] else (u - kt[i]) / (kt[i + 1] - kt[i])
    return a + (b - a) * k


def demo_scene():
    s = ck.Scene(6, background="#ffffff", intro=1.0, holds={2.5: 0.5})
    s.add(Txt("title", (0.5, 8), 20, "#000", op=1 - ramp(-0.5, -0.1)))
    with s.group("a", fade(0, 3)):
        s.add(Path([(1, 1), (5, 5), (8, 2)], "#0f9d8a", 3, draw=ramp(0.2, 1.5)))
        dot = ck.along([(1, 1), (5, 5), (8, 2)], 0.2, 1.5)
        s.add(Circ(dot, 0.15, fill=ck.keyframes({0.5: "#ff0000", 2.0: "#0000ff"}), stroke="#000", lw=1))
        s.add(Txt("moving", dot + np.array([0.2, 0.2]), 11, "#333"))
    with s.group("b", (3, 6)):
        s.add(Txt(lambda t: str(int(10 * ck.seg(t, 3.5, 5))), (4.5, 4.5), 30, "#000", "center"))
        s.add(Poly(ck.tween(ck.rect(1, 1, 2, 1.01), ck.rect(1, 1, 2, 3), 3.5, 4.5), fill="#6c5ce7"))
        s.add(Img(np.random.default_rng(0).random((8, 8, 4)), (5, 7, 1, 3), op=ramp(4, 4.5)))
    return s


class TestTiming(unittest.TestCase):
    def test_timeline_matches_card_v5_warp(self):
        INTRO, HOLD, HOLD_AT = 1.8, 0.9, [6.05, 11.85, 17.65, 24.65]

        def warp(T):                                    # verbatim from brainpfn_card_v5.py
            T -= INTRO; acc = 0.0
            for h in HOLD_AT:
                if T <= h + acc: return T - acc
                if T <= h + acc + HOLD: return h
                acc += HOLD
            return T - acc
        tl = ck.Timeline(25.0, intro=INTRO, holds=HOLD_AT, hold=HOLD)
        self.assertAlmostEqual(tl.display_duration, INTRO + 25 + 4 * HOLD)
        for T in np.linspace(0, tl.display_duration, 997):
            self.assertAlmostEqual(tl.warp(T), warp(T))
        for tau in (-1.0, 0.0, 3.0, 6.05, 12.0, 24.9):
            self.assertAlmostEqual(tl.warp(tl.to_display(tau)), tau)

    def test_fn_arithmetic_and_helpers(self):
        f = ramp(0, 1) * 0.5 + 0.25
        self.assertAlmostEqual(f(-1), 0.25); self.assertAlmostEqual(f(2), 0.75)
        g = 1 - fade(1, 3, 0.5)
        self.assertAlmostEqual(g(2), 0.0); self.assertAlmostEqual(g(0), 1.0)
        self.assertEqual(ck.keyframes({0: "#000000", 1: "#ffffff"})(5), "#ffffff")
        np.testing.assert_allclose(ck.tween([0, 0], [2, 4], 0, 1, eased=False)(0.5), [1, 2])
        np.testing.assert_allclose(ck.along([(0, 0), (2, 0)], 0, 1, eased=False)(0.25), [0.5, 0])
        self.assertEqual([ck.every(1, 0.5, 3)(t) for t in (0, 1.1, 1.6, 2.1, 2.6)], [0, 0, 1, 2, 0])


class TestScene(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.scene = demo_scene()

    def test_frame(self):
        im = self.scene.frame(1.5, width_px=360)
        self.assertEqual(im.size, (360, 360))
        self.assertEqual(im.getpixel((2, 2)), (255, 255, 255))
        wide = ck.Scene(2, size=(9, 11.25), origin=(0, -1), background="#101010")
        self.assertEqual(wide.frame(0, width_px=400).size, (400, 500))

    def test_svg_is_valid_and_simplify_is_lossless(self):
        full, small = os.path.join(self.tmp, "full.svg"), os.path.join(self.tmp, "small.svg")
        self.scene.save_svg(full, simplify=False)
        self.scene.save_svg(small, simplify=True)
        self.assertLess(os.path.getsize(small), os.path.getsize(full))
        A = [e for e in ET.parse(full).iter() if e.tag in (SVG + "animate", SVG + "animateTransform")]
        B = [e for e in ET.parse(small).iter() if e.tag in (SVG + "animate", SVG + "animateTransform")]
        self.assertEqual(len(A), len(B))
        for a, b in zip(A, B):
            for t in np.linspace(0, self.scene.duration, 233, endpoint=False):
                va, vb = smil_value(a, t), smil_value(b, t)
                if isinstance(va, str):
                    self.assertEqual(va, vb)
                else:
                    np.testing.assert_allclose(va, vb, atol=1e-6)

    def test_svg_features(self):
        p = os.path.join(self.tmp, "a.svg")
        self.scene.save_svg(p)
        root = ET.parse(p).getroot()
        self.assertEqual(root.get("viewBox"), "0 0 388.8 388.8")
        self.assertTrue(any(e.tag == SVG + "animateTransform" for e in root.iter()), "moving text should translate")
        circle = next(root.iter(SVG + "circle"))
        self.assertTrue(any(a.get("attributeName") == "cx" for a in circle))
        self.assertEqual({g.get("id") for g in root.iter(SVG + "g")}, {"ui", "a", "b"})
        snap = os.path.join(self.tmp, "snap.svg")
        self.scene.save(snap, at=2.0)
        self.assertFalse([e for e in ET.parse(snap).iter() if "animate" in e.tag], "static snapshot has no animation")

    def test_varying_point_count_is_discrete(self):
        s = ck.Scene(2)
        s.add(Poly(lambda t: ck.rect(1, 1, 2, 2) if t < 1 else ck.ngon((4, 4), 1, 6), fill="#000"))
        p = os.path.join(self.tmp, "v.svg"); s.save_svg(p)
        anim = next(e for e in ET.parse(p).iter(SVG + "animate") if e.get("attributeName") == "points")
        self.assertEqual(anim.get("calcMode"), "discrete")

    def test_raster_exports(self):
        s = ck.Scene(1, fps=10)
        s.add(Circ((4.5, 4.5), ramp(0, 1) * 2 + 0.5, fill="#0f9d8a"))
        out = s.save_all(os.path.join(self.tmp, "c"), "gif", "webp", poster=0.5, width_px=180, gif_width=120, workers=1,
                         progress=False)
        for k in ("gif", "webp", "poster"):
            self.assertTrue(os.path.getsize(out[k]) > 0)

    def test_check_text(self):
        s = ck.Scene(1)
        s.add(Txt("overlap one", (1, 4), 20), Txt("overlap two", (1.2, 4.05), 20), Txt("edge", (8.9, 1), 12))
        issues = s.check_text(0.5, verbose=False)
        self.assertTrue(any(i.startswith("OVERLAP") for i in issues))
        self.assertTrue(any("edge" in i for i in issues))

    def test_group_context_and_tuple_envelope(self):
        s = ck.Scene(4)
        with s.group("g", (1, 3)) as g:
            p = s.add(Circ((1, 1), 0.1))
        self.assertEqual(p.group, "g"); self.assertEqual(g, "g")
        self.assertAlmostEqual(s.envelopes["g"](2.0), 1.0)
        self.assertEqual(s.add(Circ((1, 1), 0.1)).group, ck.BASE)


class TestLegacyAPI(unittest.TestCase):
    def test_module_level_globals(self):
        ck.DUR = 4.0; ck.WARP = lambda t: t; ck.GROUP_ENV.clear(); ck.GROUP_WIN.clear()
        ck.GROUP_ENV["f1"] = fade(0, 4)
        prims = [Txt("hi", (1, 1), 12, op=lambda t: ck.seg(t, 0, 1), group="f1"), Circ((2, 2), 0.5, fill="#000")]
        p = os.path.join(tempfile.mkdtemp(), "legacy.svg")
        ck.export_svg(prims, p, "#ffffff", "legacy")
        root = ET.parse(p).getroot()
        self.assertEqual(root.get("width"), "900")
        self.assertIsNotNone(ck.text_bbox(prims[0], 0))
        ck.GROUP_ENV.clear()


if __name__ == "__main__":
    unittest.main()
