"""Scene: holds the primitives, the clock and the groups, and exports everything."""
import os
import tempfile
import shutil
import multiprocessing as mp

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .primitives import Prim, BASE
from .text import U, text_bbox
from .timing import Timeline, fade
from .context import Context
from . import render as _render
from . import svg as _svg
from . import export as _export


class Group(str):
    """A group name (it *is* a str).  Use as a context manager to put every primitive added inside it in the group:

        with scene.group("step1", fade(0, 6)):
            scene.add(Txt("hello", (1, 8)))
    """

    def __new__(cls, name, scene):
        g = super().__new__(cls, name)
        g._scene = scene
        return g

    def __enter__(self):
        self._scene._stack.append(str(self))
        return self

    def __exit__(self, *exc):
        self._scene._stack.pop()


class Scene:
    """An animated card.

    Scene(duration, size=(9, 9), background="#ffffff")

    duration    length of the story in scene seconds (what you time your elements in)
    size        (width, height) of the visible area in slide units; y points up, (0, 0) is the bottom-left corner
    origin      slide coordinates of the bottom-left corner (default (0, 0))
    background  colour, or an RGB(A) image / array stretched over the whole frame
    width_px    raster width in pixels (default 120 px per unit, i.e. 1080 px for a 9-unit card)
    fps         frames per second for the raster exports
    intro       extra scene time before 0, played first (author the intro at negative times)
    holds       {scene_time: seconds} pauses inserted into playback (or a list of scene times, ``hold`` s each)
    title       accessible title written into the SVG

    Units: 1 slide unit = 43.2 pt, so font sizes and line widths (in points) look the same at any output resolution.
    """

    def __init__(self, duration, size=(9, 9), background="#ffffff", *, origin=(0.0, 0.0), width_px=None, fps=30,
                 intro=0.0, holds=None, hold=0.9, title=""):
        self.timeline = Timeline(duration, intro, holds, hold)
        self.size, self.origin = tuple(size), tuple(origin)
        self.background, self.fps, self.title = background, fps, title
        self.width_px = width_px or int(round(120 * size[0]))
        self.prims, self.envelopes, self._stack = [], {}, []

    # ------------------------------------------------------------ building
    @property
    def duration(self):
        """display duration of one loop (s): intro + duration + holds"""
        return self.timeline.display_duration

    @property
    def view(self):
        (x0, y0), (w, h) = self.origin, self.size
        return (x0, x0 + w, y0, y0 + h)

    def group(self, name, envelope=None):
        """Register a group with an opacity ``envelope`` (a function of scene time, or a (t0, t1) tuple meaning
        ``fade(t0, t1)``).  Returns a :class:`Group` usable as a context manager or as ``group=`` argument."""
        if isinstance(envelope, tuple):
            envelope = fade(*envelope)
        if envelope is not None:
            self.envelopes[str(name)] = envelope
        return Group(str(name), self)

    def axes(self, rect, xlim=(0.0, 1.0), ylim=(0.0, 1.0), **kw):
        """Minimal scientific axes at ``rect`` = (x, y, width, height) in slide units, added to the scene (see
        :class:`cardkit.plot.Axes`).  Plot with ``ax.line``, ``ax.scatter``, ``ax.fill_between``, ``ax.bar`` ..."""
        from .plot import Axes
        return Axes(rect, xlim, ylim, scene=self, **kw)

    def add(self, *items):
        """Add primitives (or lists of primitives).  Returns the single primitive, or the list."""
        flat = []
        for it in items:
            if isinstance(it, Prim):
                flat.append(it)
            else:
                flat.extend(it)
        for p in flat:
            if self._stack and p.group == BASE:
                p.group = self._stack[-1]
            self.prims.append(p)
        return flat[0] if len(flat) == 1 else flat

    def to_display(self, tau):
        """scene time -> display time"""
        return self.timeline.to_display(tau)

    def context(self, windows=None, svg_width=None):
        x0, x1, y0, y1 = self.view
        sw = svg_width or round(100 * (x1 - x0))
        return Context(self.duration, self.timeline.warp, self.envelopes, windows, self.view, U,
                       svg_size=(sw, round(sw * (y1 - y0) / (x1 - x0))))

    # ------------------------------------------------------------ raster
    def frame(self, t, width_px=None, isolate=True):
        """RGB PIL image at display time ``t`` (in a notebook it displays inline)."""
        return _render.frame(self.prims, t, self.context(), self.background, width_px or self.width_px, isolate)

    def frame_at(self, tau, **kw):
        """RGB PIL image at *scene* time ``tau``."""
        return self.frame(self.to_display(tau), **kw)

    def n_frames(self, fps=None):
        return int(round(self.duration * (fps or self.fps)))

    def render_frames(self, folder, fps=None, width_px=None, workers=None, progress=True):
        """Render every frame to ``folder/00000.png ...``.  Returns the list of paths.
        workers: processes to use (default: all cores but one; 1 = serial)."""
        fps = fps or self.fps
        n = self.n_frames(fps)
        os.makedirs(folder, exist_ok=True)
        jobs = [(i, i / fps, os.path.join(folder, f"{i:05d}.png"), width_px) for i in range(n)]
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        if workers > 1 and "fork" in mp.get_all_start_methods():
            global _ACTIVE
            _ACTIVE = self                                            # inherited by the forked workers (lambdas cannot be pickled)
            try:
                with mp.get_context("fork").Pool(workers) as pool:
                    for k, _ in enumerate(pool.imap_unordered(_work, jobs, chunksize=4)):
                        if progress and k % 60 == 0:
                            print(f"  frame {k} / {n}", flush=True)
            finally:
                _ACTIVE = None
        else:
            for i, t, path, w in jobs:
                self.frame(t, w).save(path, compress_level=1)
                if progress and i % 60 == 0:
                    print(f"  frame {i} / {n}", flush=True)
        return [j[2] for j in jobs]

    # ------------------------------------------------------------ export
    def save_svg(self, path, at=None, dt=0.1, simplify=True, width=None, glyphs=True):
        """Animated SVG (SMIL, loops, plays in <img>); ``at=t`` writes a static snapshot at display time t instead.
        glyphs=True defines each glyph once and reuses it (much smaller files with lots of text or counters).
        Returns the file size in bytes."""
        windows = {g: self._window(g, dt) for g in self.envelopes}
        return _svg.export(self.prims, path, self.context(windows, width), self.background, self.title, dt, simplify, at,
                           glyphs)

    def save(self, path, *, at=None, fps=None, width_px=None, workers=None, **kw):
        """Save by extension: .svg (animated, or static with ``at``), .png / .jpg (frame at display time ``at``),
        .gif / .webp / .mp4 (renders all frames)."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".svg":
            return self.save_svg(path, at=at, **kw)
        if ext in (".png", ".jpg", ".jpeg"):
            self.frame(self.duration / 2 if at is None else at, width_px).save(path, **kw)
            return path
        return self.save_all(os.path.splitext(path)[0], ext[1:], fps=fps, width_px=width_px, workers=workers, **kw)

    def save_all(self, stem, *formats, poster=None, fps=None, width_px=None, workers=None, gif_fps=15, gif_width=720,
                 webp_quality=90, crf=16, progress=True):
        """Render the frames once and write several formats: ``scene.save_all("out/card", "gif", "webp", "mp4", "svg",
        poster=23.6)`` -> out/card.gif, .webp, .mp4, _animated.svg (the SVG name keeps the .svg free for a static
        snapshot) and _poster.png (``poster`` is a *scene* time).  Returns {format: path}."""
        formats = [f.lower().lstrip(".") for f in (formats or ("gif", "webp", "svg"))]
        fps = fps or self.fps
        out, raster = {}, [f for f in formats if f in ("gif", "webp", "mp4")]
        if os.path.dirname(stem):
            os.makedirs(os.path.dirname(stem), exist_ok=True)
        if "svg" in formats:
            out["svg"] = stem + "_animated.svg" if len(formats) > 1 else stem + ".svg"
            size = self.save_svg(out["svg"])
            progress and print(f"saved {out['svg']} ({size / 1e3:.0f} kB)")
        if poster is not None:
            out["poster"] = stem + "_poster.png"
            self.frame_at(poster, width_px=width_px).save(out["poster"])
        if raster:
            tmp = tempfile.mkdtemp(prefix="cardkit_frames_")
            try:
                paths = self.render_frames(tmp, fps, width_px, workers, progress)
                for f in raster:
                    out[f] = f"{stem}.{f}"
                    if f == "gif":
                        _export.write_gif(paths, out[f], fps, gif_fps, gif_width)
                    elif f == "webp":
                        _export.write_webp(paths, out[f], fps, webp_quality)
                    else:
                        _export.write_mp4(tmp, out[f], fps, crf)
                    progress and print(f"saved {out[f]} ({os.path.getsize(out[f]) / 1e6:.1f} MB)")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        return out

    def storyboard_keys(self, n=8):
        """Default key frames as [(label, scene_time)]: for every group, the last instant it is fully visible (the end of
        its reading pause, when everything in it has appeared); for every stretch where no group shows (an intro, an
        outro on the base layer), its midpoint.  Without groups: ``n`` evenly spaced instants."""
        D, dt = self.duration, 1.0 / self.fps
        ts = np.arange(0.0, D, dt)
        if not self.envelopes:
            return [(f"{self.timeline.warp(t):.1f}s", self.timeline.warp(t))
                    for t in np.linspace(0, D, n, endpoint=False) + D / n / 2]
        ctx = self.context()
        env = {g: np.array([ctx.env(g, t) for t in ts]) for g in self.envelopes}
        keys = []
        for g, e in env.items():
            full = np.flatnonzero(e >= 0.995)
            if len(full) == 0:
                full = [int(np.argmax(e))]
            keys.append((ts[full[-1]], str(g)))
        none = np.all([e <= 0.004 for e in env.values()], axis=0)              # nothing but the base layer on screen
        k = 0
        while k < len(ts):
            if none[k]:
                j = k
                while j + 1 < len(ts) and none[j + 1]:
                    j += 1
                if ts[j] - ts[k] >= 0.5:
                    keys.append((ts[(k + j) // 2], "intro" if k == 0 else "outro" if j == len(ts) - 1 else f"base {ts[k]:.0f}s"))
                k = j + 1
            else:
                k += 1
        merged = []
        for t, label in sorted(keys):                                         # groups ending together -> one frame
            if merged and t - merged[-1][0] < 0.25:
                merged[-1] = (t, merged[-1][1] + "+" + label)
            else:
                merged.append((t, label))
        return [(label, self.timeline.warp(t)) for t, label in merged]

    def storyboard(self, path=None, keys=None, *, cols=5, width=432, gap=12, background="#8a93a3", labels=False,
                   check=False, save_frames=None, n=8):
        """Contact sheet of key frames, each rendered at full resolution and downsampled to ``width`` px.

        keys         scene times or (name, scene_time) pairs; default :meth:`storyboard_keys` (one finished frame per group)
        labels       write each key's name under its tile
        check        run :meth:`check_text` on every key frame
        save_frames  also save each frame at full resolution as ``f"{save_frames}{name}.png"`` (e.g. ``"out/key_"``)

        Saves to ``path`` if given; returns the PIL image."""
        keys = self.storyboard_keys(n) if keys is None else [k if isinstance(k, tuple) else (f"{k:g}s", k) for k in keys]
        tiles = []
        for name, tau in keys:
            t = self.to_display(tau)
            if check:
                print(f"{name:>10}", end=" ")
                self.check_text(t)
            im = self.frame(t)
            if save_frames:
                im.save(f"{save_frames}{name}.png")
            tiles.append(im.resize((width, round(width * im.height / im.width)), Image.LANCZOS))
        w, h = tiles[0].size
        cols = min(cols, len(tiles))
        rows = -(-len(tiles) // cols)
        lab = 26 if labels else 0
        try:
            font = ImageFont.load_default(size=15)
        except TypeError:                                                     # Pillow < 10.1
            font = ImageFont.load_default()
        sheet = Image.new("RGB", (cols * (w + gap) + gap, rows * (h + gap + lab) + gap), background)
        d = ImageDraw.Draw(sheet)
        for k, ((name, tau), im) in enumerate(zip(keys, tiles)):
            x, y = gap + (k % cols) * (w + gap), gap + (k // cols) * (h + gap + lab)
            sheet.paste(im, (x, y))
            if labels:
                d.text((x, y + h + 5), f"{name}   ·   t = {tau:.2f} s", fill="#ffffff", font=font)
        if path:
            sheet.save(path)
        return sheet

    # ------------------------------------------------------------ QA
    def check_text(self, t, margin=0.25, min_alpha=0.2, verbose=True):
        """Report texts visible at display time ``t`` that overlap each other or come closer than ``margin`` to the edge.
        Returns the list of issues (empty = all good)."""
        ctx = self.context()
        items = []
        for p in self.prims:
            if p.kind != "txt" or ctx.env(p.group, t) * ctx.v(p.op, t) < min_alpha:
                continue
            bb = text_bbox(p, t, ctx.v)
            if bb:
                items.append((ctx.v(p.s, t), *bb))
        X0, X1, Y0, Y1 = self.view
        bad = [f"MARGIN  {s!r}" for s, x0, x1, y0, y1 in items
               if x0 < X0 + margin or x1 > X1 - margin or y0 < Y0 + margin or y1 > Y1 - margin]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if min(a[2], b[2]) - max(a[1], b[1]) > 0.005 and min(a[4], b[4]) - max(a[3], b[3]) > 0.005:
                    bad.append(f"OVERLAP {a[0]!r} <-> {b[0]!r}")
        if verbose:
            print(f"[check t={t:.2f}] {len(items)} texts, {len(bad)} issues")
            for b in bad:
                print("   ", b)
        return bad

    # ------------------------------------------------------------ internals
    def _window(self, g, dt):
        """display-time span where group g is visible (+ one sample of margin); the SVG samples it only there"""
        D = self.duration
        ts = np.linspace(0, D, max(2, int(round(D / dt)) + 1))
        ctx = self.context()
        on = np.flatnonzero([ctx.env(g, t) > 0.004 for t in ts])
        if len(on) == 0:
            return (0.0, D)
        return (float(max(0.0, ts[on[0]] - dt)), float(min(D, ts[on[-1]] + dt)))

    def __repr__(self):
        return (f"<Scene {self.size[0]}x{self.size[1]} units, {self.duration:.2f} s, {len(self.prims)} primitives, "
                f"groups={list(self.envelopes)}>")


_ACTIVE = None


def _work(job):
    i, t, path, w = job
    _ACTIVE.frame(t, w).save(path, compress_level=1)
    return i
