"""Frame encoders: GIF and WebP with Pillow, MP4 with ffmpeg."""
import os
import shutil
import subprocess

from PIL import Image


def _rgb(path, size=None):
    with Image.open(path) as im:
        im = im.convert("RGB")
    return im if size is None or im.size == size else im.resize(size, Image.LANCZOS)


def write_gif(paths, out, fps, gif_fps=15, width=720, palette_samples=8):
    """GIF from frame files rendered at ``fps``: resampled to ``gif_fps``, ``width`` px wide, one shared 256-colour
    palette (no dithering: crisp flat colours).  Frame delays alternate so the total length stays exact
    (GIF delays are whole centiseconds)."""
    step = fps / gif_fps
    idx = []
    k = 0.0
    while int(round(k)) < len(paths):
        idx.append(int(round(k)))
        k += step
    with Image.open(paths[0]) as first:
        size = (width, round(width * first.height / first.width))
    load = lambda i: _rgb(paths[i], size)
    picks = idx[len(idx) // (palette_samples + 1)::max(1, len(idx) // (palette_samples + 1))][:palette_samples] or idx[:1]
    sample = Image.new("RGB", (size[0], size[1] * len(picks)))
    for j, i in enumerate(picks):
        sample.paste(load(i), (0, j * size[1]))
    pal = sample.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    cs = [round(j * 100 / gif_fps) for j in range(len(idx) + 1)]
    durations = [10 * (b - a) for a, b in zip(cs, cs[1:])]
    frames = (load(i).quantize(palette=pal, dither=Image.Dither.NONE) for i in idx[1:])
    load(idx[0]).quantize(palette=pal, dither=Image.Dither.NONE).save(
        out, save_all=True, append_images=frames, duration=durations, loop=0, optimize=False)
    return out


def write_webp(paths, out, fps, quality=90):
    """Animated WebP at full resolution (lossy, ``quality`` 0-100)."""
    frames = (_rgb(p) for p in paths[1:])
    _rgb(paths[0]).save(out, save_all=True, append_images=frames, duration=int(round(1000 / fps)), loop=0,
                        quality=quality, method=4)
    return out


def ffmpeg_exe():
    """ffmpeg from $FFMPEG, the PATH, or the imageio-ffmpeg package."""
    for c in (os.environ.get("FFMPEG"), shutil.which("ffmpeg")):
        if c:
            return c
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("MP4 export needs ffmpeg: install it (brew install ffmpeg), set $FFMPEG, "
                           "or `pip install imageio-ffmpeg`") from None


def write_mp4(folder, out, fps, crf=16):
    """H.264 MP4 (yuv420p, faststart: plays everywhere, including LinkedIn / X / Slack) from folder/%05d.png."""
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(folder, "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart", out], check=True)
    return out
