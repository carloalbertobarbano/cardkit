"""Regenerate the README demos: python docs/make_docs.py  (needs nilearn for the brain example)."""
import os
import sys
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, os.pardir, "src"), os.path.join(HERE, os.pardir, "examples")]

for name in sys.argv[1:] or ["quickstart", "tour", "plots", "brain", "network", "three_d"]:
    scene = importlib.import_module(name).scene
    scene.save_all(os.path.join(HERE, name), "gif", "svg", gif_width=480, progress=False)
    if name == "tour":
        scene.storyboard(os.path.join(HERE, "tour_storyboard.png"), labels=True)
    print("built", name)
