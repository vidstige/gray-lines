import itertools

import cairo
import numpy as np
from PIL import Image


TAU = 2 * np.pi


def asarray(surface: cairo.ImageSurface) -> np.ndarray:
    buffer = surface.get_data()
    return np.ndarray(
        shape=(surface.get_width(),surface.get_height(), 4),
        dtype=np.uint8,
        buffer=buffer)


def main():
    n = 6
    img = Image.open('woman.jpeg')
    size = np.array([img.width, img.height])
    r = np.max(size)
    alpha = np.linspace(0, TAU, n, endpoint=False)
    nodes = r * np.vstack([np.cos(alpha), np.sin(alpha)]).T + 0.5 * size

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, img.width, img.height)
    ctx = cairo.Context(surface)
    for n0, n1 in itertools.combinations(nodes, 2):
        ctx.move_to(*n0)
        ctx.line_to(*n1)
        ctx.stroke()
    surface.write_to_png("output.png")

if __name__ == "__main__":
    main()
