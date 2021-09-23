import itertools

import cairo
import numpy as np
from PIL import Image


TAU = 2 * np.pi


def clone_surface(surface: cairo.ImageSurface) -> cairo.ImageSurface:
    return cairo.ImageSurface.create_for_data(
        surface.get_data(),
        surface.get_format(),
        surface.get_width(),
        surface.get_height(),
        surface.get_stride())


def asarray(surface: cairo.ImageSurface) -> np.ndarray:
    buffer = surface.get_data()
    return np.ndarray(
        shape=(surface.get_height(), surface.get_width()),
        dtype=np.uint8,
        buffer=buffer)


def cost(diff: np.ndarray) -> float:
    return np.sum(diff.ravel()**2)


def main():
    n = 6
    img = Image.open('woman.jpeg')
    size = np.array([img.width, img.height])
    target = np.asarray(img).astype(np.float32)

    r = np.max(size)
    alpha = np.linspace(0, TAU, n, endpoint=False)
    nodes = r * np.vstack([np.cos(alpha), np.sin(alpha)]).T + 0.5 * size

    accumulator = cairo.ImageSurface(cairo.FORMAT_A8, *size)
    for n0, n1 in itertools.combinations(nodes, 2):
        surface = clone_surface(accumulator)
        ctx = cairo.Context(surface)
        ctx.set_line_width(2)
        ctx.move_to(*n0)
        ctx.line_to(*n1)
        ctx.stroke()
        diff = asarray(surface).astype(np.float32) - target
        print(cost(diff))


if __name__ == "__main__":
    main()
