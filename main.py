import itertools

import cairo
import numpy as np
from PIL import Image
from tqdm import tqdm

TAU = 2 * np.pi
BACKGROUND = 1
FOREGROUND = 0

def clone_surface(surface: cairo.ImageSurface) -> cairo.ImageSurface:
    return cairo.ImageSurface.create_for_data(
        memoryview(bytearray(surface.get_data())),
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


def cost(node_pair, accumulator: cairo.ImageSurface, target: np.ndarray) -> float:
    n0, n1 = node_pair
    surface = clone_surface(accumulator)
    ctx = cairo.Context(surface)
    ctx.set_line_width(2)
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.set_source_rgba(0, 0, 0, FOREGROUND)
    ctx.move_to(*n0)
    ctx.line_to(*n1)
    ctx.stroke()
    diff = asarray(surface).astype(np.float32) - target
    return np.sum(diff.ravel()**2)


def load_target(path: str) -> np.ndarray:
    img = Image.open(path)
    target = np.asarray(img).astype(np.float32)
    if len(target.shape) == 3:
        return np.mean(target, axis=-1)
    return target


def size(img: np.ndarray) -> np.ndarray:
    return np.array([img.shape[1], img.shape[0]])


def main():
    n = 64
    target = load_target('skull.jpeg')

    radius = np.max(size(target))
    alpha = np.linspace(0, TAU, n, endpoint=False)
    nodes = radius * np.vstack([np.cos(alpha), np.sin(alpha)]).T + 0.5 * size(target)

    accumulator = cairo.ImageSurface(cairo.FORMAT_A8, *size(target))
    ctx = cairo.Context(accumulator)
    ctx.set_source_rgba(1, 1, 1, BACKGROUND)
    ctx.rectangle(0, 0, accumulator.get_width(), accumulator.get_height())
    ctx.fill()

    ctx.set_line_width(2)
    ctx.set_source_rgba(0, 0, 0, FOREGROUND)
    ctx.set_operator(cairo.OPERATOR_SOURCE)

    combinations = list(itertools.combinations(nodes, 2))

    for _ in tqdm(range(100)):
        index = np.argmin([cost(node_pair, accumulator, target) for node_pair in combinations])

        n0, n1 = combinations[index]
        ctx.move_to(*n0)
        ctx.line_to(*n1)
        ctx.stroke()

        del combinations[index]

    accumulator.write_to_png('output.png')


if __name__ == "__main__":
    main()
