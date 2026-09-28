import math


def band(x0, y0, x1, y1, thickness):
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    px, py = -dy / length * thickness / 2, dx / length * thickness / 2
    points = [
        (x0 + px, y0 + py),
        (x1 + px, y1 + py),
        (x1 - px, y1 - py),
        (x0 - px, y0 - py),
    ]
    return "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points) + " Z"


# M: width 88, cap 100, stroke 18. Diagonals meet at the center.
print("left", band(18, 2, 44, 68, 18))
print("right", band(70, 2, 44, 68, 18))
