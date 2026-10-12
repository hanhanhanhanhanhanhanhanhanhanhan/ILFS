import math

from .constants import ANIONS, RADII, THIO_FACTOR


def anion_radius(n):
    nx = sum(n.get(a, 0.0) for a in ANIONS)
    return sum(n.get(a, 0.0) * RADII[a] for a in ANIONS) / nx


def v_perfect(family, r_x):
    d_oct = RADII["Li"] + r_x
    if family == "LGPS-type":
        a = 2 * d_oct + 2 * math.sqrt(6) / 3 * (RADII["P"] + r_x)
        c = 2 * (math.sqrt(2) + 1) * d_oct
        return a * a * c
    return THIO_FACTOR * 4 / 3 * d_oct ** 3
