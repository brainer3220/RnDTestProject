import math


def v_add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def v_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def v_norm(a):
    return math.sqrt(v_dot(a, a))


def v_unit(a, eps=1e-9):
    n = v_norm(a)
    if n < eps:
        return [0.0, 0.0, 0.0]
    return [a[0] / n, a[1] / n, a[2] / n]


def v_scale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def angle_between(a, b, eps=1e-9):
    na = v_norm(a)
    nb = v_norm(b)
    if na < eps or nb < eps:
        return 0.0
    cos_val = max(-1.0, min(1.0, v_dot(a, b) / (na * nb)))
    return math.degrees(math.acos(cos_val))


def signed_angle(a, b, normal, eps=1e-9):
    na = v_norm(a)
    nb = v_norm(b)
    if na < eps or nb < eps:
        return 0.0
    cross = v_cross(a, b)
    sin_val = v_dot(normal, cross) / (na * nb)
    cos_val = v_dot(a, b) / (na * nb)
    sin_val = max(-1.0, min(1.0, sin_val))
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.atan2(sin_val, cos_val))


def project_to_plane(v, normal):
    # Remove component along normal.
    n_unit = v_unit(normal)
    return v_sub(v, v_scale(n_unit, v_dot(v, n_unit)))
