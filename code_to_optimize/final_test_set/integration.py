def f(x):
    return x * (x - 1)


def integrate_f(a, b, N):
    dx = (b - a) / N
    s = 0
    adx = a
    for i in range(N):
        s += adx * (adx - 1)
        adx += dx
    return s * dx
