def problem_p00079():
    import math

    f = []

    while True:

        try:

            st = input().strip().split(",")

            x, y = list(map(float, st))

            f.append(x + y * 1j)

        except EOFError:

            break

    px = [p.real for p in f]

    ox = (max(px) + min(px)) / 2.0

    py = [p.imag for p in f]

    oy = (max(py) + min(py)) / 2.0

    fo = ox + oy * 1j

    ss = 0.0

    for i in range(len(f)):

        if i == len(f) - 1:

            j = 0

        else:

            j = i + 1

        a = abs(f[i] - fo)

        b = abs(f[j] - fo)

        c = abs(f[j] - f[i])

        z = (a + b + c) / 2.0

        s = math.sqrt(z * (z - a) * (z - b) * (z - c))

        ss += s

    print(("%.6f" % ss))


problem_p00079()
