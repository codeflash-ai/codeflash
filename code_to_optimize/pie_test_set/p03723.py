def problem_p03723():
    a, b, c = list(map(int, input().split()))

    e = (a - b) | (b - c)

    print((bool(e | (a | b | c) % 2) * (e ^ ~-e).bit_length() - 1))


problem_p03723()
