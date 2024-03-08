def problem_p03778():
    W, a, b = list(map(int, input().split()))

    s = {i for i in range(a, a + W + 1)} & {i for i in range(b, b + W + 1)}

    if s:

        print((0))

    else:

        print((min(abs(b - a - W), abs(b + W - a))))


problem_p03778()
