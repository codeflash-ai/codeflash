def problem_p00060():
    import sys

    for s in sys.stdin:

        C = [0] + [1] * 10

        a, b, c = list(map(int, s.split()))

        C[a] = 0

        C[b] = 0

        C[c] = 0

        if sum(C[: 21 - a - b]) >= 4:
            print("YES")

        else:
            print("NO")


problem_p00060()
