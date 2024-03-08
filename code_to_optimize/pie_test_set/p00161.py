def problem_p00161():
    while True:

        n = int(input())

        if n == 0:

            break

        R = sorted(
            [list(map(int, input().split())) for _ in range(n)],
            key=lambda x: sum(60 * m + s for m, s in zip(x[1::2], x[2::2])),
        )

        print("\n".join(map(str, [R[0][0], R[1][0], R[-2][0]])))


problem_p00161()
