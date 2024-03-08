def problem_p03845():
    n, t = int(input()), list(map(int, input().split()))
    T = sum(t)

    print(
        *[
            T - t[i - 1] + j
            for i, j in [list(map(int, input().split())) for _ in range(int(input()))]
        ],
        sep="\n",
    )


problem_p03845()
