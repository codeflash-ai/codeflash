def problem_p03536():
    N = int(eval(input()))

    HPs = [tuple(map(int, input().split())) for i in range(N)]

    HPs.sort(key=lambda x: x[0] + x[1])

    INF = float("inf")

    dp = [0]

    for H, P in HPs:

        if dp[-1] != INF:

            dp += [INF]

        dp = list(map(min, list(zip([0] + [hgt + P if hgt <= H else INF for hgt in dp], dp))))

    dp += [INF]

    print((dp.index(INF) - 1))


problem_p03536()
