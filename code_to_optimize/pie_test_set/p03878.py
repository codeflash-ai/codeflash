def problem_p03878():
    import sys

    read = sys.stdin.read

    N, *ab = list(map(int, read().split()))

    mod = 10**9 + 7

    ab = list(zip(ab[:N], [0] * N)) + list(zip(ab[N:], [1] * N))

    ab.sort()

    remain_pc = 0

    remain_power = 0

    answer = 1

    for x, p in ab:

        if p == 0:

            if remain_power == 0:

                remain_pc += 1

                continue

            answer *= remain_power

            remain_power -= 1

        else:

            if remain_pc == 0:

                remain_power += 1

                continue

            answer *= remain_pc

            remain_pc -= 1

        answer %= mod

    print(answer)


problem_p03878()
