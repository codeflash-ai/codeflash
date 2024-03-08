def problem_p03839():
    N, K = list(map(int, input().split()))

    src = list(map(int, input().split()))

    cums = [0]

    cump = [0]

    for a in src:

        cums.append(cums[-1] + a)

        cump.append(cump[-1] + max(0, a))

    ans = 0

    for i in range(N - K + 1):

        tmp = cump[i]

        tmp += max(0, cums[i + K] - cums[i])

        tmp += cump[N] - cump[i + K]

        ans = max(tmp, ans)

    print(ans)


problem_p03839()
