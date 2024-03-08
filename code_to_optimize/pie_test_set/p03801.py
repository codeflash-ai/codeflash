def problem_p03801():
    n = int(input())

    a = list(map(int, input().split()))

    tmp = [[e, n - i] for i, e in enumerate(a)]

    tmp.sort(reverse=True)

    aa = [[e, n - i] for e, i in tmp] + [[0, -1]]

    v_prev, i_prev = aa[0]

    i = 0

    ans = [0] * n

    sm = 0

    while i < n:

        while aa[i][1] >= i_prev:

            sm += aa[i][0]

            i += 1

        ans[i_prev] += sm - aa[i][0] * i

        sm = aa[i][0] * i

        v_prev, i_prev = aa[i]

    print(*ans, sep="\n")


problem_p03801()
