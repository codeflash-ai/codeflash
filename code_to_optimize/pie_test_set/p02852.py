def problem_p02852():
    from collections import deque

    INF = 10**18

    def append_dq(i, x, dq, dp1):

        while True:

            if not len(dq):

                dq.append(i)

                break

            lxi = dq[-1]

            if x > dp1[lxi]:

                dq.append(i)

                break

            dq.pop()

        return dq

    dq = deque()

    N, M = list(map(int, input().split()))

    S = input().strip()

    dp1 = [0] + [INF] * N

    dp2 = [-1 for _ in range(N + 1)]

    dq.append(0)

    for i, c in enumerate(S[1:]):

        i += 1

        dq = append_dq(i, dp1[i], dq, dp1)

        idx = dq[0]

        if c != "1" and dp1[idx] != INF:

            dp1[i] = dp1[idx] + 1

            dp2[i] = idx

        if i >= M:

            sxi = dq[0]

            if sxi == (i - M):

                dq.popleft()

    if dp1[-1] == INF:

        print((-1))

    else:

        r = []

        s = N

        while True:

            if s == 0:

                break

            ns = dp2[s]

            r.append(s - ns)

            s = ns

        print((" ".join(map(str, r[::-1]))))


problem_p02852()
