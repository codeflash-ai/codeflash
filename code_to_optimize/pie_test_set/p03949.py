def problem_p03949():
    import sys

    from collections import defaultdict as dd

    from collections import deque as dq

    import heapq

    hpush = heapq.heappush

    hpop = heapq.heappop

    input = sys.stdin.readline

    N = int(eval(input()))

    e = dd(list)

    for _ in range(N - 1):

        u, v = list(map(int, input().split()))

        e[u].append(v)

        e[v].append(u)

    K = int(eval(input()))

    ls = [-(10**10)] * (N + 1)

    rs = [10**10] * (N + 1)

    Q = dq([])

    h = []

    res = [-(10**10)] * (N + 1)

    for _ in range(K):

        x, v = list(map(int, input().split()))

        ls[x] = v

        rs[x] = v

        hpush(h, (v, x))

        res[x] = v

        Q.append(x)

    vis = [0] * (N + 1)

    while len(Q):

        x = Q.popleft()

        if vis[x]:
            continue

        vis[x] = 1

        for y in e[x]:

            ls[y] = max(ls[y], ls[x] - 1)

            rs[y] = min(rs[y], rs[x] + 1)

            if ls[y] % 2 == ls[x] % 2:

                print("No")

                exit(0)

            if vis[y]:
                continue

            Q.append(y)

    for x in range(1, N + 1):

        if ls[x] > rs[x]:

            print("No")

            exit(0)

    while len(h):

        _, x = hpop(h)

        for y in e[x]:

            if res[y] != -(10**10):
                continue

            res[y] = res[x] + 1

            hpush(h, (res[y], y))

        # print(res, h)

    print("Yes")

    for r in res[1:]:
        print(r)


problem_p03949()
