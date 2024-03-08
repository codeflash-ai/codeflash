def problem_p04003():
    import sys

    from collections import deque, defaultdict

    def bfs01(s, t, links):

        q = deque([(0, s, -1)])  # cost, station, last-company

        visited = set()

        while q:

            d, v, e = q.popleft()

            if v == t:

                return d

            if (v, e) in visited:

                continue

            visited.add((v, e))

            if e == 0:

                lv = links[v]

                for c in lv:

                    for u in lv[c]:

                        if (u, c) in visited:

                            continue

                        q.append((d + 1, u, c))

            else:

                for u in links[v][e]:

                    if (u, e) in visited:

                        continue

                    q.appendleft((d, u, e))

                if (v, 0) not in visited:

                    q.appendleft((d, v, 0))

        return -1

    readline = sys.stdin.buffer.readline

    read = sys.stdin.buffer.read

    n, m = list(map(int, readline().split()))

    links = [defaultdict(set) for _ in range(n)]

    pqc = list(map(int, read().split()))

    for p, q, c in zip(pqc[0::3], pqc[1::3], pqc[2::3]):

        p -= 1

        q -= 1

        links[p][c].add(q)

        links[q][c].add(p)

    print((bfs01(0, n - 1, links)))


problem_p04003()
