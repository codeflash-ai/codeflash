def problem_p02328():
    import sys

    from collections import deque

    readline = sys.stdin.readline

    n = int(readline())

    li = list(map(int, readline().split()))

    def square(P):

        G = []

        L = deque()

        for i, v in enumerate(P):

            if not L:

                L.append((i, v))

                continue

            if v > L[-1][1]:

                L.append((i, v))

            elif v < L[-1][1]:

                k = i - 1

                while L and v < L[-1][1]:

                    a = L.pop()

                    G.append((k - a[0] + 1) * a[1])

                L.append((a[0], v))

        while L:

            a = L.pop()

            G.append((len(P) - a[0]) * a[1])

        return max(G)

    print((square(li)))


problem_p02328()
