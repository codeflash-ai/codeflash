def problem_p03091():
    import sys

    sys.setrecursionlimit(10**7)

    INF = 10**18

    MOD = 10**9 + 7

    def LI():
        return [int(x) for x in sys.stdin.readline().split()]

    def LI_():
        return [int(x) - 1 for x in sys.stdin.readline().split()]

    def LF():
        return [float(x) for x in sys.stdin.readline().split()]

    def LS():
        return sys.stdin.readline().split()

    def II():
        return int(sys.stdin.readline())

    def SI():
        return eval(input())

    from collections import defaultdict, deque

    def main():

        N, M = LI()

        edges = defaultdict(set)

        for _ in range(M):

            a, b = LI_()

            edges[a].add(b)

            edges[b].add(a)

        quads = set()

        sixes = 0

        for i, e in list(edges.items()):

            c = len(e)

            if c % 2 == 1:

                return False

            if c >= 6:

                sixes += 1

            if c == 4:

                quads.add(i)

        if sixes > 0:

            return True

        if len(quads) > 2:

            return True

        if len(quads) < 2:

            return False

        s = quads.pop()

        t = quads.pop()

        for v0 in edges[s]:

            q = deque([v0])

            visited = {s, t}

            while len(q) > 0:

                v = q.pop()

                for to in edges[v]:

                    if v != v0 and to == s:

                        return True

                    if to not in visited:

                        q.append(to)

                        visited.add(to)

        return False

    print(("Yes" if main() else "No"))


problem_p03091()
