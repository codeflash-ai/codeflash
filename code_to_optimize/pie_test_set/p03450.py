def problem_p03450():
    import sys

    from collections import deque

    def solve():

        input = sys.stdin.readline

        N, M = list(map(int, input().split()))

        Edge = [[] for _ in range(N)]

        for _ in range(M):

            l, r, d = list(map(int, input().split()))

            Edge[l - 1].append((r - 1, d))

            Edge[r - 1].append((l - 1, -1 * d))

        Dist = [10**20] * N

        possible = True

        for i in range(N):

            if Dist[i] == 10**20:

                q = deque()

                q.append((i, 0))

                while q:

                    nn, nd = q.popleft()

                    if Dist[nn] == 10**20:

                        Dist[nn] = nd

                        for ne, add in Edge[nn]:
                            q.append((ne, nd + add))

                    else:

                        if Dist[nn] == nd:
                            continue

                        else:

                            possible = False

                            break

                if not possible:

                    print("No")

                    break

        else:
            print("Yes")

        return 0

    if __name__ == "__main__":

        solve()


problem_p03450()
