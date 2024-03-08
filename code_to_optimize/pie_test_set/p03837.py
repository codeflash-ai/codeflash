def problem_p03837():
    import queue

    INF = 123456789

    def main():

        n, m = list(map(int, input().split()))

        G = [[] for _ in range(n)]

        for _ in range(m):

            a, b, c = list(map(int, input().split()))

            a -= 1

            b -= 1

            G[a].append((b, c))

            G[b].append((a, c))

        used = set()

        for i in range(n):

            d = [INF for _ in range(n)]

            par = [-1 for _ in range(n)]

            # dijkstra

            que = queue.PriorityQueue()

            d[i] = 0

            que.put((0, i))

            while not que.empty():

                p = que.get()

                v = p[1]

                if d[v] < p[0]:

                    continue

                for to, cost in G[v]:

                    if d[to] > d[v] + cost:

                        d[to] = d[v] + cost

                        par[to] = v

                        que.put((d[to], to))

            for j in range(n):

                if i != j:

                    a, b = j, par[j]

                    if a > b:

                        a, b = b, a

                    used.add((a, b))

        print((m - len(used)))

    if __name__ == "__main__":

        main()


problem_p03837()
