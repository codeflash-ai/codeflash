def problem_p02361():
    #!usr/bin/env python3

    from collections import defaultdict

    def main():

        # Read stdin

        fl = input().split(" ")

        V = int(fl[0])

        E = int(fl[1])

        R = int(fl[2])

        # Adjacency list

        G = defaultdict(list)

        for i in range(int(E)):

            s, t, w = [int(x) for x in input().split(" ")]

            G[s].append((t, w))

        # initialized

        d = {}

        INF = float("inf")

        # INF = 10001

        for i in range(V):

            d[i] = INF

        d[R] = 0

        q = [R]

        while q:

            u = q.pop(0)

            for v in G[u]:

                if d[v[0]] > d[u] + v[1]:

                    d[v[0]] = d[u] + v[1]

                    q.append(v[0])

        for k in range(V):

            if d[k] == float("inf"):

                print("INF")

            else:

                print((d[k]))

    if __name__ == "__main__":

        main()


problem_p02361()
