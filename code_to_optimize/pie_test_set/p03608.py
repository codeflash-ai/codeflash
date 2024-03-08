def problem_p03608():
    def main():

        from itertools import permutations

        n, m, r, *x = list(map(int, open(0).read().split()))

        to = x[:r]

        dis = [[float("Inf")] * (n + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):

            dis[i][i] = 0

        for a, b, c in zip(*[iter(x[r:])] * 3):

            dis[a][b] = c

            dis[b][a] = c

        for k in range(1, n + 1):

            for i in range(1, n + 1):

                for j in range(1, n + 1):

                    dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])

        ans = float("Inf")

        for p in permutations(to):

            tmp = 0

            nw = p[0]

            for nx in p[1:]:

                tmp += dis[nw][nx]

                nw = nx

            ans = min(ans, tmp)

        print(ans)

    if __name__ == "__main__":

        main()


problem_p03608()
