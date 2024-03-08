def problem_p04000():
    from collections import defaultdict

    def main():

        H, W, N = map(int, input().split())

        d = defaultdict(int)

        for _ in range(N):

            a, b = map(int, input().split())

            for y in range(-2, 1):

                for x in range(-2, 1):

                    n_y, n_x = y + a, x + b

                    if 0 < n_y <= H - 2 and 0 < n_x <= W - 2:

                        d[(n_y, n_x)] += 1

        ans = [0] * 10

        for k, v in d.items():

            ans[v] += 1

        ans[0] = (H - 2) * (W - 2) - sum(ans)

        print(*ans, sep="\n")

    if __name__ == "__main__":

        main()


problem_p04000()
