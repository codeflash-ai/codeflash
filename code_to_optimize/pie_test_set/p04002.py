def problem_p04002():
    from collections import defaultdict

    def gen_bound(x, B):

        if x < 3:
            lb = 2

        else:
            lb = x - 1

        if x > B - 2:
            ub = B - 1

        else:
            ub = x + 1

        return lb, ub + 1

    def solve():

        count = defaultdict(int)

        ans = [0] * 10

        H, W, N = list(map(int, input().split()))

        ans[0] = (H - 2) * (W - 2)

        for i in range(N):

            a, b = list(map(int, input().split()))

            xlb, xub = gen_bound(a, H)

            ylb, yub = gen_bound(b, W)

            for x in range(xlb, xub):

                for y in range(ylb, yub):

                    k = x + (y << 34)

                    bef_c = count[k]

                    ans[bef_c] -= 1

                    ans[bef_c + 1] += 1

                    count[k] += 1

        for c in ans:

            print(c)

    solve()


problem_p04002()
