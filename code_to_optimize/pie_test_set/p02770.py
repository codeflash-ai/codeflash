def problem_p02770():
    # 解説放送

    # tee, isliceをやめた

    def main():

        k, q = list(map(int, input().split()))

        (*d,) = list(map(int, input().split()))

        for _ in range(q):

            n, x, m = list(map(int, input().split()))

            (*g,) = [x % m for x in d]

            rep, rest = divmod(n - 1, k)

            last = x + (sum(g) * rep) + sum(g[:rest])

            eq = sum(rep + (1 if i < rest else 0) for i, gg in enumerate(g) if gg == 0)

            ans = (n - 1) - eq - (last // m - x // m)

            print(ans)

    if __name__ == "__main__":

        main()


problem_p02770()
