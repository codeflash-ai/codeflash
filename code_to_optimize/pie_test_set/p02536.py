def problem_p02536():
    def main():

        n, m = list(map(int, input().split()))

        # マイナスの場合は leader であり、同時にサイズ(*-1)を保持する

        uf = [-1] * (n + 1)

        def uf_leader(a):

            if uf[a] < 0:

                return a

            uf[a] = uf_leader(uf[a])

            return uf[a]

        def uf_unite(a, b):

            ua, ub = uf_leader(a), uf_leader(b)

            if ua == ub:

                return False

            if uf[ua] > uf[ub]:

                a, b = b, a

            uf[ua] += uf[ub]

            uf[ub] = ua

            return True

        def uf_leaders():

            return set([uf_leader(v) for v in range(1, n + 1)])

        # print(uf[1:])

        for _ in range(m):

            a, b = list(map(int, input().split()))

            uf_unite(a, b)

            # print(uf[1:])

        # print(uf_leaders())

        ans = len(uf_leaders()) - 1

        print(ans)

    main()


problem_p02536()
