def problem_p03441():
    # 大混乱したので乱択

    from random import randint

    def main():

        import sys

        sys.setrecursionlimit(500000)

        N = int(eval(input()))

        if N == 2:

            print((1))

            exit()

        E = [[] for _ in range(N)]

        for _ in range(N - 1):

            a, b = list(map(int, input().split()))

            E[a].append(b)

            E[b].append(a)

        def dfs(v=0, p=-1, root=0):

            cnt_child = 0

            cnt_false = 0

            res = 0

            for u in E[v]:

                if u != p:

                    cnt_child += 1

                    d = dfs(u, v, root)

                    res += d

                    cnt_false += d == 0

            return res + max(0, cnt_false - 1)

        # for i in range(N):

        #    print(dfs(i, -1, i))

        print((max(dfs(i, -1, i) for i in sorted(list(range(N)), key=lambda x: -len(E[x]))[:5])))

    main()


problem_p03441()
