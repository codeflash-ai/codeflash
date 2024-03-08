def problem_p03089():
    import sys

    sys.setrecursionlimit(10**9)

    def rec(arr, l, lst):

        # lstは操作の逆順

        # print(arr, l, lst[::-1])

        if l == 0:

            for p in lst[::-1]:

                print(p)

            exit()

            return

        for i in sorted(list(range(1, l + 1)), reverse=True):

            # 1-indexed

            if arr[i - 1] == i:

                # print(aa, '除去')

                # 最後に入れた数は、何番目と数の大きさが一致する

                rec(arr[: i - 1] + arr[i:], l - 1, lst + [i])

            elif arr[i - 1] > i:

                # 何番目よりも数が大きい

                return

        return

    N = int(eval(input()))

    b = list(map(int, input().split()))

    rec(b, N, [])

    print((-1))


problem_p03089()
