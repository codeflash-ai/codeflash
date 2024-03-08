def problem_p03031():
    # cs: c nnected switches

    n, m = list(map(int, input().split()))

    # cs: connected switches

    cs = [list(map(int, input().split())) for _ in range(m)]

    ps = list(map(int, input().split()))

    # bit全探索(0: すべてoffは不要)

    ans = 0

    # on: 0:消えてる、1:点いてる

    # 10: 1 2  3   4

    # 0b: 1 10 11, 100..

    for on in range(2**n):

        # bin: バイナリ文字列に変換

        # 3 -> '0b11'

        # rust で右0埋め

        patern = bin(on)[2:].rjust(n, "0")

        ok = True

        for i in range(m):

            ct = 0

            for j in range(cs[i][0]):

                if patern[cs[i][j + 1] - 1] == "1":

                    ct += 1

            if ct % 2 != ps[i]:

                ok = False

        if ok:

            ans += 1

    print(ans)


problem_p03031()
