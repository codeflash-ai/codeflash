def problem_p03641():
    import sys

    input = sys.stdin.readline

    import numpy as np

    from heapq import heappush, heappop

    N = int(eval(input()))

    A = np.array(input().split(), dtype=np.int32)

    a_to_i = {a: i for i, a in enumerate(A)}

    # sparse table を使ってRMQ

    # parityの同じところだけを見るようにしておく

    U = len(A).bit_length()

    sp = [None, A]

    for i in range(2, U):

        L = 1 << (i - 1)

        sp.append(np.minimum(sp[-1][:-L], sp[-1][L:]))

    def RMQ(x, y):

        # x番目から偶数番目だけ見て[x,y]での最小値を返す

        d = y - x

        if d <= 1:

            return A[x]

        n = d.bit_length()

        return min(sp[n - 1][x], sp[n - 1][y + 2 - (1 << (n - 1))])

    def F(x, y):

        # 辞書式で最小の2つ組をとる

        # そのあと、今後調べないといけない区間の一覧を返す

        x1 = RMQ(x, y - 1)

        i1 = a_to_i[x1]

        x2 = RMQ(i1 + 1, y)

        i2 = a_to_i[x2]

        task = ((x, y) for x, y in ((x, i1 - 1), (i1 + 1, i2 - 1), (i2 + 1, y)) if y > x)

        return x1, x2, task

    q = [(None, None, ((0, N - 1),))]

    answer = []

    while q:

        x, y, task = heappop(q)

        answer.append(x)

        answer.append(y)

        for left, right in task:

            heappush(q, F(left, right))

    print((" ".join(map(str, answer[2:]))))


problem_p03641()
