def problem_p03554():
    n = int(eval(input()))

    b = list(map(int, input().split()))

    ope = [[] for i in range(n)]

    Q = int(eval(input()))

    for i in range(Q):

        l, r = list(map(int, input().split()))

        ope[r - 1].append(l - 1)

    res = b.count(0)

    Data = [(-1) ** ((b[i] == 1) + 1) for i in range(n)]

    for i in range(1, n):

        Data[i] += Data[i - 1]

    Data = [0] + Data

    for i in range(n):

        ope[i].sort(reverse=True)

    # N: 処理する区間の長さ

    N = n + 1

    N0 = 2 ** (N - 1).bit_length()

    data = [None] * (2 * N0)

    INF = (-(2**31), -(2**31))

    # 区間[l, r+1)の値をvに書き換える

    # vは(t, value)という値にする (新しい値ほどtは大きくなる)

    def update(l, r, v):

        L = l + N0
        R = r + N0

        while L < R:

            if R & 1:

                R -= 1

                if data[R - 1]:

                    data[R - 1] = max(v, data[R - 1])

                else:

                    data[R - 1] = v

            if L & 1:

                if data[L - 1]:

                    data[L - 1] = max(v, data[L - 1])

                else:

                    data[L - 1] = v

                L += 1

            L >>= 1
            R >>= 1

    # a_iの現在の値を取得

    def _query(k):

        k += N0 - 1

        s = INF

        while k >= 0:

            if data[k]:

                s = max(s, data[k])

            k = (k - 1) // 2

        return s

    # これを呼び出す

    def query(k):

        return _query(k)[1]

    for i in range(n + 1):

        update(i, i + 1, (-Data[i], -Data[i]))

    if ope[0]:

        update(1, 2, (0, 0))

    for i in range(1, n):

        val = query(i)

        update(i + 1, i + 2, (val + Data[i] - Data[i + 1], val + Data[i] - Data[i + 1]))

        for l in ope[i]:

            val = query(l)

            update(l + 1, i + 2, (val, val))

    print((n - (res + query(n) + Data[n])))


problem_p03554()
