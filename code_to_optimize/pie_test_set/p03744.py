def problem_p03744():
    import sys

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    # 各日に対して、関数：残量 -> 最適な熱量 を持つことを考える

    # これは折れ線。傾きが増加する部分 → マージしてまとめる

    # 常に、傾きの減少列の形で持つことになる

    from collections import deque

    N, L = list(map(int, readline().split()))

    m = list(map(int, read().split()))

    TV = list(zip(m, m))

    answer = []

    INF = 10**18

    # 1日目

    t, v = next(TV)

    answer.append(t)

    qT = deque([t])

    qV = deque([v])

    sum_q = t * v  # 今持っている総熱量

    for t, v in TV:

        qT.appendleft(t)
        qV.appendleft(v)

        sum_q += t * v

        # 右側からvだけ捨てる

        rest = v

        while qV[-1] <= rest:

            x = qV.pop()

            rest -= x

            sum_q -= x * qT.pop()

        qV[-1] -= rest

        sum_q -= rest * qT[-1]

        # 左側から傾きの増大部分をマージ

        while len(qT) >= 2:

            t0 = qT[0]
            t1 = qT[1]

            if t0 > t1:

                break

            qT.popleft()

            v0 = qV.popleft()
            v1 = qV[0]

            t2 = (t0 * v0 + t1 * v1) / (v0 + v1)

            qT[0] = t2
            qV[0] += v0

        answer.append(sum_q / L)

    print(("\n".join(map(str, answer))))


problem_p03744()
