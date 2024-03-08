def problem_p02615():
    import numpy as np

    import heapq

    N = int(eval(input()))

    A = np.sort([int(x) for x in input().split(" ")])[::-1]

    q = [(-A[1], (0, 1)), (-A[1], (0, 1))]

    confort = A[0]

    heapq.heapify(q)

    i = 2

    while N > i:

        m = heapq.heappop(q)

        # print(f"{m[1][0]}と{m[1][1]}の間に割り込む. 気持ち良さ {-m[0]}")

        confort -= m[0]

        heapq.heappush(q, (-A[i], (i, m[1][0])))

        heapq.heappush(q, (-A[i], (i, m[1][1])))

        i += 1

    print(confort)


problem_p02615()
