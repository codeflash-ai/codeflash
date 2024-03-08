def problem_p03702():
    n, a, b = list(map(int, input().split()))

    h = [int(eval(input())) for i in range(n)]

    h.sort()

    def check(k):

        top = 0

        tmp = h[::]

        for i in range(n):

            amari = tmp[i] - b * k

            if amari > 0:

                top += -(-amari // (a - b))

        return top <= k

    # 二分探索

    left = 0

    right = -(-1 * sum(h) // b)

    while right - left > 1:

        next_ind = (left + right) // 2

        if check(next_ind):

            right = next_ind

        else:

            left = next_ind

    print(right)


problem_p03702()
