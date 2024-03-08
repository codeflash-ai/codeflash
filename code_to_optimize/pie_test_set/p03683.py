def problem_p03683():
    import math

    n, m = list(map(int, input().split()))

    sum = 0

    if -2 < n - m < 2:

        if n < m:

            n, m = m, n

        sum = math.factorial(n) * math.factorial(m) * 2 ** (m - n + 1)

        sum %= 1000000007

    print(sum)


problem_p03683()
