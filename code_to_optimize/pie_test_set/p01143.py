def problem_p01143():
    while 1:

        n, m, p = list(map(int, input().split()))

        if n == 0:
            break

        x = [eval(input()) for i in range(n)]

        print((100 - p) * sum(x) / x[m - 1] if x[m - 1] != 0 else 0)


problem_p01143()
