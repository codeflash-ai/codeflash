def problem_p00014():
    import sys

    for n in map(int, sys.stdin):

        s = sum([i * i for i in range(n, 600, n)]) * n

        print(s)


problem_p00014()
