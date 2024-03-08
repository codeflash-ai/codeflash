def problem_p00002():
    import sys

    for line in sys.stdin:

        x, y = list(map(int, line.split()))

        print((len(str(x + y))))


problem_p00002()
