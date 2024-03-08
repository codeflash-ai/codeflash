def problem_p02381():
    while 1:

        n = int(eval(input()))

        if n == 0:
            break

        s = [int(x) for x in input().split()]

        print(((sum([(m - sum(s) / n) ** 2 for m in s]) / n) ** 0.5))


problem_p02381()
