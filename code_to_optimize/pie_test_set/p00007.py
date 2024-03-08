def problem_p00007():
    from math import ceil

    debt = 100000

    n = int(eval(input()))

    for _ in range(n):

        tmp = ceil(debt * 1.05)

        debt = ceil((tmp / 1000)) * 1000

    print(debt)


problem_p00007()
