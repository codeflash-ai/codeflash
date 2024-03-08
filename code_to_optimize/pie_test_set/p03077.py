def problem_p03077():
    # C

    import math

    n = int(eval(input()))

    a = []

    for i in range(5):

        a.append(int(eval(input())))

    ans = math.ceil((n / min(a) + 5) - 1)

    print(ans)


problem_p03077()
