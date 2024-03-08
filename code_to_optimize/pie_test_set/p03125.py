def problem_p03125():
    a, b = list(map(int, input().split()))

    if (b % a) == 0:

        print((a + b))

    else:

        print((b - a))


problem_p03125()
