def problem_p02257():
    c = 0

    n = int(eval(input()))

    for _ in range(n):

        x = int(eval(input()))

        for i in range(2, int(x**0.5 + 1)):

            if x % i == 0:
                c += 1
                break

    print((n - c))


problem_p02257()
