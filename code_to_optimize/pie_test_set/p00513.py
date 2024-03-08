def problem_p00513():
    a = 0

    for _ in [0] * int(eval(input())):

        n = int(eval(input()))

        for i in range(1, int(n**0.5) + 1):

            if (n - i) % (2 * i + 1) == 0:
                break

        else:
            a += 1

    print(a)


problem_p00513()
