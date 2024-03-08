def problem_p02759():
    n = int(eval(input()))

    print((int(n / 2) if n % 2 == 0 else n // 2 + 1))


problem_p02759()
