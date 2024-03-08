def problem_p00515():
    a = []

    for _ in range(5):

        x = int(eval(input()))

        a.append(x if x >= 40 else 40)

    print((sum(a) // 5))


problem_p00515()
