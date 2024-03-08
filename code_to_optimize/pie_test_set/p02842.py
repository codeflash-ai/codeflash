def problem_p02842():
    n = int(eval(input()))
    print((([m for m in range(n + 1) if int(m * 1.08) == n] + [":("])[0]))


problem_p02842()
