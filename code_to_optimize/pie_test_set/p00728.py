def problem_p00728():
    while 1:

        n = int(eval(input()))

        if n == 0:
            break

        print(((sum(sorted([int(eval(input())) for _ in range(n)])[1:-1])) // (n - 2)))


problem_p00728()
