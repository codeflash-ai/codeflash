def problem_p00158():
    while 1:

        n = int(input())

        if n == 0:
            break

        c = 0

        while n != 1:

            n = [n / 2, n * 3 + 1][n % 2]

            c += 1

        print(c)


problem_p00158()
