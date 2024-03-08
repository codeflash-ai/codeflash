def problem_p03324():
    d, n = list(map(int, input().split()))

    for i in range(1, 100000000):

        if i % (100**d) == 0 and i % (100 * 100**d) != 0:

            n -= 1

        if n == 0:

            print(i)

            exit()


problem_p03324()
