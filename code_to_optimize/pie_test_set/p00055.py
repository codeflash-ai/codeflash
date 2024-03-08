def problem_p00055():
    while True:

        try:

            n = float(input())

            d = [0] * 10

            d[0] = n

            for i in range(1, 10):

                if i % 2 == 0:
                    d[i] = d[i - 1] / 3

                else:
                    d[i] = d[i - 1] * 2

            print(round(sum(d), 8))

        except:

            break


problem_p00055()
