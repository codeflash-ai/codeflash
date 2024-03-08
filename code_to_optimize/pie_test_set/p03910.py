def problem_p03910():
    from itertools import accumulate

    n = int(eval(input()))

    l = list(accumulate(list(range(4473))))

    while True:

        for i, x in enumerate(l):

            if n == x:

                for y in range(1, i + 1):

                    print(y)

                exit()

            elif n < x:

                print(i)

                n -= i

                break


problem_p03910()
