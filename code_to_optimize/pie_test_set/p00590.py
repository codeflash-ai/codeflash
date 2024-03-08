def problem_p00590():
    import math

    r = 10000

    sqrt = int(math.sqrt(r))

    p = [1 for i in range(r)]

    p[0] = 0

    for i in range(sqrt):

        if p[i]:

            p[2 * i + 1 :: i + 1] = [0 for j in range(2 * i + 1, r, i + 1)]

    while True:

        try:

            n = int(input())

            count = 0

            for i in range(n):

                if p[i] * p[n - i - 1] == 1:

                    count += 1

            print(count)

        except:

            break


problem_p00590()
