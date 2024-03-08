def problem_p00031():
    while True:

        try:

            tmp = eval(input())

        except:

            break

        i = 1

        while tmp > 0:

            if tmp % 2 == 1:
                print(i, end=" ")

            tmp /= 2

            i *= 2

        print()


problem_p00031()
