def problem_p00024():
    import math

    while True:

        try:

            v = eval(input())

            t = v / 9.8

            y = v / 2 * t

            n = math.ceil(y / 5.0) + 1

            print(int(n))

        except:

            break


problem_p00024()
