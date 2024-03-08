def problem_p03556():
    import math

    n = int(eval(input()))

    for i in range(n, 0, -1):

        if math.sqrt(i).is_integer():

            print(i)

            break


problem_p03556()
