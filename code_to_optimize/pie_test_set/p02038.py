def problem_p02038():
    from functools import reduce

    def M(x, y):

        if x == "T" and y == "T":

            return "T"

        elif x == "T" and y == "F":

            return "F"

        elif x == "F" and y == "T":

            return "T"

        else:

            return "T"

    _ = eval(input())

    P = input().split()

    print((reduce(M, P)))


problem_p02038()
