def problem_p00790():
    while 1:

        n = eval(input())

        if n == 0:
            break

        D = list(range(1, 7))

        ans = set(D)

        for i in range(n):

            c = input()[0]

            if c == "s":
                S = [1, 5, 2, 3, 0, 4]

            elif c == "n":
                S = [4, 0, 2, 3, 5, 1]

            elif c == "w":
                S = [3, 1, 0, 5, 4, 2]

            elif c == "e":
                S = [2, 1, 5, 0, 4, 3]

            D = [D[s] for s in S]

        print(D[0])


problem_p00790()
