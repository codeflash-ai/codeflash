def problem_p00425():
    while 1:

        n = eval(input())

        if n == 0:
            break

        d = list(range(1, 7))

        ans = 1

        for i in range(n):

            c = input()[0]

            if c == "N":
                S = [1, 5, 2, 3, 0, 4]

            elif c == "S":
                S = [4, 0, 2, 3, 5, 1]

            elif c == "E":
                S = [3, 1, 0, 5, 4, 2]

            elif c == "W":
                S = [2, 1, 5, 0, 4, 3]

            elif c == "R":
                S = [0, 2, 4, 1, 3, 5]

            elif c == "L":
                S = [0, 3, 1, 4, 2, 5]

            d = [d[s] for s in S]

            ans += d[0]

        print(ans)


problem_p00425()
