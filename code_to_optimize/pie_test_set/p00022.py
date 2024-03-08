def problem_p00022():
    n = eval(input())

    while n:

        R = [eval(input()) for i in range(n)]

        x = R

        s = max(x)

        while n - 1:

            R = R[1:]

            x = [a + b for a, b in zip(x[:-1], R)]

            s = max(s, max(x))

            n -= 1

        print(s)

        n = eval(input())


problem_p00022()
