def problem_p01751(input_data):
    def solve(a, b, c):

        l = 0
        r = a

        for t in range(114514):

            t = l // 60

            p = 60 * t + c

            if l <= p <= r:

                return p

                exit()

            l = r + b

            r = l + a

        return -1

    a, b, c = list(map(int, input_data.split()))

    solve(a, b, c)
