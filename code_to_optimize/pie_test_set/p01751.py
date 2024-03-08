def problem_p01751():
    def solve(a, b, c):

        l = 0
        r = a

        for t in range(114514):

            t = l // 60

            p = 60 * t + c

            if l <= p <= r:

                print(p)

                exit()

            l = r + b

            r = l + a

        print((-1))

    a, b, c = list(map(int, input().split()))

    solve(a, b, c)


problem_p01751()
