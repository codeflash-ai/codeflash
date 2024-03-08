def problem_p00009():
    def mk_table(n):

        res = [1] * (n + 1)

        res[:2] = [0, 0]

        for i in range(2, n):

            if i**2 > n:

                break

            if res[i] == 1:

                j = 2

                while i * j <= n:

                    res[i * j] = 0

                    j += 1

        return res

    tbl = mk_table(999999)

    try:

        while 1:

            print((len([x for x in tbl[: int(eval(input())) + 1] if x == 1])))

    except Exception:

        pass


problem_p00009()
