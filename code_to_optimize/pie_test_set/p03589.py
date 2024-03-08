def problem_p03589():
    class EndLoop(Exception):

        pass

    def solve():

        N = int(eval(input()))

        try:

            for h in range(1, 3501):

                for n in range(1, 3501):

                    if 4 * h * n - n * N - h * N > 0:

                        if (h * n * N) % (4 * h * n - n * N - h * N) == 0:

                            w = int((h * n * N) / (4 * h * n - n * N - h * N))

                            print((h, n, int(w)))

                            raise EndLoop

        except:

            pass

    solve()


problem_p03589()
