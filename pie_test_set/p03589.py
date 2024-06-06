def problem_p03589(input_data):
    class EndLoop(Exception):

        pass

    def solve():

        N = int(eval(input_data))

        try:

            for h in range(1, 3501):

                for n in range(1, 3501):

                    if 4 * h * n - n * N - h * N > 0:

                        if (h * n * N) % (4 * h * n - n * N - h * N) == 0:

                            w = int((h * n * N) / (4 * h * n - n * N - h * N))

                            return (h, n, int(w))

                            raise EndLoop

        except:

            pass

    solve()
