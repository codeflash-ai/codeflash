def problem_p03880():
    import sys

    def solve():

        file = sys.stdin.readline

        N = int(file())

        A = [None] * N

        mindig = [0 for i in range(30)]

        nim = 0

        for i in range(N):

            a = int(file())

            A[i] = a

            nim ^= a

            reverse = format(a ^ (a - 1), "b").zfill(30)

            for j in range(30):

                if reverse[j] == "1":

                    mindig[j] += 1

                    break

        if nim == 0:
            print((0))

        else:

            nim = format(nim, "b").zfill(30)

            flip = 0

            for i in range(30):

                if nim[i] == "1":

                    if flip % 2 == 0:

                        if mindig[i] == 0:

                            print((-1))

                            break

                        else:
                            flip += 1

                else:

                    if flip % 2 == 1:

                        if mindig[i] == 0:

                            print((-1))

                            break

                        else:
                            flip += 1

            else:
                print(flip)

        return

    if __name__ == "__main__":

        solve()


problem_p03880()
