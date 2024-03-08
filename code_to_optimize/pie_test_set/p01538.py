def problem_p01538():
    import sys

    import math

    Q = sys.stdin.readline()

    for _ in range(int(Q)):

        cnt = 0

        N = int(sys.stdin.readline())

        if N != 0:

            digLeng = int(math.log10(N)) + 1

        else:

            digLeng = 1

        while digLeng > 1:

            Nmax = 0

            for dig in range(digLeng):

                lNum = N // int(math.pow(10, dig))

                rNum = N % int(math.pow(10, dig))

                Nmax = max(Nmax, lNum * rNum)

            N = Nmax

            cnt += 1

            if N != 0:

                digLeng = int(math.log10(N)) + 1

            else:

                digLeng = 1

        print(cnt)


problem_p01538()
