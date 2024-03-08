def problem_p03948():
    #!/usr/bin/env python3

    def main():

        n, t = list(map(int, input().split()))

        an = list(map(int, input().split()))

        mi = [an[0]]

        se_mi = {0}

        for i in range(1, n):

            if mi[i - 1] > an[i]:

                se_mi = {i}

            elif mi[i - 1] == an[i]:

                se_mi.add(i)

            mi.append(min(mi[i - 1], an[i]))

        ma = 0

        se_ma = set()

        for i in range(1, n):

            be = an[i] - mi[i - 1]

            if be > ma:

                ma = be

                se_ma = {i}

            elif be == ma:

                se_ma.add(i)

        print((len(se_ma)))

    main()


problem_p03948()
