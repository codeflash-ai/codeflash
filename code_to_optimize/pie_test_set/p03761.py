def problem_p03761():
    from collections import defaultdict

    INF = int(1e9 + 7)

    def main():

        Sn = list(open(0).read().split()[1:])

        d = defaultdict(lambda: INF)

        for Si in Sn:

            for i in range(26):

                d[i] = min(d[i], Si.count(chr(ord("a") + i)))

        for i in range(26):

            if d[i] == INF:

                continue

            print(chr(ord("a") + i) * d[i], end="")

    main()


problem_p03761()
