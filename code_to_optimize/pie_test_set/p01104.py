def problem_p01104():
    ans = []

    while 1:

        n, m = map(int, input().split())

        if n + m == 0:

            break

        C = {0: 0}

        for i in range(n):

            b = int(input(), 2)

            for k, v in dict(C).items():

                if C.get(k ^ b, 0) < v + 1:

                    C[k ^ b] = v + 1

        ans.append(C[0])

    print(*ans, sep="\n")


problem_p01104()
