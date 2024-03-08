def problem_p00735():
    p = [1] * 300003

    p[0] = p[1] = 0

    for i in range(2, 300003):

        if p[i] and i % 7 in (1, 6):

            p[2 * i :: i] = [0] * len(p[2 * i :: i])

        else:

            p[i] = 0

    prime = [i for i in range(2, 300003) if p[i]]

    while 1:

        N = int(input())

        if N == 1:
            break

        ans = [p for p in prime if N % p == 0]

        print("%d: %s" % (N, " ".join(map(str, ans))))


problem_p00735()
