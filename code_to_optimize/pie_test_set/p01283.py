def problem_p01283():
    import math

    M = 256

    while 1:

        N = int(input())

        if N == 0:
            break

        H = 1e10

        l = list(map(int, input().split()))

        ans = [0, 0, 0]

        for S in range(16):

            for A in range(16):

                for C in range(16):

                    R = S

                    O = [0] * N

                    for i in range(N):

                        R = (A * R + C) % M

                        O[i] = (l[i] + R) % M

                    L = [0] * 256

                    for i in O:
                        L[i] += 1

                    tmp = -sum(float(i) / N * math.log(float(i) / N, 2) for i in L if i != 0)

                    if tmp < H:

                        H = tmp

                        ans = [S, A, C]

        print(" ".join(map(str, ans)))


problem_p01283()
