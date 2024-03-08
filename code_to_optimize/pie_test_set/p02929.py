def problem_p02929():
    import sys

    input = sys.stdin.readline

    from collections import *

    N = int(eval(input()))

    S = input()[:-1]

    if S[0] == "W":

        print((0))

        exit()

    l = [0]

    ans = 1

    MOD = 10**9 + 7

    for i in range(1, 2 * N):

        if S[i] == S[i - 1]:

            l.append(1 ^ l[-1])

        else:

            l.append(l[-1])

    if l.count(0) != N:

        print((0))

        exit()

    zero = 0

    for li in l:

        if li == 0:

            zero += 1

        else:

            ans *= zero

            ans %= MOD

            zero -= 1

    for i in range(1, N + 1):

        ans *= i

        ans %= MOD

    print(ans)


problem_p02929()
