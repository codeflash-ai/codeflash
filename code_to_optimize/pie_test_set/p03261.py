def problem_p03261():
    ri = lambda: int(input())

    rl = lambda: list(map(int, input().split()))

    rr = lambda N: [ri() for _ in range(N)]

    YN = lambda b: print("YES") if b else print("NO")

    yn = lambda b: print("Yes") if b else print("No")

    OE = lambda x: print("Odd") if x % 2 else print("Even")

    INF = 10**18

    N = ri()

    W = [0] * N

    ans = 1

    for i in range(N):

        W[i] = input()

        if i != 0 and W[i - 1][-1] != W[i][0]:

            ans = 0

    import collections

    Wc = collections.Counter(W).most_common()

    if Wc[0][1] != 1:

        ans = 0

    yn(ans)


problem_p03261()
