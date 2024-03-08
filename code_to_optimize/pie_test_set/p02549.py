def problem_p02549():
    def getN():

        return int(eval(input()))

    def getNM():

        return list(map(int, input().split()))

    def getList():

        return list(map(int, input().split()))

    def getArray(intn):

        return [int(eval(input())) for i in range(intn)]

    def input():

        return sys.stdin.readline().rstrip()

    def rand_N(ran1, ran2):

        return random.randint(ran1, ran2)

    def rand_List(ran1, ran2, rantime):

        return [random.randint(ran1, ran2) for i in range(rantime)]

    def rand_ints_nodup(ran1, ran2, rantime):

        ns = []

        while len(ns) < rantime:

            n = random.randint(ran1, ran2)

            if not n in ns:

                ns.append(n)

        return sorted(ns)

    def rand_query(ran1, ran2, rantime):

        r_query = []

        while len(r_query) < rantime:

            n_q = rand_ints_nodup(ran1, ran2, 2)

            if not n_q in r_query:

                r_query.append(n_q)

        return sorted(r_query)

    from sys import exit

    import sys

    sys.setrecursionlimit(1000000000)

    mod = 998244353

    #############

    # Main Code #

    #############

    N, K = getNM()

    que = [getList() for i in range(K)]

    dp = [0] * (N + 1)

    dp[1] = 1

    dp[2] = -1

    for i in range(1, N + 1):

        dp[i] += dp[i - 1]

        dp[i] %= mod

        for l, r in que:

            if i + l <= N:

                dp[i + l] += dp[i]

            if i + r + 1 <= N:

                dp[i + r + 1] -= dp[i]

    print((dp[N] % mod))


problem_p02549()
