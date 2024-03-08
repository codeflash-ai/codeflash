def problem_p00444():
    import sys

    coins = [1, 5, 10, 50, 100, 500]

    for i in sys.stdin:

        if int(i) == 0:

            break

        n = 1000 - int(i)

        c = [i for i in range(n + 1)]

        for i in range(len(coins)):

            coin = coins[i]

            if coin <= n:

                c[coin] = 1

                coins.append(coin)

        for coin in coins:

            for p in range(coin + 1, n + 1):

                minc = c[p]

                if minc == 1:

                    continue

                c[p] = c[p - coin] + 1 if minc > c[p - coin] + 1 else minc

        print((c[n]))


problem_p00444()
