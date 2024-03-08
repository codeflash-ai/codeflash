def problem_p00881():
    from collections import Counter

    while True:

        m, n = (int(s) for s in input().split())

        if not m:

            break

        objs = [int(eval(input()), 2) for i in range(n)]

        dp = [[0] * (1 << m) for i in range(1 << m)]

        bits = [1 << i for i in range(m)]

        for mask in reversed(list(range(1 << m))):

            s = Counter(obj & mask for obj in objs)

            for masked, value in list(s.items()):

                if value > 1:

                    dp[mask][masked] = min(
                        max(dp[mask | b][masked], dp[mask | b][masked | b]) + 1
                        for b in bits
                        if not b & mask
                    )

        print((dp[0][0]))


problem_p00881()
