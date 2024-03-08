def problem_p03051():
    import sys

    from collections import defaultdict

    from itertools import accumulate

    from operator import xor

    def solve_0(acc):

        MOD = 10**9 + 7

        cnt_0 = 0

        counts = defaultdict(list)

        last_0 = {}

        for a in acc:

            if a == 0:

                cnt_0 += 1

            else:

                cnt = counts[a]

                if len(cnt) == 0:

                    cnt.append(1)

                    last_0[a] = cnt_0

                else:

                    btw_0 = cnt_0 - last_0[a]

                    if btw_0 > 0:

                        cnt.append(btw_0)

                        cnt.append(1)

                        last_0[a] = cnt_0

                    else:

                        cnt[-1] += 1

        ans = pow(2, cnt_0 - 1, MOD)

        for i, cnt in list(counts.items()):

            dp0 = 1

            dp1 = cnt[0]

            for c0, c1 in zip(cnt[1::2], cnt[2::2]):

                dp0 = (dp0 + dp1 * c0) % MOD

                dp1 = (dp1 + dp0 * c1) % MOD

            ans = (ans + dp1) % MOD

        return ans

    def solve_1(acc):

        MOD = 10**9 + 7

        common = acc[-1]

        dp0 = 1

        dp1 = 0

        for a in acc:

            if a == 0:

                dp0 = (dp0 + dp1) % MOD

            elif a == common:

                dp1 = (dp1 + dp0) % MOD

        return dp0

    n = int(eval(input()))

    aaa = list(map(int, input().split()))

    if "PyPy" in sys.version:

        acc = [0] * n

        t = 0

        for i in range(n):

            t = acc[i] = t ^ aaa[i]

    else:

        acc = list(accumulate(aaa, func=xor))

    # print(acc)

    if acc[-1] == 0:

        print((solve_0(acc)))

    else:

        print((solve_1(acc)))


problem_p03051()
