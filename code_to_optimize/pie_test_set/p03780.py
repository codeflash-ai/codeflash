def problem_p03780():
    # x < yでyが不要とする。y抜きで[K-y,K)を作れる。

    # このときにxを使っていないのならば、xも不要。

    # xを使っているのならば、xy抜きで[K-x-y,K-x)を作れる。

    # yを加えて、x抜きで[K-x,K)を作れる。

    # よってy不要ならばx不要。

    import numpy as np

    import itertools

    N, K = list(map(int, input().split()))

    A = [int(x) for x in input().split()]

    A.sort()

    # カードは[0,N)-indexed

    def test(i):

        # i番目のカードは不要

        dp = np.zeros(K, dtype=np.bool)

        dp[0] = True

        for a in itertools.chain(A[:i], A[i + 1 :]):

            dp[a:] = np.logical_or(dp[a:], dp[:-a])

        return not dp[-A[i] :].any()

    left = -1  # 不要だと判明

    right = N  # 不要でないと判明

    while right - left > 1:

        mid = (left + right) // 2

        if test(mid):

            left = mid

        else:

            right = mid

    answer = left + 1

    print(answer)


problem_p03780()
